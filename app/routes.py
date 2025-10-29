from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from .analytics import analyze_dataset, prediction_summary, transform_single_record
from .csv_utils import read_uploaded_dataframe
from .forms import build_context, parse_form_to_record
from .supabase_service import ingest_dataset
import logging
logger = logging.getLogger(__name__)

from fastapi.responses import FileResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
import os
import tempfile
from bs4 import BeautifulSoup

def register_routes(app) -> None:
    templates = app.state.templates

    @app.get("/", response_class=HTMLResponse)
    async def upload_page(request: Request) -> HTMLResponse:
        store = app.state.store
        try:
            context = store.ensure_context()
        except RuntimeError as exc:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "result": None,
                    "error": str(exc),
                    "notifications": [],
                    "filename": None,
                    "model_ready": False,
                    "prediction": None,
                    "prediction_error": None,
                    "input_metadata": [],
                    "importance_image": None,
                    "top_features": [],
                    "rf_visuals": {},
                },
            )

        notifications = store.last_analysis["notifications"] if store.last_analysis else []
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                store=store,
                context=context,
                result=store.last_analysis,
                error=None,
                notifications=notifications,
            ),
        )

    
    @app.get("/analyze")
    async def reroute_get_analyze():
        return RedirectResponse(url="/", status_code=302)


    @app.post("/analyze", response_class=HTMLResponse)
    async def analyze_csv(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
        store = app.state.store
        templates = app.state.templates
        try:
            context = store.ensure_context()
        except RuntimeError as exc:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "result": store.last_analysis,
                    "error": str(exc),
                    "notifications": store.last_analysis["notifications"] if store.last_analysis else [],
                    "filename": None,
                    "model_ready": False,
                    "prediction": None,
                    "prediction_error": None,
                    "input_metadata": [],
                    "importance_image": None,
                    "top_features": [],
                    "rf_visuals": {},
                },
            )

        if not file.filename.lower().endswith(".csv"):
            return templates.TemplateResponse(
                "index.html",
                build_context(
                    request,
                    store=store,
                    context=context,
                    result=store.last_analysis,
                    error="Пожалуйста, загрузите файл в формате CSV.",
                    notifications=store.last_analysis["notifications"] if store.last_analysis else [],
                ),
            )

        content = await file.read()

        try:
            dataframe = read_uploaded_dataframe(content)
            dataframe.attrs["source_filename"] = file.filename
        except Exception as exc:
            return templates.TemplateResponse(
                "index.html",
                build_context(
                    request,
                    store=store,
                    context=context,
                    result=store.last_analysis,
                    error=f"Не удалось прочитать CSV: {exc}",
                    notifications=store.last_analysis["notifications"] if store.last_analysis else [],
                ),
            )
        filenames: List[str] = []
        supabase_warning: Optional[str] = None
        client = store.supabase_client
        table_name = store.settings.supabase_table

        if client is not None and table_name:
            try:
                if "." in table_name:
                    schema_name, table_name_only = table_name.split(".", maxsplit=1)
                    table_query = client.schema(schema_name).table(table_name_only)
                else:
                    table_query = client.table(table_name)
                response = table_query.select("filename").execute()
                select_result = getattr(response, "data", None) or []
                filenames = [
                    item.get("filename")
                    for item in select_result
                    if isinstance(item, dict) and item.get("filename")
                ]
                logger.info("Файл %s найден в Supabase таблице %s.", file.filename, table_name)
            except Exception as exc:
                logger.warning("Не удалось получить список файлов в таблице %s: %s", table_name, exc)
                supabase_warning = f"Не удалось получить список файлов из Supabase: {exc}"
        else:
            logger.info("Supabase клиент не настроен — пропускаем проверку уникальности файлов.")

        def merge_notifications(base_notifications: Optional[List[str]]) -> List[str]:
            merged = list(base_notifications or [])
            if supabase_warning:
                merged.append(supabase_warning)
            return merged

        if file.filename not in filenames:
            try:
                analysis, storage_df = analyze_dataset(dataframe, context, store)
            except Exception as exc:
                return templates.TemplateResponse(
                    "index.html",
                    build_context(
                        request,
                        store=store,
                        context=context,
                        result=store.last_analysis,
                        error=f"Ошибка при анализе данных: {exc}",
                        notifications=merge_notifications(
                            store.last_analysis["notifications"] if store.last_analysis else []
                        ),
                    ),
                )
            ingest_dataset(store, storage_df, file.filename, source="csv")
            store.last_analysis = analysis
            store.current_metadata = analysis["metadata"]

            return templates.TemplateResponse(
                "index.html",
                build_context(
                    request,
                    store=store,
                    context=context,
                    result=analysis,
                    error=None,
                    notifications=merge_notifications(analysis["notifications"]),
                    filename=file.filename,
                ),
            )
        else:
            return templates.TemplateResponse(
                "index.html",
                build_context(
                    request,
                    store=store,
                    context=context,
                    result=None,
                    error="File is already uploaded",
                    notifications=merge_notifications(None),
                    filename=file.filename,
                ),
            ) 

    @app.post("/predict", response_class=HTMLResponse)
    async def predict_single(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
        store = app.state.store
        templates = app.state.templates
        try:
            context = store.ensure_context()
        except RuntimeError as exc:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "result": store.last_analysis,
                    "error": str(exc),
                    "notifications": store.last_analysis["notifications"] if store.last_analysis else [],
                    "filename": None,
                    "model_ready": False,
                    "prediction": None,
                    "prediction_error": None,
                    "input_metadata": [],
                    "importance_image": None,
                    "top_features": [],
                    "rf_visuals": {},
                },
            )

        if not file.filename.lower().endswith(".csv"):
            return templates.TemplateResponse(
                "index.html",
                build_context(
                    request,
                    store=store,
                    context=context,
                    result=store.last_analysis,
                    error=None,
                    notifications=store.last_analysis["notifications"] if store.last_analysis else [],
                    prediction=None,
                    prediction_error="Для прогноза загрузите CSV с одной строкой.",
                ),
            )

        content = await file.read()
        try:
            dataframe = read_uploaded_dataframe(content)
            dataframe.attrs["source_filename"] = file.filename
        except Exception as exc:
            return templates.TemplateResponse(
                "index.html",
                build_context(
                    request,
                    store=store,
                    context=context,
                    result=store.last_analysis,
                    error=None,
                    notifications=store.last_analysis["notifications"] if store.last_analysis else [],
                    prediction=None,
                    prediction_error=f"Не удалось прочитать CSV для прогноза: {exc}",
                ),
            )

        if dataframe.empty or len(dataframe) != 1:
            return templates.TemplateResponse(
                "index.html",
                build_context(
                    request,
                    store=store,
                    context=context,
                    result=store.last_analysis,
                    error=None,
                    notifications=store.last_analysis["notifications"] if store.last_analysis else [],
                    prediction=None,
                    prediction_error="CSV должен содержать ровно одну запись.",
                ),
            )

        try:
            features, prep_notifications = transform_single_record(dataframe, context)
        except Exception as exc:
            return templates.TemplateResponse(
                "index.html",
                build_context(
                    request,
                    store=store,
                    context=context,
                    result=store.last_analysis,
                    error=None,
                    notifications=store.last_analysis["notifications"] if store.last_analysis else [],
                    prediction=None,
                    prediction_error=f"Не удалось подготовить запись: {exc}",
                ),
            )

        prediction = prediction_summary(features, context)
        prediction["preview_html"] = dataframe.to_html(index=False, classes="preview-table")
        prediction["preprocessing_notes"] = prep_notifications
        prediction["source"] = "csv"

        storage_df = dataframe.copy()
        storage_df["predicted_anomaly"] = prediction["anomaly_label"]
        storage_df["is_suspicious"] = int(prediction["suspicious"])
        storage_df["analysis_run_at"] = datetime.utcnow().isoformat()
        storage_df["source_filename"] = file.filename
        ingest_dataset(store, storage_df, file.filename, source="single_csv_prediction")

        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                store=store,
                context=context,
                result=store.last_analysis,
                error=None,
                notifications=store.last_analysis["notifications"] if store.last_analysis else [],
                prediction=prediction,
                prediction_error=None,
            ),
        )

    @app.post("/predict-form", response_class=HTMLResponse)
    async def predict_from_form(request: Request) -> HTMLResponse:
        store = app.state.store
        templates = app.state.templates
        try:
            context = store.ensure_context()
        except RuntimeError as exc:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "result": store.last_analysis,
                    "error": str(exc),
                    "notifications": store.last_analysis["notifications"] if store.last_analysis else [],
                    "filename": None,
                    "model_ready": False,
                    "prediction": None,
                    "prediction_error": None,
                    "input_metadata": [],
                    "importance_image": None,
                    "top_features": [],
                    "rf_visuals": {},
                },
            )

        form_data = await request.form()
        form_dict = dict(form_data)

        metadata = store.current_metadata or store.generate_metadata(None)

        try:
            record = parse_form_to_record(form_dict, metadata)
        except Exception as exc:
            return templates.TemplateResponse(
                "index.html",
                build_context(
                    request,
                    store=store,
                    context=context,
                    result=store.last_analysis,
                    error=None,
                    notifications=store.last_analysis["notifications"] if store.last_analysis else [],
                    prediction=None,
                    prediction_error=f"Не удалось обработать значения формы: {exc}",
                    form_values=form_dict,
                ),
            )

        record_df = pd.DataFrame([record])
        try:
            features, prep_notifications = transform_single_record(record_df, context)
        except Exception as exc:
            return templates.TemplateResponse(
                "index.html",
                build_context(
                    request,
                    store=store,
                    context=context,
                    result=store.last_analysis,
                    error=None,
                    notifications=store.last_analysis["notifications"] if store.last_analysis else [],
                    prediction=None,
                    prediction_error=f"Не удалось подготовить запись: {exc}",
                    form_values=form_dict,
                ),
            )

        prediction = prediction_summary(features, context)
        prediction["preview_html"] = record_df.to_html(index=False, classes="preview-table")
        prediction["preprocessing_notes"] = prep_notifications
        prediction["source"] = "form"

        storage_df = record_df.copy()
        storage_df["predicted_anomaly"] = prediction["anomaly_label"]
        storage_df["is_suspicious"] = int(prediction["suspicious"])
        storage_df["analysis_run_at"] = datetime.utcnow().isoformat()
        storage_df["source_filename"] = "form_submission"
        # ingest_dataset(store, storage_df, "form_submission", source="form_submission")

        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                store=store,
                context=context,
                result=store.last_analysis,
                error=None,
                notifications=store.last_analysis["notifications"] if store.last_analysis else [],
                prediction=prediction,
                prediction_error=None,
                form_values=form_dict,
            ),
        )

    MAX_CELL_LENGTH = 50  # максимальное количество символов в ячейке

    def truncate_text(text: str, max_len: int = MAX_CELL_LENGTH) -> str:
        if len(text) > max_len:
            return text[:max_len-3] + "..."
        return text

    @app.get("/export-pdf")
    async def export_pdf(request: Request):
        store = request.app.state.store
        result = store.last_analysis
        if not result:
            return HTMLResponse("<h3>Нет данных для экспорта</h3>")

        font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
        if not os.path.exists(font_path):
            return HTMLResponse("<h3>Файл шрифта DejaVuSans.ttf не найден. Поместите его в папку /fonts рядом с routes.py</h3>")
        pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))

        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(tmp_pdf.name, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
        available_width = doc.width

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="TitleRu", fontName="DejaVuSans", fontSize=18, leading=22, alignment=1, textColor=colors.darkblue))
        styles.add(ParagraphStyle(name="Heading", fontName="DejaVuSans", fontSize=14, leading=18, textColor=colors.darkred))
        styles.add(ParagraphStyle(name="Body", fontName="DejaVuSans", fontSize=10, leading=12))

        story = []

        # === Заголовок ===
        story.append(Paragraph("📊 Отчёт по загруженному датасету", styles["TitleRu"]))
        story.append(Spacer(1, 16))

        # === Основные метрики ===
        story.append(Paragraph("Основные метрики загруженного набора:", styles["Heading"]))
        data = [["Показатель", "Значение"]]
        data.append(["Объём строк", truncate_text(str(result.get("records", "-")))])
        data.append(["Количество признаков", truncate_text(str(result.get("columns", "-")))])
        if result.get("prediction_counts"):
            data.append(["Количество классов модели", truncate_text(str(len(result["prediction_counts"])))])

        metrics_table = Table([[Paragraph(str(c), styles["Body"]) for c in row] for row in data],
                            hAlign="LEFT", colWidths=[200, 250])
        metrics_table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
            ("ALIGN", (0,0), (-1,0), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 12))

        # === Предпросмотр данных (первые 10 строк) ===
        if result.get("preview_html"):
            story.append(Paragraph("Предпросмотр данных (первые 10 строк):", styles["Heading"]))
            soup = BeautifulSoup(result["preview_html"], "html.parser")
            rows = soup.find_all("tr")
            table_data = []
            for row in rows[:11]:  # заголовок + 10 строк
                cols = [truncate_text(c.get_text(strip=True)) for c in row.find_all(["th","td"])]
                cols = [Paragraph(c, styles["Body"]) for c in cols]
                table_data.append(cols)

            if table_data:
                num_cols = len(table_data[0])
                if num_cols == 0:
                    num_cols = 1
                col_width = max(40, available_width / num_cols)
                total_width = col_width * num_cols
                if total_width > available_width:
                    col_width = available_width / num_cols
                col_widths = [col_width] * num_cols
                preview_table = Table(table_data, repeatRows=1, colWidths=col_widths)
                preview_table.setStyle(TableStyle([
                    ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                    ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                    ("VALIGN", (0,0), (-1,-1), "TOP"),
                    ("LEFTPADDING", (0,0), (-1,-1), 2),
                    ("RIGHTPADDING", (0,0), (-1,-1), 2),
                    ("TOPPADDING", (0,0), (-1,-1), 2),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 2),
                ]))
                story.append(preview_table)
                story.append(Spacer(1, 12))

        # Распределение классов
        if result.get("prediction_counts"):
            story.append(PageBreak())
            story.append(Paragraph("Распределение классов модели:", styles["Heading"]))
            class_data = [["Класс", "Количество"]]
            for cls, cnt in result["prediction_counts"].items():
                class_data.append([str(cls), str(cnt)])
            class_table = Table(class_data, hAlign="LEFT", colWidths=[200, 200])
            class_table.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
                ("FONTNAME", (0,0), (-1,-1), "DejaVuSans"),
                ("FONTSIZE", (0,0), (-1,-1), 10),
                ("ALIGN", (0,0), (-1,0), "CENTER"),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ]))
            story.append(class_table)
            story.append(Spacer(1, 12))

        # === Добавляем разрыв страницы перед средними вероятностями ===
        if result.get("average_probabilities"):
            story.append(PageBreak())
            story.append(Paragraph("Средние вероятности по классам:", styles["Heading"]))
            prob_data = [["Класс", "Средняя вероятность"]]
            for label, prob in result["average_probabilities"].items():
                prob_data.append([truncate_text(str(label)), f"{prob*100:.2f}%"])

            prob_table = Table([[Paragraph(str(c), styles["Body"]) for c in row] for row in prob_data],
                            hAlign="LEFT", colWidths=[200, 250])
            prob_table.setStyle(TableStyle([
                ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
                ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
                ("ALIGN", (0,0), (-1,0), "CENTER"),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
            ]))
            story.append(prob_table)
            story.append(Spacer(1, 12))

        doc.build(story)
        return FileResponse(tmp_pdf.name, filename="dataset_analysis.pdf", media_type="application/pdf")
