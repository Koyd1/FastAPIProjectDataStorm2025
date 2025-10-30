from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import File, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from app.preprocessing import clean_column_values

from .analytics import analyze_dataset, prediction_summary, transform_single_record
from .records_statistis import load_and_create_stats_graphs
from .csv_utils import read_uploaded_dataframe
from .forms import build_context, parse_form_to_record
from .supabase_service import ingest_dataset, save_request_with_prediction
import logging
logger = logging.getLogger(__name__)

from fastapi.responses import FileResponse
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import hashlib
import os
import tempfile
from bs4 import BeautifulSoup

def build_vertical_preview_html(dataframe: pd.DataFrame) -> str:
    if dataframe is None or dataframe.empty:
        return '<table class="preview-table preview-table--vertical"></table>'

    row = dataframe.head(1).T.reset_index()
    row.columns = ["Признак", "Значение"]

    def format_label(label: Any) -> str:
        text = str(label).replace("_", " ").strip()
        return text or "-"

    def format_value(value: Any) -> str:
        if pd.isna(value):
            return "-"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    row["Признак"] = row["Признак"].map(format_label)
    row["Значение"] = row["Значение"].map(format_value)

    return row.to_html(index=False, classes=["preview-table", "preview-table--vertical"], border=0)

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
        file_hash = hashlib.sha256(content).hexdigest()

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

        cached_entry = store.dataset_cache.get(file_hash)
        if cached_entry:
            store.last_analysis = cached_entry["analysis"]
            store.current_metadata = cached_entry["metadata"]
            store.last_filename = file.filename
            store.filename_to_hash[file.filename] = file_hash
            cached_entry["filename"] = file.filename

            cached_notifications = cached_entry.get("notifications")
            notifications_list = merge_notifications(cached_notifications)
            notifications_list.append("Использован ранее загруженный файл — показаны сохранённые результаты.")

            return templates.TemplateResponse(
                "index.html",
                build_context(
                    request,
                    store=store,
                    context=context,
                    result=cached_entry["analysis"],
                    error=None,
                    notifications=notifications_list,
                    filename=file.filename,
                ),
            )

        skip_ingest = file.filename in filenames

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

        if not skip_ingest:
            ingest_dataset(store, storage_df, file.filename, source="csv")
        else:
            logger.info("Пропускаем загрузку в Supabase для файла %s: уже существует.", file.filename)

        store.last_analysis = analysis
        store.current_metadata = analysis["metadata"]
        store.last_filename = file.filename
        store.dataset_cache[file_hash] = {
            "analysis": analysis,
            "metadata": analysis["metadata"],
            "filename": file.filename,
            "notifications": analysis.get("notifications"),
        }
        store.filename_to_hash[file.filename] = file_hash

        notifications = merge_notifications(analysis.get("notifications"))
        if skip_ingest:
            notifications.append("Файл с таким именем уже присутствует — результаты обновлены без повторной загрузки.")

        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                store=store,
                context=context,
                result=analysis,
                error=None,
                notifications=notifications,
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
        dataframe, clean_notifications = clean_column_values(dataframe)
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
        prediction["preview_vertical_html"] = build_vertical_preview_html(dataframe)
        prediction["preprocessing_notes"] = prep_notifications
        prediction["source"] = "csv"
        prediction["generated_at"] = datetime.utcnow().isoformat()
        store.last_prediction = prediction

        storage_df = dataframe.copy()
        storage_df["predicted_anomaly"] = prediction["anomaly_label"]
        storage_df["is_suspicious"] = int(prediction["suspicious"])
        storage_df["analysis_run_at"] = datetime.utcnow().isoformat()
        storage_df["source_filename"] = file.filename
        # ingest_dataset(store, storage_df, file.filename, source="single_csv_prediction")

        record_to_save = dataframe.iloc[0].to_dict()
        record_to_save["predicted_class"] = prediction["anomaly_label"]
        record_to_save["class_probability"] = int(prediction["suspicious"])
        
        try:
            save_request_with_prediction(store,record_to_save)
        except Exception as e:
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
                    prediction_error=f"Ошибка при сохранении результата: {e}",
                ),
            )
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
        record_df, clean_notifications = clean_column_values(record_df)
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
        prediction["preview_vertical_html"] = build_vertical_preview_html(record_df)
        prediction["preprocessing_notes"] = prep_notifications
        prediction["source"] = "form"
        prediction["generated_at"] = datetime.utcnow().isoformat()
        store.last_prediction = prediction

        storage_df = record_df.copy()
        storage_df["predicted_anomaly"] = prediction["anomaly_label"]
        storage_df["is_suspicious"] = int(prediction["suspicious"])
        storage_df["analysis_run_at"] = datetime.utcnow().isoformat()
        storage_df["source_filename"] = "form_submission"
        # ingest_dataset(store, storage_df, "form_submission", source="form_submission")
        # Сохраняем запись с предсказанием в Supabase (аналогично /predict)
        record_to_save = record_df.iloc[0].to_dict()
        record_to_save["predicted_class"] = prediction["anomaly_label"]
        record_to_save["class_probability"] = int(prediction["suspicious"])

        try:
            save_request_with_prediction(store, record_to_save)
        except Exception as e:
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
                    prediction_error=f"Ошибка при сохранении результата: {e}",
                    form_values=form_dict,
                ),
            )
        
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
            return text[: max_len - 3] + "..."
        return text

    def ensure_pdf_font() -> Optional[HTMLResponse]:
        font_path = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")
        if not os.path.exists(font_path):
            return HTMLResponse(
                "<h3>Файл шрифта DejaVuSans.ttf не найден. Поместите его в папку /fonts рядом с routes.py</h3>"
            )
        if "DejaVuSans" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
        return None

    def build_dataset_story(result: Dict[str, Any], styles, available_width: float) -> List[Any]:
        story: List[Any] = []
        story.append(Paragraph("📊 Отчёт по загруженному датасету", styles["TitleRu"]))
        story.append(Spacer(1, 16))

        story.append(Paragraph("Основные метрики загруженного набора:", styles["Heading"]))
        metrics_rows = [
            ["Показатель", "Значение"],
            ["Объём строк", truncate_text(str(result.get("records", "-")))],
            ["Количество признаков", truncate_text(str(result.get("columns", "-")))],
        ]
        if result.get("prediction_counts"):
            metrics_rows.append(["Количество классов модели", truncate_text(str(len(result["prediction_counts"])))])

        metrics_table = Table(
            [[Paragraph(str(cell), styles["Body"]) for cell in row] for row in metrics_rows],
            hAlign="LEFT",
            colWidths=[200, 250],
        )
        metrics_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(metrics_table)
        story.append(Spacer(1, 12))

        preview_html = result.get("preview_html")
        if preview_html:
            story.append(Paragraph("Предпросмотр данных (первые 10 строк):", styles["Heading"]))
            soup = BeautifulSoup(preview_html, "html.parser")
            rows = soup.find_all("tr")
            table_data: List[List[Paragraph]] = []
            for row in rows[:11]:
                cols = [truncate_text(cell.get_text(strip=True)) for cell in row.find_all(["th", "td"])]
                table_data.append([Paragraph(col, styles["Body"]) for col in cols])

            if table_data:
                num_cols = len(table_data[0]) or 1
                col_width = max(40, available_width / num_cols)
                total_width = col_width * num_cols
                if total_width > available_width:
                    col_width = available_width / num_cols
                col_widths = [col_width] * num_cols
                preview_table = Table(table_data, repeatRows=1, colWidths=col_widths)
                preview_table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 2),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                            ("TOPPADDING", (0, 0), (-1, -1), 2),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                        ]
                    )
                )
                story.append(preview_table)
                story.append(Spacer(1, 12))

        if result.get("prediction_counts"):
            story.append(PageBreak())
            story.append(Paragraph("Распределение классов модели:", styles["Heading"]))
            class_data = [["Класс", "Количество"]]
            for cls, cnt in result["prediction_counts"].items():
                class_data.append([str(cls), str(cnt)])
            class_table = Table(class_data, hAlign="LEFT", colWidths=[200, 200])
            class_table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                        ("FONTNAME", (0, 0), (-1, -1), "DejaVuSans"),
                        ("FONTSIZE", (0, 0), (-1, -1), 10),
                        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            story.append(class_table)
            story.append(Spacer(1, 12))

        if result.get("average_probabilities"):
            story.append(PageBreak())
            story.append(Paragraph("Средние вероятности по классам:", styles["Heading"]))
            prob_rows = [["Класс", "Средняя вероятность"]]
            for label, prob in result["average_probabilities"].items():
                prob_rows.append([truncate_text(str(label)), f"{prob * 100:.2f}%"])

            prob_table = Table(
                [[Paragraph(str(cell), styles["Body"]) for cell in row] for row in prob_rows],
                hAlign="LEFT",
                colWidths=[200, 250],
            )
            prob_table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(prob_table)
            story.append(Spacer(1, 12))

        return story

    def build_single_prediction_story(prediction: Dict[str, Any], styles, available_width: float) -> List[Any]:
        story: List[Any] = []
        story.append(Paragraph("Индивидуальный прогноз по транзакции", styles["TitleRu"]))
        story.append(Spacer(1, 16))

        summary_rows: List[List[str]] = []
        generated_at = prediction.get("generated_at")
        if generated_at:
            try:
                timestamp = datetime.fromisoformat(generated_at)
                summary_rows.append(["Дата и время прогноза", timestamp.strftime("%d.%m.%Y %H:%M:%S")])
            except ValueError:
                summary_rows.append(["Дата и время прогноза", truncate_text(str(generated_at))])

        source_label = "CSV" if prediction.get("source") == "csv" else "Форма"
        summary_rows.append(["Источник данных", source_label])
        summary_rows.append(["Определённая аномалия", truncate_text(str(prediction.get("anomaly_label", "-")))])

        predicted_probability = prediction.get("predicted_probability")
        if predicted_probability is not None:
            summary_rows.append(["Вероятность выбранного класса", f"{predicted_probability * 100:.2f}%"])

        suspicious = prediction.get("suspicious")
        if suspicious is not None:
            suspicious_text = "Подозрительная транзакция" if suspicious else "Критических признаков не обнаружено"
            summary_rows.append(["Статус транзакции", suspicious_text])

        suspicious_prob = prediction.get("suspicious_probability")
        if suspicious_prob is not None:
            summary_rows.append(["Уровень риска", f"{suspicious_prob * 100:.1f}%"])

        summary_table = Table(
            [[Paragraph(str(cell), styles["Body"]) for cell in row] for row in [["Параметр", "Значение"], *summary_rows]],
            hAlign="LEFT",
            colWidths=[220, 230],
        )
        summary_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(summary_table)
        story.append(Spacer(1, 14))

        prob_items = prediction.get("anomaly_probabilities")
        if prob_items:
            story.append(Paragraph("Вероятности по всем классам:", styles["Heading"]))
            prob_rows = [["Класс", "Вероятность"]]
            for item in prob_items:
                prob_rows.append(
                    [
                        truncate_text(str(item.get("label", "-"))),
                        f"{float(item.get("probability", 0.0)) * 100:.2f}%",
                    ]
                )
            prob_table = Table(
                [[Paragraph(str(cell), styles["Body"]) for cell in row] for row in prob_rows],
                hAlign="LEFT",
                colWidths=[220, 150],
            )
            prob_table.setStyle(
                TableStyle(
                    [
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
                        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(prob_table)
            story.append(Spacer(1, 14))

        preview_html = prediction.get("preview_vertical_html") or prediction.get("preview_html")
        if preview_html:
            story.append(Paragraph("Предпросмотр исходных признаков:", styles["Heading"]))
            soup = BeautifulSoup(preview_html, "html.parser")
            record_pairs: List[List[Paragraph]] = []

            vertical_rows = soup.select("table.preview-table--vertical tbody tr")
            if vertical_rows:
                for row in vertical_rows:
                    cells = row.find_all(["th", "td"])
                    if not cells:
                        continue
                    header_text = truncate_text(cells[0].get_text(strip=True), 40)
                    value_text = "-"
                    if len(cells) > 1:
                        value_text = truncate_text(cells[1].get_text(strip=True), 80)
                    record_pairs.append(
                        [
                            Paragraph(header_text, styles["Body"]),
                            Paragraph(value_text, styles["Body"]),
                        ]
                    )
            else:
                header_cells = soup.select("thead tr th")
                row_cells = soup.select("tbody tr:first-child td")

                if not header_cells:
                    rows = soup.find_all("tr")
                    if rows:
                        potential_headers = rows[0].find_all("th")
                        if potential_headers:
                            header_cells = potential_headers
                            if len(rows) > 1:
                                row_cells = rows[1].find_all("td")

                if header_cells and row_cells:
                    for index, header_cell in enumerate(header_cells):
                        header_text = truncate_text(header_cell.get_text(strip=True), 40)
                        value_text = truncate_text(
                            row_cells[index].get_text(strip=True) if index < len(row_cells) else "-", 80
                        )
                        record_pairs.append(
                            [
                                Paragraph(header_text, styles["Body"]),
                                Paragraph(value_text, styles["Body"]),
                            ]
                        )
                else:
                    # fallback: treat each table cell sequentially
                    first_row = soup.find("tr")
                    if first_row:
                        cells = [
                            truncate_text(cell.get_text(strip=True), 80)
                            for cell in first_row.find_all(["th", "td"])
                        ]
                        for idx, cell_value in enumerate(cells, start=1):
                            record_pairs.append(
                                [
                                    Paragraph(f"Признак {idx}", styles["Body"]),
                                    Paragraph(cell_value or "-", styles["Body"]),
                                ]
                            )

            if record_pairs:
                preview_table = Table(
                    [[Paragraph("Признак", styles["Body"]), Paragraph("Значение", styles["Body"])], *record_pairs],
                    colWidths=[available_width * 0.35, available_width * 0.6],
                    hAlign="LEFT",
                )
                preview_table.setStyle(
                    TableStyle(
                        [
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                story.append(preview_table)
                story.append(Spacer(1, 14))

        prep_notes = prediction.get("preprocessing_notes") or []
        if prep_notes:
            story.append(Paragraph("Примечания предобработки:", styles["Heading"]))
            for note in prep_notes:
                story.append(Paragraph(f"• {truncate_text(str(note), 120)}", styles["Body"]))
            story.append(Spacer(1, 12))

        recommendation: Optional[str] = None
        if suspicious is not None:
            if suspicious:
                recommendation = (
                    "FalcoNS"
                )
            else:
                recommendation = "FalcoNS"
        if recommendation:
            story.append(Paragraph(recommendation, styles["Body"]))

        return story

    @app.get("/export-pdf")
    async def export_pdf(request: Request, scope: str = Query("dataset")):
        font_error = ensure_pdf_font()
        if font_error:
            return font_error

        store = request.app.state.store
        content: Optional[List[Any]] = None
        filename = "dataset_analysis.pdf"

        tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        doc = SimpleDocTemplate(
            tmp_pdf.name,
            pagesize=A4,
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30,
        )

        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name="TitleRu",
                fontName="DejaVuSans",
                fontSize=18,
                leading=22,
                alignment=1,
                textColor=colors.darkblue,
            )
        )
        styles.add(
            ParagraphStyle(
                name="Heading",
                fontName="DejaVuSans",
                fontSize=14,
                leading=18,
                textColor=colors.HexColor("#1f2937"),
            )
        )
        styles.add(ParagraphStyle(name="Body", fontName="DejaVuSans", fontSize=10, leading=14))

        if scope == "single":
            prediction = store.last_prediction
            if not prediction:
                return HTMLResponse("<h3>Нет прогноза для экспорта</h3>")
            content = build_single_prediction_story(prediction, styles, doc.width)
            filename = "single_prediction.pdf"
        else:
            result = store.last_analysis
            if not result:
                return HTMLResponse("<h3>Нет данных для экспорта</h3>")
            content = build_dataset_story(result, styles, doc.width)

        doc.build(content)
        return FileResponse(tmp_pdf.name, filename=filename, media_type="application/pdf")
