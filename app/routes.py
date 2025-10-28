from __future__ import annotations

from datetime import datetime

import pandas as pd
from fastapi import File, Request, UploadFile
from fastapi.responses import HTMLResponse

from .analytics import analyze_dataset, prediction_summary, transform_single_record
from .csv_utils import read_uploaded_dataframe
from .forms import build_context, parse_form_to_record
from .supabase_service import ingest_dataset


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
                    notifications=store.last_analysis["notifications"] if store.last_analysis else [],
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
                notifications=analysis["notifications"],
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
        ingest_dataset(store, storage_df, "form_submission", source="form_submission")

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
