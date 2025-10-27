import base64
import io
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import pandas as pd
import seaborn as sns
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_PATH = Path("./models/lgbm_anomaly_multi.pkl")

DEFAULT_RECORD: Dict[str, Any] = {
    "issuer_bank": "Moldindconbank",
    "account_type": "Business",
    "card_type": "Debit",
    "product_tier": "Standard",
    "region": "Soroca",
    "urban": 1,
    "age": 51,
    "tenure_months": 10,
    "cluster_id_expected": 4,
    "cluster_name_expected": "MPay_Utilities",
    "channel": "POS",
    "merchant_category": "Utilities/MPay",
    "merchant_country": "MD",
    "currency": "MDL",
    "fx_rate_to_mdl": 1,
    "amount_txn_ccy": 4250,
    "amount_mdl": 4250,
    "card_present": 1,
    "auth_method": "MAGSTRIPE",
    "is_3ds": 0,
    "three_ds_result": "NA",
    "device_type": "mobile_app",
    "device_trust_score": 0.802,
    "ip_risk_score": 0.045,
    "geo_distance_km": 4.6,
    "hour_of_day": 5,
    "day_of_week": 0,
    "is_night": 0,
    "is_weekend": 0,
    "txn_count_1h": 0,
    "txn_amount_1h_mdl": 596.31,
    "txn_count_24h": 0,
    "txn_amount_24h_mdl": 6200.00,
    "merchant_risk_score": 0.245,
    "velocity_risk_score": 0,
    "new_device_flag": 0,
    "cross_border": 0,
    "campaign_q2_2025": 0,
    "amount_log_z": 1.53,
}

CATEGORICAL_COLS = [
    "issuer_bank",
    "region",
    "cluster_name_expected",
    "channel",
    "merchant_category",
    "merchant_country",
    "currency",
    "auth_method",
    "three_ds_result",
    "device_type",
]

ORDINAL_ENCODINGS: Dict[str, Dict[str, int]] = {
    "account_type": {"Individual": 0, "Business": 1},
    "card_type": {"Debit": 0, "Credit": 1},
    "product_tier": {"Standard": 0, "Gold": 1, "Platinum": 2},
}

BOOLEAN_COLUMNS = {
    "card_present",
    "is_3ds",
    "is_night",
    "is_weekend",
    "new_device_flag",
    "cross_border",
    "campaign_q2_2025",
}

REQUIRED_COLUMNS = list(DEFAULT_RECORD.keys())
NUMERIC_COLUMNS = [
    col
    for col, value in DEFAULT_RECORD.items()
    if isinstance(value, (int, float)) and col not in BOOLEAN_COLUMNS
]

PREFIX_TO_COLUMN = {
    "issuer": "issuer_bank",
    "region": "region",
    "cluster": "cluster_name_expected",
    "channel": "channel",
    "merchant": "merchant_category",  # updated later using heuristic
    "currency": "currency",
    "auth": "auth_method",
    "three": "three_ds_result",
    "device": "device_type",
}


@dataclass
class ModelContext:
    model: lgb.LGBMClassifier
    feature_names: List[str]
    classes: List[str]
    importance_df: pd.DataFrame
    importance_image: str
    category_options: Dict[str, List[str]]


app = FastAPI(title="Anomaly Analyzer")
templates = Jinja2Templates(directory="templates")

MODEL_CONTEXT: Optional[ModelContext] = None
LAST_ANALYSIS_RESULT: Optional[Dict[str, Any]] = None
CURRENT_METADATA: List[Dict[str, Any]] = []


def fig_to_base64(fig: plt.Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def compute_feature_importance(model: lgb.LGBMClassifier) -> pd.DataFrame:
    booster = model.booster_
    importance = booster.feature_importance(importance_type="gain")
    features = booster.feature_name()
    df_importance = pd.DataFrame({"feature": features, "importance": importance})
    return df_importance.sort_values("importance", ascending=False).reset_index(drop=True)


def build_feature_importance_figure(importance_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    top_df = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top_df))))
    sns.barplot(data=top_df, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_xlabel("Gain")
    ax.set_ylabel("Feature")
    ax.set_title("Top Features for Anomaly Prediction")
    return fig


def derive_category_options(feature_names: List[str]) -> Dict[str, List[str]]:
    options: Dict[str, set] = {col: set() for col in CATEGORICAL_COLS}
    for feature in feature_names:
        for prefix, column in PREFIX_TO_COLUMN.items():
            token = f"{prefix}_"
            if feature.startswith(token):
                value = feature[len(token) :]
                if prefix == "merchant":
                    if len(value) <= 3 and value.isupper():
                        options["merchant_country"].add(value)
                    else:
                        options["merchant_category"].add(value)
                else:
                    options[column].add(value)
                break
    return {column: sorted(values) for column, values in options.items() if values}


def aggregate_column_importance(importance_df: pd.DataFrame) -> Dict[str, float]:
    importance_map: Dict[str, float] = {col: 0.0 for col in REQUIRED_COLUMNS}
    for _, row in importance_df.iterrows():
        feature = str(row["feature"])
        value = float(row["importance"])
        if feature in importance_map:
            importance_map[feature] += value
            continue
        for prefix, base_column in PREFIX_TO_COLUMN.items():
            token = f"{prefix}_"
            if feature.startswith(token):
                if prefix == "merchant":
                    suffix = feature[len(token) :]
                    if len(suffix) <= 3 and suffix.isupper():
                        column_name = "merchant_country"
                    else:
                        column_name = "merchant_category"
                else:
                    column_name = base_column
                importance_map[column_name] = importance_map.get(column_name, 0.0) + value
                break
    return importance_map


def load_model_context() -> ModelContext:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найден файл модели: {MODEL_PATH.resolve()}")

    model: lgb.LGBMClassifier = joblib.load(MODEL_PATH)
    feature_names = list(model.booster_.feature_name())
    classes = list(model.classes_)
    importance_df = compute_feature_importance(model)
    importance_image = fig_to_base64(build_feature_importance_figure(importance_df))
    category_options = derive_category_options(feature_names)
    return ModelContext(model, feature_names, classes, importance_df, importance_image, category_options)


MODEL_CONTEXT = load_model_context()
COLUMN_IMPORTANCE = aggregate_column_importance(MODEL_CONTEXT.importance_df)
TOP_IMPORTANT_COLUMNS = {
    column
    for column, _ in sorted(COLUMN_IMPORTANCE.items(), key=lambda item: item[1], reverse=True)[:10]
    if COLUMN_IMPORTANCE[column] > 0
}


def ensure_required_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    notifications: List[str] = []
    df_work = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in df_work.columns:
            default_value = DEFAULT_RECORD[col]
            df_work[col] = default_value
            notifications.append(f"Колонка {col} отсутствовала в данных и заполнена значением по умолчанию ({default_value}).")
    return df_work, notifications


def clean_column_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    notifications: List[str] = []
    df_work = df.copy()

    for col in BOOLEAN_COLUMNS:
        if col in df_work.columns:
            df_work[col] = pd.to_numeric(df_work[col], errors="coerce").fillna(DEFAULT_RECORD[col]).astype(int)

    for col in NUMERIC_COLUMNS:
        if col in df_work.columns:
            series = pd.to_numeric(df_work[col], errors="coerce")
            missing = series.isna()
            if missing.any():
                notifications.append(f"Некоторые значения в {col} были некорректны и заменены на значение по умолчанию.")
            series = series.fillna(DEFAULT_RECORD[col])
            df_work[col] = series

    for col in CATEGORICAL_COLS:
        if col in df_work.columns:
            df_work[col] = df_work[col].fillna(DEFAULT_RECORD[col]).astype(str)

    for col, mapping in ORDINAL_ENCODINGS.items():
        if col in df_work.columns:
            mapped = df_work[col].map(mapping)
            missing_mask = mapped.isna()
            if missing_mask.any():
                replacement = mapping.get(DEFAULT_RECORD[col], next(iter(mapping.values())))
                mapped.loc[missing_mask] = replacement
                notifications.append(f"Неизвестные значения в {col} заменены на {DEFAULT_RECORD[col]}.")
            df_work[col] = mapped.astype(int)

    return df_work, notifications


def encode_features(df: pd.DataFrame, categorical_present: List[str]) -> pd.DataFrame:
    encoded = pd.get_dummies(
        df,
        columns=categorical_present,
        prefix=[c.split("_")[0] for c in categorical_present],
        dtype=int,
    )
    return encoded


def align_features(encoded: pd.DataFrame) -> pd.DataFrame:
    aligned = encoded.reindex(columns=MODEL_CONTEXT.feature_names, fill_value=0)
    return aligned


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df_with_defaults, notifications_missing = ensure_required_columns(df)
    df_clean, notifications_clean = clean_column_values(df_with_defaults)
    categorical_present = [col for col in CATEGORICAL_COLS if col in df_clean.columns]
    encoded = encode_features(df_clean, categorical_present)
    aligned = align_features(encoded)
    return aligned, notifications_missing + notifications_clean


def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    working_df = df.copy()
    y_true: Optional[pd.Series] = None
    if "anomaly_type" in working_df.columns:
        y_true = working_df["anomaly_type"].astype(str)
        working_df = working_df.drop(columns=["anomaly_type"])

    features, notifications = preprocess_features(working_df)
    predictions = MODEL_CONTEXT.model.predict(features)
    probabilities = MODEL_CONTEXT.model.predict_proba(features)

    preview_df = df.copy()
    preview_df["predicted_anomaly"] = predictions
    preview_df["is_suspicious"] = (preview_df["predicted_anomaly"] != "NORMAL").astype(int)

    report_text: Optional[str] = None
    confusion_image: Optional[str] = None
    if y_true is not None:
        report_text = classification_report(y_true, predictions, digits=3, zero_division=0)
        matrix = confusion_matrix(y_true, predictions, labels=MODEL_CONTEXT.classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=MODEL_CONTEXT.classes,
            yticklabels=MODEL_CONTEXT.classes,
            ax=ax,
        )
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_title("Confusion Matrix — anomaly_type")
        confusion_image = fig_to_base64(fig)
    else:
        notifications.append("Колонка anomaly_type отсутствует, метрики точности не рассчитаны.")

    counts = pd.Series(predictions).value_counts().sort_values(ascending=False)
    probability_df = pd.DataFrame(probabilities, columns=MODEL_CONTEXT.classes)
    avg_probabilities = probability_df.mean().sort_values(ascending=False)

    metadata_source = df.copy()
    for col, default in DEFAULT_RECORD.items():
        if col in metadata_source.columns:
            metadata_source[col] = metadata_source[col].fillna(default)
        else:
            metadata_source[col] = default
    metadata_source = metadata_source[REQUIRED_COLUMNS].copy()
    metadata = generate_metadata(metadata_source)

    return {
        "preview_html": preview_df.head(10).to_html(index=False, classes="preview-table"),
        "notifications": notifications,
        "classification_report": report_text,
        "confusion_image": confusion_image,
        "prediction_counts": counts.to_dict(),
        "average_probabilities": avg_probabilities.to_dict(),
        "records": len(df),
        "columns": df.shape[1],
        "metadata": metadata,
    }


def transform_single_record(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    aligned, notifications = preprocess_features(df)
    return aligned, notifications


def prediction_summary(features: pd.DataFrame) -> Dict[str, Any]:
    proba = MODEL_CONTEXT.model.predict_proba(features)[0]
    label = MODEL_CONTEXT.model.predict(features)[0]
    probability_table = [
        {"label": cls, "probability": float(prob)}
        for cls, prob in sorted(zip(MODEL_CONTEXT.classes, proba), key=lambda item: item[1], reverse=True)
    ]
    class_index = list(MODEL_CONTEXT.classes).index(label)
    predicted_probability = float(proba[class_index])
    if "NORMAL" in MODEL_CONTEXT.classes:
        normal_index = list(MODEL_CONTEXT.classes).index("NORMAL")
        suspicious_probability = float(1.0 - proba[normal_index])
    else:
        suspicious_probability = float(1.0 - predicted_probability)
    return {
        "anomaly_label": label,
        "anomaly_probabilities": probability_table,
        "suspicious": label != "NORMAL",
        "suspicious_probability": suspicious_probability,
        "predicted_probability": predicted_probability,
    }


def generate_metadata(df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    reference_df = df if df is not None and not df.empty else pd.DataFrame([DEFAULT_RECORD])
    metadata: List[Dict[str, Any]] = []

    for order_index, column in enumerate(REQUIRED_COLUMNS):
        importance_score = COLUMN_IMPORTANCE.get(column, 0.0)
        entry: Dict[str, Any] = {
            "name": column,
            "order_index": order_index,
            "importance_score": importance_score,
            "is_top_feature": column in TOP_IMPORTANT_COLUMNS,
        }

        if column in BOOLEAN_COLUMNS:
            default_value = int(reference_df[column].mode(dropna=True).iloc[0]) if column in reference_df else DEFAULT_RECORD[column]
            entry.update(
                {
                    "input_type": "select",
                    "python_type": "bool",
                    "default": str(default_value),
                    "choices": [
                        {"value": "1", "label": "Да"},
                        {"value": "0", "label": "Нет"},
                    ],
                }
            )
        elif column in ORDINAL_ENCODINGS:
            default_value = reference_df[column].mode(dropna=True).iloc[0] if column in reference_df else DEFAULT_RECORD[column]
            choices = [
                {"value": key, "label": key}
                for key in ORDINAL_ENCODINGS[column].keys()
            ]
            entry.update(
                {
                    "input_type": "select",
                    "python_type": "string",
                    "default": str(default_value),
                    "choices": choices,
                }
            )
        elif column in CATEGORICAL_COLS:
            values = set()
            if column in reference_df:
                values.update(str(val) for val in reference_df[column].dropna().unique())
            values.update(MODEL_CONTEXT.category_options.get(column, []))
            values.add(str(DEFAULT_RECORD[column]))
            choices = [{"value": val, "label": val} for val in sorted(values, key=str)]
            default_value = reference_df[column].mode(dropna=True).iloc[0] if column in reference_df else DEFAULT_RECORD[column]
            entry.update(
                {
                    "input_type": "select",
                    "python_type": "string",
                    "default": str(default_value),
                    "choices": choices,
                }
            )
        elif column in NUMERIC_COLUMNS:
            raw_series = reference_df[column] if column in reference_df else pd.Series([DEFAULT_RECORD[column]])
            numeric_series = pd.to_numeric(raw_series, errors="coerce")
            if numeric_series.isna().all():
                numeric_series = pd.Series([DEFAULT_RECORD[column]], dtype=float)
            median_value = float(numeric_series.median())
            suggestions: List[str] = []
            if len(numeric_series) > 1:
                quantiles = numeric_series.quantile([0, 0.25, 0.5, 0.75, 1.0]).dropna().unique()
                for val in quantiles:
                    try:
                        number = float(val)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(number):
                        continue
                    suggestions.append(f"{number:.6g}")
            integer_flags = numeric_series.dropna().map(lambda x: float(x).is_integer())
            if not integer_flags.empty and integer_flags.all():
                python_type = "int"
                step = "1"
            else:
                python_type = "float"
                step = "0.01"
            entry.update(
                {
                    "input_type": "number",
                    "python_type": python_type,
                    "default": f"{median_value:.6g}",
                    "step": step,
                    "datalist_id": f"{column}-options",
                    "suggestions": suggestions,
                }
            )
        else:
            default_value = reference_df[column].mode(dropna=True).iloc[0] if column in reference_df else DEFAULT_RECORD[column]
            entry.update(
                {
                    "input_type": "text",
                    "python_type": "string",
                    "default": str(default_value),
                    "datalist_id": f"{column}-options",
                    "suggestions": [],
                }
            )

        metadata.append(entry)

    metadata.sort(key=lambda item: item["order_index"])
    return metadata


CURRENT_METADATA = generate_metadata(pd.DataFrame([DEFAULT_RECORD]))


def build_context(
    request: Request,
    *,
    result: Optional[Dict[str, Any]],
    error: Optional[str],
    notifications: List[str],
    filename: Optional[str] = None,
    prediction: Optional[Dict[str, Any]] = None,
    prediction_error: Optional[str] = None,
    form_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata_for_template: List[Dict[str, Any]] = []
    for field in CURRENT_METADATA:
        field_copy = deepcopy(field)
        desired_value = field_copy.get("default", "")
        if form_values and field_copy["name"] in form_values:
            desired_value = str(form_values[field_copy["name"]])

        if field_copy["input_type"] == "select":
            for choice in field_copy["choices"]:
                choice["selected"] = str(choice["value"]) == str(desired_value)
            field_copy["value"] = desired_value
        else:
            field_copy["value"] = desired_value

        metadata_for_template.append(field_copy)

    return {
        "request": request,
        "result": result,
        "error": error,
        "notifications": notifications,
        "filename": filename,
        "model_ready": MODEL_CONTEXT is not None,
        "prediction": prediction,
        "prediction_error": prediction_error,
        "input_metadata": metadata_for_template,
        "importance_image": MODEL_CONTEXT.importance_image,
        "top_features": MODEL_CONTEXT.importance_df.head(15).to_dict("records"),
    }


def convert_form_value(raw_value: str, python_type: str) -> Any:
    if raw_value is None or raw_value == "":
        return None

    lower_value = raw_value.lower()
    if lower_value == "nan":
        return None

    if python_type == "float":
        value = float(raw_value)
        if math.isnan(value):
            return None
        return value
    if python_type == "int":
        value = float(raw_value)
        if math.isnan(value):
            return None
        return int(value)
    if python_type == "bool":
        return 1 if lower_value in {"1", "true", "yes", "да"} else 0
    return raw_value


def parse_form_to_record(form_data: Dict[str, str], metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    record: Dict[str, Any] = {}
    for field in metadata:
        name = field["name"]
        python_type = field["python_type"]
        raw_value = form_data.get(name)
        if raw_value is None or raw_value == "":
            default_value = field.get("default", "")
            record[name] = convert_form_value(str(default_value), python_type) if default_value != "" else None
        else:
            record[name] = convert_form_value(raw_value, python_type)
    return record


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request) -> HTMLResponse:
    notifications = LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else []
    return templates.TemplateResponse(
        "index.html",
        build_context(
            request,
            result=LAST_ANALYSIS_RESULT,
            error=None,
            notifications=notifications,
        ),
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze_csv(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    global LAST_ANALYSIS_RESULT, CURRENT_METADATA  # noqa: PLW0602

    if not file.filename.lower().endswith(".csv"):
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                result=LAST_ANALYSIS_RESULT,
                error="Пожалуйста, загрузите файл в формате CSV.",
                notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
            ),
        )

    content = await file.read()

    try:
        dataframe = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                result=LAST_ANALYSIS_RESULT,
                error=f"Не удалось прочитать CSV: {exc}",
                notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
            ),
        )

    try:
        analysis = analyze_dataset(dataframe)
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                result=LAST_ANALYSIS_RESULT,
                error=f"Ошибка при анализе данных: {exc}",
                notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
            ),
        )

    LAST_ANALYSIS_RESULT = analysis
    CURRENT_METADATA = analysis["metadata"]

    return templates.TemplateResponse(
        "index.html",
        build_context(
            request,
            result=analysis,
            error=None,
            notifications=analysis["notifications"],
            filename=file.filename,
        ),
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_single(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    if MODEL_CONTEXT is None:
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                result=LAST_ANALYSIS_RESULT,
                error="Модель не загружена.",
                notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
            ),
        )

    if not file.filename.lower().endswith(".csv"):
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                result=LAST_ANALYSIS_RESULT,
                error=None,
                notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
                prediction=None,
                prediction_error="Для прогноза загрузите CSV с одной строкой.",
            ),
        )

    content = await file.read()
    try:
        dataframe = pd.read_csv(io.BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                result=LAST_ANALYSIS_RESULT,
                error=None,
                notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
                prediction=None,
                prediction_error=f"Не удалось прочитать CSV для прогноза: {exc}",
            ),
        )

    if dataframe.empty or len(dataframe) != 1:
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                result=LAST_ANALYSIS_RESULT,
                error=None,
                notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
                prediction=None,
                prediction_error="CSV должен содержать ровно одну запись.",
            ),
        )

    try:
        features, prep_notifications = transform_single_record(dataframe)
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                result=LAST_ANALYSIS_RESULT,
                error=None,
                notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
                prediction=None,
                prediction_error=f"Не удалось подготовить запись: {exc}",
            ),
        )

    prediction = prediction_summary(features)
    prediction["preview_html"] = dataframe.to_html(index=False, classes="preview-table")
    prediction["preprocessing_notes"] = prep_notifications

    return templates.TemplateResponse(
        "index.html",
        build_context(
            request,
            result=LAST_ANALYSIS_RESULT,
            error=None,
            notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
            prediction=prediction,
            prediction_error=None,
        ),
    )


@app.post("/predict-form", response_class=HTMLResponse)
async def predict_from_form(request: Request) -> HTMLResponse:
    if MODEL_CONTEXT is None:
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                result=LAST_ANALYSIS_RESULT,
                error="Модель не загружена.",
                notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
            ),
        )

    form_data = await request.form()
    form_dict = dict(form_data)
    try:
        record = parse_form_to_record(form_dict, CURRENT_METADATA)
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                result=LAST_ANALYSIS_RESULT,
                error=None,
                notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
                prediction=None,
                prediction_error=f"Не удалось обработать значения формы: {exc}",
                form_values=form_dict,
            ),
        )

    record_df = pd.DataFrame([record])
    try:
        features, prep_notifications = transform_single_record(record_df)
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            "index.html",
            build_context(
                request,
                result=LAST_ANALYSIS_RESULT,
                error=None,
                notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
                prediction=None,
                prediction_error=f"Не удалось подготовить запись: {exc}",
                form_values=form_dict,
            ),
        )

    prediction = prediction_summary(features)
    prediction["preview_html"] = record_df.to_html(index=False, classes="preview-table")
    prediction["preprocessing_notes"] = prep_notifications
    prediction["source"] = "form"

    return templates.TemplateResponse(
        "index.html",
        build_context(
            request,
            result=LAST_ANALYSIS_RESULT,
            error=None,
            notifications=LAST_ANALYSIS_RESULT["notifications"] if LAST_ANALYSIS_RESULT else [],
            prediction=prediction,
            prediction_error=None,
            form_values=form_dict,
        ),
    )
