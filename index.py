from __future__ import annotations

import base64
import csv
import io
import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


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
    "merchant": "merchant_category",
    "currency": "currency",
    "auth": "auth_method",
    "three": "three_ds_result",
    "device": "device_type",
}

SUPPORTED_ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "utf-8", "cp1251", "latin-1")
SEPARATOR_CANDIDATES: Tuple[Optional[str], ...] = (None, ",", ";", "\t", "|")


def _int_from_env(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s: %s. Using default %s.", key, value, default)
        return default


@dataclass
class Settings:
    base_dir: Path = Path(__file__).resolve().parent
    model_filename: str = field(default_factory=lambda: os.getenv("MODEL_FILENAME", "lgbm_anomaly_multi.pkl"))
    metadata_filename: str = field(default_factory=lambda: os.getenv("MODEL_METADATA_FILENAME", "lgbm_anomaly_meta.pkl"))
    template_dir_env: Optional[str] = field(default_factory=lambda: os.getenv("TEMPLATE_DIR"))
    preview_rows: int = field(default_factory=lambda: _int_from_env("PREVIEW_ROWS", 10))
    top_feature_badge_count: int = field(default_factory=lambda: _int_from_env("TOP_FEATURE_BADGE_COUNT", 10))
    top_feature_list_count: int = field(default_factory=lambda: _int_from_env("TOP_FEATURE_LIST_COUNT", 15))
    model_dir: Path = field(init=False)
    model_path: Path = field(init=False)
    metadata_path: Path = field(init=False)
    templates_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.model_dir = Path(os.getenv("MODEL_DIR", self.base_dir / "models"))
        self.model_path = Path(os.getenv("MODEL_PATH", self.model_dir / self.model_filename))
        self.metadata_path = Path(os.getenv("MODEL_METADATA_PATH", self.model_dir / self.metadata_filename))
        template_dir = self.template_dir_env or str(self.base_dir / "templates")
        self.templates_dir = Path(template_dir)


@dataclass
class ModelContext:
    model: lgb.LGBMClassifier
    feature_names: List[str]
    classes: List[str]
    importance_df: pd.DataFrame
    importance_image: str
    category_options: Dict[str, List[str]]
    metadata: Dict[str, Any]


@dataclass
class AppStore:
    settings: Settings
    model_context: Optional[ModelContext] = None
    column_importance: Dict[str, float] = field(default_factory=dict)
    top_columns: Set[str] = field(default_factory=set)
    last_analysis: Optional[Dict[str, Any]] = None
    current_metadata: List[Dict[str, Any]] = field(default_factory=list)

    def set_context(self, context: ModelContext) -> None:
        self.model_context = context
        self.column_importance = aggregate_column_importance(context.importance_df)
        self.top_columns = compute_top_columns(self.column_importance, self.settings.top_feature_badge_count)

    def ensure_context(self) -> ModelContext:
        if self.model_context is None:
            raise RuntimeError("Model context is not loaded.")
        return self.model_context


settings = Settings()
app = FastAPI(title="Anomaly Analyzer")
templates = Jinja2Templates(directory=str(settings.templates_dir))
app.state.settings = settings
app.state.templates = templates
app.state.store = AppStore(settings=settings)


@app.on_event("startup")
async def on_startup() -> None:
    store: AppStore = app.state.store
    try:
        context = load_model_context(settings)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to load model context: %s", exc)
        raise

    store.set_context(context)
    store.current_metadata = generate_metadata(None, context, store)
    logger.info("Model loaded: %s", settings.model_path)


def get_store(request: Request) -> AppStore:
    return request.app.state.store


def get_templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates


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
    fig.tight_layout()
    return fig


def derive_category_options(feature_names: List[str]) -> Dict[str, List[str]]:
    options: Dict[str, Set[str]] = {col: set() for col in CATEGORICAL_COLS}
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


def compute_top_columns(importance_map: Dict[str, float], top_n: int) -> Set[str]:
    sorted_cols = sorted(
        ((col, score) for col, score in importance_map.items() if score > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    return {col for col, _ in sorted_cols[:top_n]}


def load_metadata(settings: Settings) -> Dict[str, Any]:
    if not settings.metadata_path.exists():
        logger.warning("Metadata file not found: %s", settings.metadata_path)
        return {}
    try:
        metadata = joblib.load(settings.metadata_path)
        if not isinstance(metadata, dict):
            logger.warning("Metadata file %s does not contain a dict.", settings.metadata_path)
            return {}
        return metadata
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load metadata %s: %s", settings.metadata_path, exc)
        return {}


def load_model_context(settings: Settings) -> ModelContext:
    if not settings.model_path.exists():
        raise FileNotFoundError(f"Не найден файл модели: {settings.model_path}")

    model: lgb.LGBMClassifier = joblib.load(settings.model_path)
    metadata = load_metadata(settings)

    feature_names = metadata.get("features") if isinstance(metadata.get("features"), list) else None
    if not feature_names:
        feature_names = list(model.booster_.feature_name())

    classes = metadata.get("classes") if isinstance(metadata.get("classes"), list) else None
    if not classes:
        classes = list(model.classes_)

    importance_df = compute_feature_importance(model)
    importance_image = fig_to_base64(build_feature_importance_figure(importance_df))
    category_options = derive_category_options(feature_names)

    return ModelContext(
        model=model,
        feature_names=feature_names,
        classes=classes,
        importance_df=importance_df,
        importance_image=importance_image,
        category_options=category_options,
        metadata=metadata,
    )


def _decode_csv_content(content: bytes) -> str:
    last_error: Optional[Exception] = None
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    raise ValueError("Не удалось декодировать CSV. Используйте UTF-8 или CP1251.") from last_error


def normalize_csv_text(text: str) -> str:
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    semicolon_count = text.count(";")
    comma_count = text.count(",")
    tab_count = text.count("\t")
    pipe_count = text.count("|")

    most_common = max(semicolon_count, comma_count, tab_count, pipe_count)
    if most_common == 0:
        return text

    if semicolon_count == most_common and semicolon_count >= 2:
        # Строки с ; иногда содержат \t перед последним столбцом — нормализуем.
        return text.replace("\t", ";")

    if tab_count == most_common and tab_count >= 2:
        return text.replace(";", "\t")

    return text


def read_uploaded_dataframe(content: bytes) -> pd.DataFrame:
    decoded = _decode_csv_content(content)
    normalized = normalize_csv_text(decoded)
    header_line = normalized.splitlines()[0] if normalized else ""

    attempts: List[Tuple[str, Dict[str, Any]]] = []
    seen: Set[Tuple[Optional[str], Optional[str]]] = set()

    def add_attempt(label: str, *, sep: Optional[str] = None, engine: Optional[str] = None) -> None:
        key = (sep, engine)
        if key in seen:
            return
        seen.add(key)
        kwargs: Dict[str, Any] = {}
        if sep is not None:
            kwargs["sep"] = sep
        if engine is not None:
            kwargs["engine"] = engine
        attempts.append((label, kwargs))

    try:
        dialect = csv.Sniffer().sniff(normalized[:4096], delimiters=[",", ";", "\t", "|"])
        add_attempt(f"sniffer({dialect.delimiter!r})", sep=dialect.delimiter)
    except csv.Error:
        add_attempt("auto", sep=None, engine="python")

    for sep in (",", ";", "\t", "|"):
        add_attempt(f"sep={sep!r}", sep=sep)
    add_attempt("auto-python", sep=None, engine="python")
    add_attempt("default", sep=None, engine=None)

    errors: List[str] = []
    for label, kwargs in attempts:
        try:
            dataframe = pd.read_csv(io.StringIO(normalized), **kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {exc}")
            continue

        if dataframe.empty and dataframe.columns.size == 0:
            errors.append(f"{label}: пустой результат")
            continue

        sep = kwargs.get("sep")
        if sep and dataframe.columns.size == 1 and header_line.count(sep) >= 1:
            errors.append(f"{label}: предполагаемый разделитель {sep!r} не сработал (получена 1 колонка)")
            continue

        if not sep and dataframe.columns.size == 1:
            # Если автоопределение не смогло разделить столбцы, пробуем другие кандидаты.
            if any(symbol in header_line for symbol in (",", ";", "\t", "|")):
                errors.append(f"{label}: auto не смог разделить столбцы")
                continue

        return dataframe

    message = "Не удалось прочитать CSV. Проверьте разделители (, ; табуляция) и структуру файла."
    if errors:
        details = "; ".join(errors[:3])
        message = f"{message} Подробности: {details}"
    raise ValueError(message)


def ensure_required_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    notifications: List[str] = []
    df_work = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in df_work.columns:
            default_value = DEFAULT_RECORD[col]
            df_work[col] = default_value
            notifications.append(
                f"Колонка {col} отсутствовала в данных и заполнена значением по умолчанию ({default_value})."
            )
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
                notifications.append(
                    f"Некоторые значения в {col} были некорректны и заменены на значение по умолчанию."
                )
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
    if not categorical_present:
        return df
    encoded = pd.get_dummies(
        df,
        columns=categorical_present,
        prefix=[c.split("_")[0] for c in categorical_present],
        dtype=int,
    )
    return encoded


def align_features(encoded: pd.DataFrame, context: ModelContext) -> pd.DataFrame:
    return encoded.reindex(columns=context.feature_names, fill_value=0)


def preprocess_features(df: pd.DataFrame, context: ModelContext) -> Tuple[pd.DataFrame, List[str]]:
    df_with_defaults, notifications_missing = ensure_required_columns(df)
    df_clean, notifications_clean = clean_column_values(df_with_defaults)
    categorical_present = [col for col in CATEGORICAL_COLS if col in df_clean.columns]
    encoded = encode_features(df_clean, categorical_present)
    aligned = align_features(encoded, context)
    return aligned, notifications_missing + notifications_clean


def analyze_dataset(
    df: pd.DataFrame,
    context: ModelContext,
    store: AppStore,
) -> Dict[str, Any]:
    working_df = df.copy()
    y_true: Optional[pd.Series] = None
    if "anomaly_type" in working_df.columns:
        y_true = working_df["anomaly_type"].astype(str)
        working_df = working_df.drop(columns=["anomaly_type"])

    features, notifications = preprocess_features(working_df, context)
    predictions = context.model.predict(features)
    probabilities = context.model.predict_proba(features)

    preview_df = df.copy()
    preview_df["predicted_anomaly"] = predictions
    preview_df["is_suspicious"] = (preview_df["predicted_anomaly"] != "NORMAL").astype(int)

    report_text: Optional[str] = None
    confusion_image: Optional[str] = None
    if y_true is not None:
        report_text = classification_report(y_true, predictions, digits=3, zero_division=0)
        matrix = confusion_matrix(y_true, predictions, labels=context.classes)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=context.classes,
            yticklabels=context.classes,
            ax=ax,
        )
        ax.set_xlabel("Предсказанный класс")
        ax.set_ylabel("Истинный класс")
        ax.set_title("Confusion Matrix — anomaly_type")
        confusion_image = fig_to_base64(fig)
    else:
        notifications.append("Колонка anomaly_type отсутствует, метрики точности не рассчитаны.")

    counts = pd.Series(predictions).value_counts().sort_values(ascending=False)
    probability_df = pd.DataFrame(probabilities, columns=context.classes)
    avg_probabilities = probability_df.mean().sort_values(ascending=False)

    metadata_source = df.copy()
    for col, default in DEFAULT_RECORD.items():
        if col in metadata_source.columns:
            metadata_source[col] = metadata_source[col].fillna(default)
        else:
            metadata_source[col] = default
    metadata_source = metadata_source[REQUIRED_COLUMNS].copy()
    metadata = generate_metadata(metadata_source, context, store)

    return {
        "preview_html": preview_df.head(store.settings.preview_rows).to_html(index=False, classes="preview-table"),
        "notifications": notifications,
        "classification_report": report_text,
        "confusion_image": confusion_image,
        "prediction_counts": counts.to_dict(),
        "average_probabilities": avg_probabilities.to_dict(),
        "records": len(df),
        "columns": df.shape[1],
        "metadata": metadata,
    }


def transform_single_record(df: pd.DataFrame, context: ModelContext) -> Tuple[pd.DataFrame, List[str]]:
    aligned, notifications = preprocess_features(df, context)
    return aligned, notifications


def prediction_summary(features: pd.DataFrame, context: ModelContext) -> Dict[str, Any]:
    proba = context.model.predict_proba(features)[0]
    label = context.model.predict(features)[0]
    probability_table = [
        {"label": cls, "probability": float(prob)}
        for cls, prob in sorted(zip(context.classes, proba), key=lambda item: item[1], reverse=True)
    ]
    class_index = list(context.classes).index(label)
    predicted_probability = float(proba[class_index])
    if "NORMAL" in context.classes:
        normal_index = list(context.classes).index("NORMAL")
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


def generate_metadata(
    df: Optional[pd.DataFrame],
    context: ModelContext,
    store: AppStore,
) -> List[Dict[str, Any]]:
    reference_df = df if df is not None and not df.empty else pd.DataFrame([DEFAULT_RECORD])
    metadata: List[Dict[str, Any]] = []

    for order_index, column in enumerate(REQUIRED_COLUMNS):
        importance_score = store.column_importance.get(column, 0.0)
        entry: Dict[str, Any] = {
            "name": column,
            "order_index": order_index,
            "importance_score": importance_score,
            "is_top_feature": column in store.top_columns,
        }

        if column in BOOLEAN_COLUMNS:
            default_value = (
                int(reference_df[column].mode(dropna=True).iloc[0]) if column in reference_df else DEFAULT_RECORD[column]
            )
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
            values.update(context.category_options.get(column, []))
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


def build_context(
    request: Request,
    *,
    store: AppStore,
    context: ModelContext,
    result: Optional[Dict[str, Any]],
    error: Optional[str],
    notifications: Optional[List[str]] = None,
    filename: Optional[str] = None,
    prediction: Optional[Dict[str, Any]] = None,
    prediction_error: Optional[str] = None,
    form_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_metadata = store.current_metadata or generate_metadata(None, context, store)
    metadata_for_template: List[Dict[str, Any]] = []
    for field in base_metadata:
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
        "notifications": notifications or [],
        "filename": filename,
        "model_ready": store.model_context is not None,
        "prediction": prediction,
        "prediction_error": prediction_error,
        "input_metadata": metadata_for_template,
        "importance_image": context.importance_image,
        "top_features": context.importance_df.head(store.settings.top_feature_list_count).to_dict("records"),
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
async def upload_page(
    request: Request,
    store: AppStore = Depends(get_store),
    tmpl: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    try:
        context = store.ensure_context()
    except RuntimeError as exc:
        return tmpl.TemplateResponse(
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
            },
        )

    notifications = store.last_analysis["notifications"] if store.last_analysis else []
    return tmpl.TemplateResponse(
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
async def analyze_csv(
    request: Request,
    file: UploadFile = File(...),
    store: AppStore = Depends(get_store),
    tmpl: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    try:
        context = store.ensure_context()
    except RuntimeError as exc:
        return tmpl.TemplateResponse(
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
            },
        )

    if not file.filename.lower().endswith(".csv"):
        return tmpl.TemplateResponse(
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
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to read CSV: %s", exc)
        return tmpl.TemplateResponse(
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
        analysis = analyze_dataset(dataframe, context, store)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to analyze dataset: %s", exc)
        return tmpl.TemplateResponse(
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

    store.last_analysis = analysis
    store.current_metadata = analysis["metadata"]

    return tmpl.TemplateResponse(
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
async def predict_single(
    request: Request,
    file: UploadFile = File(...),
    store: AppStore = Depends(get_store),
    tmpl: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    try:
        context = store.ensure_context()
    except RuntimeError as exc:
        return tmpl.TemplateResponse(
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
            },
        )

    if not file.filename.lower().endswith(".csv"):
        return tmpl.TemplateResponse(
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
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to read single prediction CSV: %s", exc)
        return tmpl.TemplateResponse(
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
        return tmpl.TemplateResponse(
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
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to transform single record: %s", exc)
        return tmpl.TemplateResponse(
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

    return tmpl.TemplateResponse(
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
async def predict_from_form(
    request: Request,
    store: AppStore = Depends(get_store),
    tmpl: Jinja2Templates = Depends(get_templates),
) -> HTMLResponse:
    try:
        context = store.ensure_context()
    except RuntimeError as exc:
        return tmpl.TemplateResponse(
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
            },
        )

    form_data = await request.form()
    form_dict = dict(form_data)

    metadata = store.current_metadata or generate_metadata(None, context, store)

    try:
        record = parse_form_to_record(form_dict, metadata)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to parse form data: %s", exc)
        return tmpl.TemplateResponse(
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
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to transform form record: %s", exc)
        return tmpl.TemplateResponse(
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

    return tmpl.TemplateResponse(
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
