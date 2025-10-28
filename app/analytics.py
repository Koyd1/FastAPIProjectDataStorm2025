from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from .constants import DEFAULT_RECORD, REQUIRED_COLUMNS
from .model import fig_to_base64
from .preprocessing import preprocess_features

import matplotlib.pyplot as plt


def analyze_dataset(
    df: pd.DataFrame,
    context,
    store,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
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
    metadata = store.generate_metadata(metadata_source)

    result = {
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

    storage_df = preview_df.copy()
    storage_df["analysis_run_at"] = datetime.utcnow().isoformat()
    if df.attrs.get("source_filename"):
        storage_df["source_filename"] = df.attrs["source_filename"]

    return result, storage_df


def transform_single_record(df: pd.DataFrame, context) -> Tuple[pd.DataFrame, List[str]]:
    aligned, notifications = preprocess_features(df, context)
    return aligned, notifications


def prediction_summary(features: pd.DataFrame, context) -> Dict[str, Any]:
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
