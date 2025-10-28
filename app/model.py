from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Dict, List, Set

import joblib
import lightgbm as lgb
import matplotlib
import pandas as pd
import seaborn as sns

from .constants import (
    CATEGORICAL_COLS,
    DEFAULT_RECORD,
    PREFIX_TO_COLUMN,
    REQUIRED_COLUMNS,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class ModelContext:
    model: lgb.LGBMClassifier
    feature_names: List[str]
    classes: List[str]
    importance_df: pd.DataFrame
    importance_image: str
    category_options: Dict[str, List[str]]
    metadata: Dict[str, any]


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


def load_metadata(metadata_path: str) -> Dict[str, any]:
    try:
        metadata = joblib.load(metadata_path)
        if not isinstance(metadata, dict):
            return {}
        return metadata
    except Exception:
        return {}


def load_model_context(settings) -> ModelContext:
    if not settings.model_path.exists():
        raise FileNotFoundError(f"Не найден файл модели: {settings.model_path}")

    model: lgb.LGBMClassifier = joblib.load(settings.model_path)
    metadata = load_metadata(settings.metadata_path)

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
