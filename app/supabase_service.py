from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

from .constants import REQUIRED_COLUMNS, SENSITIVE_COLUMNS
from .csv_utils import read_uploaded_dataframe
from .model import aggregate_column_importance, fig_to_base64
from .preprocessing import preprocess_features

try:  # pragma: no cover - optional dependency
    from supabase import Client as SupabaseClient
    from supabase import create_client as create_supabase_client
except ImportError:  # pragma: no cover - optional dependency
    SupabaseClient = Any  # type: ignore[assignment]
    create_supabase_client = None


logger = logging.getLogger(__name__)


def init_supabase_client(settings) -> Optional[SupabaseClient]:
    if not settings.supabase_enabled:
        logger.info("Supabase интеграция отключена (нет переменных окружения).")
        return None
    if create_supabase_client is None:
        logger.warning("Библиотека supabase не установлена — Supabase недоступен.")
        return None
    try:
        client = create_supabase_client(settings.supabase_url, settings.supabase_service_key)
        logger.info("Supabase клиент инициализирован.")
        return client
    except Exception as exc:
        logger.exception("Не удалось инициализировать Supabase: %s", exc)
        return None


def sanitize_dataframe_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    sanitized = df.copy()
    for column in SENSITIVE_COLUMNS:
        if column in sanitized.columns:
            sanitized = sanitized.drop(columns=[column])
    sanitized.columns = [col.strip() for col in sanitized.columns]
    sanitized = sanitized.loc[:, ~sanitized.columns.duplicated()]
    return sanitized


def persist_dataset_to_supabase(store, filename: Optional[str], sanitized_df: pd.DataFrame) -> Optional[str]:
    client = store.supabase_client
    if client is None or sanitized_df.empty:
        return None
    bucket = store.settings.supabase_bucket
    table_name = store.settings.supabase_table
    safe_stem = Path(filename or "dataset.csv").stem.replace(" ", "_")[:50] or "dataset"
    object_key = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid4().hex}_{safe_stem}.csv"
    csv_bytes = sanitized_df.to_csv(index=False).encode("utf-8")

    try:
        client.storage.from_(bucket).upload(object_key, csv_bytes, {"content-type": "text/csv"})
        logger.info("Файл %s сохранён в Supabase bucket %s.", object_key, bucket)
    except Exception as exc:
        logger.warning("Не удалось загрузить файл в Supabase bucket %s: %s", bucket, exc)
        return None

    metadata = {
        "filename": filename or "dataset.csv",
        "object_key": object_key,
        "row_count": len(sanitized_df),
        "uploaded_at": datetime.utcnow().isoformat(),
    }
    try:
        client.table(table_name).insert(metadata).execute()
        logger.info("Метаданные по загрузке записаны в таблицу %s.", table_name)
    except Exception as exc:
        logger.warning("Не удалось записать метаданные в таблицу %s: %s", table_name, exc)

    return object_key

def fetch_supabase_history(store) -> pd.DataFrame:
    client = store.supabase_client
    if client is None:
        return pd.DataFrame()
    bucket = store.settings.supabase_bucket
    try:
        objects = client.storage.from_(bucket).list()
    except Exception as exc:
        logger.warning("Не удалось получить список файлов из Supabase bucket %s: %s", bucket, exc)
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    for item in objects:
        name = item.get("name")
        if not name or not name.endswith(".csv"):
            continue
        try:
            payload = client.storage.from_(bucket).download(name)
        except Exception as exc:
            logger.warning("Не удалось скачать %s из Supabase: %s", name, exc)
            continue

        data_bytes: bytes
        if isinstance(payload, bytes):
            data_bytes = payload
        else:
            data_bytes = payload.read()  # type: ignore[call-arg]

        if not data_bytes:
            continue
        try:
            dataframe = read_uploaded_dataframe(data_bytes)
        except Exception as exc:
            logger.warning("Не удалось распарсить сохранённый файл %s: %s", name, exc)
            continue
        frames.append(dataframe)

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .reset_index(drop=True)
    )


def build_random_forest_visuals(aggregated_df: pd.DataFrame, context) -> Dict[str, Any]:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    import seaborn as sns
    import matplotlib.pyplot as plt

    if aggregated_df.empty or "predicted_anomaly" not in aggregated_df.columns:
        return {}

    working = aggregated_df.copy()
    target = working["predicted_anomaly"].astype(str)
    if target.nunique() < 2 or len(target) < 10:
        return {}

    feature_df = working[[col for col in REQUIRED_COLUMNS if col in working.columns]].copy()
    features, _ = preprocess_features(feature_df, context)
    if features.empty:
        return {}

    try:
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(features, target)
    except Exception as exc:
        logger.warning("Не удалось обучить RandomForest: %s", exc)
        return {}

    try:
        embedding = PCA(n_components=2).fit_transform(features)
    except Exception as exc:
        logger.warning("Не удалось выполнить PCA для визуализации: %s", exc)
        return {}

    palette = sns.color_palette("husl", target.nunique())
    label_to_color = {label: palette[idx % len(palette)] for idx, label in enumerate(sorted(target.unique()))}
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color in label_to_color.items():
        mask = target == label
        ax.scatter(
            embedding[mask.values, 0],
            embedding[mask.values, 1],
            s=32,
            alpha=0.75,
            label=label,
            color=color,
            edgecolor="none",
        )
    ax.set_title("Кластеры Random Forest (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Класс", loc="best")

    cluster_image = fig_to_base64(fig)

    rf_importance_df = pd.DataFrame(
        {"feature": context.feature_names, "importance": rf.feature_importances_}
    )
    rf_column_importance = aggregate_column_importance(rf_importance_df)
    top_items = [
        item for item in sorted(rf_column_importance.items(), key=lambda pair: pair[1], reverse=True) if item[1] > 0
    ][:10]
    if top_items:
        top_df = pd.DataFrame(top_items, columns=["feature", "importance"])
        fig_imp, ax_imp = plt.subplots(figsize=(8, max(4, 0.4 * len(top_df))))
        sns.barplot(data=top_df, x="importance", y="feature", palette="mako", ax=ax_imp)
        ax_imp.set_title("Вклад признаков (Random Forest)")
        ax_imp.set_xlabel("Важность")
        ax_imp.set_ylabel("Признак")
        feature_image = fig_to_base64(fig_imp)
    else:
        feature_image = ""

    class_counts = target.value_counts().sort_index().to_dict()

    return {
        "sample_size": int(len(target)),
        "class_counts": class_counts,
        "cluster_image": cluster_image,
        "feature_image": feature_image,
    }


def update_aggregated_state(store, sanitized_df: pd.DataFrame) -> None:
    if sanitized_df.empty:
        return
    if store.aggregated_df.empty:
        combined = sanitized_df.copy()
    else:
        combined = (
            pd.concat([store.aggregated_df, sanitized_df], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
    store.aggregated_df = combined
    try:
        context = store.ensure_context()
    except RuntimeError:
        return
    store.rf_visuals = build_random_forest_visuals(combined, context)


def ingest_dataset(store, dataframe: pd.DataFrame, filename: Optional[str], source: str = "csv") -> None:
    sanitized = sanitize_dataframe_for_storage(dataframe)
    if sanitized.empty:
        logger.info("Санитизированный датасет пуст — Supabase пропущен.")
        return
    update_aggregated_state(store, sanitized)
    object_key: Optional[str] = None
    if store.supabase_client is not None:
        object_key = persist_dataset_to_supabase(store, filename, sanitized)
    if object_key and store.db_service is not None:
        store.db_service.insert_upload_metadata(
            object_key=object_key,
            filename=filename or "dataset.csv",
            row_count=len(sanitized),
            source=source,
        )
