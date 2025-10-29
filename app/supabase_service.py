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
        # client.table(table_name).insert(metadata).execute()
        schemas_splited = table_name.split(".")
        schemas_table = schemas_splited[1]
        schema = schemas_splited[0]
        client.schema(schema).table(schemas_table).insert(metadata).execute()
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
    from mpl_toolkits.mplot3d import Axes3D
    import io
    import base64

    def fig_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_base64


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


    # === обучение RandomForest ===
    try:
        rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(features, target)
    except Exception as exc:
        logger.warning("Не удалось обучить RandomForest: %s", exc)
        return {}

    if "is_anomaly_ground_truth" in working.columns:
        target_color = (
            working["is_anomaly_ground_truth"]
            .map({0: "Норма", 1: "Аномалия"})
            .fillna("Норма")
        )
    else:
        # Fallback: derive labels from модельных предсказаний
        target_color = working["predicted_anomaly"].map(lambda label: "Норма" if str(label) == "NORMAL" else "Аномалия")
    # === PCA 2D ===
    try:
        embedding_2d = PCA(n_components=2).fit_transform(features)
    except Exception as exc:
        logger.warning("Не удалось выполнить PCA 2D: %s", exc)
        return {}

    palette = {"Норма": "#079362", "Аномалия": "#6A0598"}

    fig2d, ax = plt.subplots(figsize=(8, 6))
    for label in target_color.unique():
        mask = target_color == label
        ax.scatter(
            embedding_2d[mask.values, 0],
            embedding_2d[mask.values, 1],
            s=40,
            alpha=0.7,
            label=label,
            color=palette[label],
            edgecolor="none",
        )
    ax.set_title("Кластеры (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Класс")
    cluster_image_2d = fig_to_base64(fig2d)

    # === PCA 3D ===
    try:
        embedding_3d = PCA(n_components=3).fit_transform(features)
        fig3d = plt.figure(figsize=(8, 6))
        ax3d = fig3d.add_subplot(111, projection="3d")

        for label in target_color.unique():
            mask = target_color == label
            ax3d.scatter(
                embedding_3d[mask.values, 0],
                embedding_3d[mask.values, 1],
                embedding_3d[mask.values, 2],
                s=50,
                alpha=0.8,
                color=palette[label],
                label=label,
            )

        ax3d.set_title("Кластеры (PCA 3D)")
        ax3d.set_xlabel("PC1")
        ax3d.set_ylabel("PC2")
        ax3d.set_zlabel("PC3")
        ax3d.legend(title="Класс")
        cluster_image_3d = fig_to_base64(fig3d)
    except Exception as exc:
        logger.warning("Не удалось выполнить PCA 3D: %s", exc)
        cluster_image_3d = ""

    # === Важность признаков ===
    rf_importance_df = pd.DataFrame({"feature": context.feature_names, "importance": rf.feature_importances_})
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

    # === Тепловая карта корреляций ===
    try:
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(working.corr(numeric_only=True), cmap="viridis", center=0, ax=ax_corr)
        ax_corr.set_title("Тепловая карта корреляций признаков")
        corr_image = fig_to_base64(fig_corr)
    except Exception as exc:
        logger.warning("Не удалось построить тепловую карту корреляций: %s", exc)
        corr_image = ""

    # === KDE-график распределения amount_mdl с учетом predicted_anomaly ===
    try:
        fig_kde, ax_kde = plt.subplots(figsize=(8, 6))
        KDE_palette= {
            'NORMAL': "#015428",
            'ECOM_NIGHT_NO3DS_HIGHAMT': "#420061",
            'VELOCITY_BURST': "#420061",
            'NEW_DEVICE_IP_CROSS': "#420061",
            'POS_MAGSTRIPE_NIGHT': "#420061",
            'RARE_COUNTRY_CCY': "#420061",
            'IMPOSSIBLE_TRAVEL': "#420061",
        }

        sns.kdeplot(data=working, x="amount_mdl", hue="predicted_anomaly", fill=True, ax=ax_kde, palette=KDE_palette)
        ax_kde.set_title("Распределение суммы транзакций (amount_mdl) с выделением аномалий")
        kde_image = fig_to_base64(fig_kde)
    except Exception as exc:
        logger.warning("Не удалось построить KDE-график: %s", exc)
        kde_image = ""

    class_counts = target.value_counts().sort_index().to_dict()

    return {
        "sample_size": int(len(target)),
        "class_counts": class_counts,
        "cluster_image_2d": cluster_image_2d,
        "cluster_image_3d": cluster_image_3d,
        "feature_image": feature_image,
        "correlation_heatmap": corr_image,
        "amount_kde_plot": kde_image,
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

    rf_visuals = build_random_forest_visuals(combined, context)
    # print(rf_visuals)
    store.rf_visuals = rf_visuals
    client = store.supabase_client
    if client is None:
        return None

    snapshots_table = store.settings.supabase_snapshots_table

    try:
        # --- Извлекаем нужные поля из визуализаций ---
        sample_size = len(combined)

        # предполагается, что build_random_forest_visuals возвращает словарь со статистикой
        class_counts = rf_visuals.get("class_counts", {})
        # redundant 
        # feature_importance = rf_visuals.get("feature_importance", [])

        # --- Формируем запись для таблицы rf_snapshots ---
        snapshot_record = {
            "created_at": pd.Timestamp.now().isoformat(),
            "sample_size": sample_size,
            "class_counts": class_counts,
            # redundant
            # "feature_importance": feature_importance,
            "cluster_image_2d": rf_visuals.get("cluster_image_2d", ""),
            "cluster_image_3d": rf_visuals.get("cluster_image_3d", ""),
            "feature_image": rf_visuals.get("feature_image", ""),
            "correlation_heatmap": rf_visuals.get("corr_image", ""),
            "amount_kde_plot": rf_visuals.get("kde_image", ""),
            # "notes": "Aggregated update from backend process"
        }

        # --- Сохраняем в таблицу Supabase ---
        # response = client.table(snapshots_table).insert(snapshot_record).execute()
        schemas_splited = snapshots_table.split(".")
        schemas_table = schemas_splited[1]
        schema = schemas_splited[0]
        response = client.schema(schema).table(schemas_table).insert(snapshot_record).execute()

        logger.info(
            "RF snapshot успешно сохранён в таблицу %s (sample_size=%d).",
            snapshots_table,
            sample_size,
        )
        return response

    except Exception as exc:
        logger.warning("Ошибка при сохранении rf_snapshot в таблицу %s: %s", snapshots_table, exc)
        return None
    

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
