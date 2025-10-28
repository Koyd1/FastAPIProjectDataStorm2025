from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .constants import (
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLS,
    DEFAULT_RECORD,
    NUMERIC_COLUMNS,
    ORDINAL_ENCODINGS,
    REQUIRED_COLUMNS,
)
from .model import ModelContext


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
