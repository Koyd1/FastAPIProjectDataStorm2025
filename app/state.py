from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from .config import Settings
from .constants import (
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLS,
    DEFAULT_RECORD,
    NUMERIC_COLUMNS,
    ORDINAL_ENCODINGS,
    REQUIRED_COLUMNS,
)
from .model import ModelContext, aggregate_column_importance, compute_top_columns


@dataclass
class AppStore:
    settings: Settings
    model_context: Optional[ModelContext] = None
    column_importance: Dict[str, float] = field(default_factory=dict)
    top_columns: Set[str] = field(default_factory=set)
    last_analysis: Optional[Dict[str, Any]] = None
    current_metadata: List[Dict[str, Any]] = field(default_factory=list)
    supabase_client: Optional[Any] = None
    db_service: Optional[Any] = None
    aggregated_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    rf_visuals: Dict[str, Any] = field(default_factory=dict)

    def set_context(self, context: ModelContext) -> None:
        self.model_context = context
        self.column_importance = aggregate_column_importance(context.importance_df)
        self.top_columns = compute_top_columns(self.column_importance, self.settings.top_feature_badge_count)

    def ensure_context(self) -> ModelContext:
        if self.model_context is None:
            raise RuntimeError("Model context is not loaded.")
        return self.model_context

    def generate_metadata(self, df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
        context = self.ensure_context()
        reference_df = df if df is not None and not df.empty else pd.DataFrame([DEFAULT_RECORD])
        metadata: List[Dict[str, Any]] = []

        for order_index, column in enumerate(REQUIRED_COLUMNS):
            importance_score = self.column_importance.get(column, 0.0)
            entry: Dict[str, Any] = {
                "name": column,
                "order_index": order_index,
                "importance_score": importance_score,
                "is_top_feature": column in self.top_columns,
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
