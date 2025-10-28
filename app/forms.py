from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Dict, List, Optional


def build_context(
    request,
    *,
    store,
    context,
    result: Optional[Dict[str, Any]],
    error: Optional[str],
    notifications: Optional[List[str]] = None,
    filename: Optional[str] = None,
    prediction: Optional[Dict[str, Any]] = None,
    prediction_error: Optional[str] = None,
    form_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    base_metadata = store.current_metadata or store.generate_metadata(None)
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
        "rf_visuals": store.rf_visuals or {},
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
