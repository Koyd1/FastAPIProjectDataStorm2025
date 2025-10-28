from typing import Any, Dict, List, Optional, Set, Tuple

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
SENSITIVE_COLUMNS: Set[str] = {
    "transaction_id",
    "customer_id",
    "timestamp",
    "card_number",
    "cardholder_name",
    "email",
    "phone",
}
