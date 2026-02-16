"""Shared feature contract for training and serving."""

FEATURES_V1 = [
    "amount",
    "amount_ratio",
    "balance_diff_orig",
    "is_transfer",
    "is_cashout",
    "hour_of_day",
    "txn_velocity_1h",
]

LEAKAGE_FIELDS = {"is_fraud", "isFraud", "label"}
