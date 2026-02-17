"""Feature extraction helpers for streaming events."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def _parse_hour(timestamp: str) -> int:
    ts = timestamp
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    return int(dt.hour)


def extract_features(
    event: dict[str, Any],
    *,
    txn_velocity_1h: int = 1,
) -> dict[str, Any]:
    required = [
        "amount",
        "old_balance_orig",
        "new_balance_orig",
        "type",
        "timestamp",
    ]
    missing = [k for k in required if k not in event]
    if missing:
        raise ValueError(f"Missing required event fields for feature extraction: {missing}")

    amount = float(event["amount"])
    old_balance = float(event["old_balance_orig"])
    new_balance = float(event["new_balance_orig"])
    txn_type = str(event["type"])

    features = {
        "amount": amount,
        "amount_ratio": amount / (old_balance + 1.0),
        "balance_diff_orig": old_balance - new_balance - amount,
        "is_transfer": int(txn_type == "TRANSFER"),
        "is_cashout": int(txn_type in {"CASH-OUT", "CASH_OUT"}),
        "hour_of_day": _parse_hour(str(event["timestamp"])),
        "txn_velocity_1h": int(txn_velocity_1h),
    }
    return features


def enrich_event_with_features(
    event: dict[str, Any],
    *,
    txn_velocity_1h: int = 1,
) -> dict[str, Any]:
    out = dict(event)
    out["features"] = extract_features(event, txn_velocity_1h=txn_velocity_1h)
    return out
