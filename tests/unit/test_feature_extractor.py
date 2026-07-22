import pandas as pd
import pytest

from src.common.feature_contract import FEATURES_V1
from src.data.feature_engineering import add_training_features
from src.streaming.feature_extractor import enrich_event_with_features, extract_features


def _event() -> dict:
    return {
        "event_id": "evt-1",
        "timestamp": "2026-02-16T23:45:00Z",
        "user_id": "C123",
        "type": "TRANSFER",
        "amount": 100.0,
        "old_balance_orig": 500.0,
        "new_balance_orig": 400.0,
    }


def test_extract_features_happy_path():
    features = extract_features(_event(), txn_velocity_1h=3)
    assert features["amount"] == 100.0
    assert round(features["amount_ratio"], 6) == round(100.0 / 501.0, 6)
    assert features["balance_diff_orig"] == 0.0
    assert features["is_transfer"] == 1
    assert features["is_cashout"] == 0
    assert features["hour_of_day"] == 23
    assert features["txn_velocity_1h"] == 3


def test_extract_features_handles_cashout_underscore():
    ev = _event()
    ev["type"] = "CASH_OUT"
    features = extract_features(ev)
    assert features["is_transfer"] == 0
    assert features["is_cashout"] == 1


def test_extract_features_missing_required_field_raises():
    ev = _event()
    ev.pop("timestamp")
    with pytest.raises(ValueError, match="Missing required event fields"):
        extract_features(ev)


def test_enrich_event_with_features_attaches_features():
    enriched = enrich_event_with_features(_event(), txn_velocity_1h=2)
    assert "features" in enriched
    assert enriched["features"]["txn_velocity_1h"] == 2


def test_offline_and_online_feature_values_match_for_equivalent_event():
    offline = add_training_features(
        pd.DataFrame(
            {
                "step": [1],
                "type": ["CASH_OUT"],
                "amount": [100.0],
                "nameOrig": ["C123"],
                "oldbalanceOrg": [500.0],
                "newbalanceOrig": [400.0],
            }
        )
    ).iloc[0]
    event = {
        "event_id": "evt-parity",
        "timestamp": f"{offline['timestamp'].isoformat()}Z",
        "user_id": "C123",
        "type": "CASH_OUT",
        "amount": 100.0,
        "old_balance_orig": 500.0,
        "new_balance_orig": 400.0,
    }

    online = extract_features(event, txn_velocity_1h=1)

    assert [float(online[name]) for name in FEATURES_V1] == [float(offline[name]) for name in FEATURES_V1]
