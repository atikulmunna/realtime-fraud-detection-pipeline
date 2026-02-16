import pandas as pd

from src.data.feature_engineering import add_training_features


def test_add_training_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "step": [1, 1, 2],
            "type": ["TRANSFER", "PAYMENT", "CASH-OUT"],
            "amount": [50.0, 5.0, 10.0],
            "nameOrig": ["C1", "C1", "C2"],
            "oldbalanceOrg": [100.0, 50.0, 20.0],
            "newbalanceOrig": [50.0, 45.0, 10.0],
        }
    )

    out = add_training_features(df)
    assert "amount_ratio" in out.columns
    assert "balance_diff_orig" in out.columns
    assert "is_transfer" in out.columns
    assert "is_cashout" in out.columns
    assert "hour_of_day" in out.columns
    assert "txn_velocity_1h" in out.columns


def test_add_training_features_computes_values():
    df = pd.DataFrame(
        {
            "step": [1, 1],
            "type": ["TRANSFER", "TRANSFER"],
            "amount": [50.0, 20.0],
            "nameOrig": ["C1", "C1"],
            "oldbalanceOrg": [100.0, 50.0],
            "newbalanceOrig": [50.0, 30.0],
        }
    )

    out = add_training_features(df)
    first = out.iloc[0]
    assert round(first["amount_ratio"], 6) == round(50.0 / 101.0, 6)
    assert first["balance_diff_orig"] == 0.0
    assert int(first["is_transfer"]) == 1
    assert int(first["is_cashout"]) == 0
    assert int(first["txn_velocity_1h"]) == 1
    assert int(out.iloc[1]["txn_velocity_1h"]) == 2


def test_add_training_features_handles_underscore_cashout():
    df = pd.DataFrame(
        {
            "step": [1],
            "type": ["CASH_OUT"],
            "amount": [10.0],
            "nameOrig": ["C1"],
            "oldbalanceOrg": [20.0],
            "newbalanceOrig": [10.0],
        }
    )
    out = add_training_features(df)
    assert int(out.iloc[0]["is_cashout"]) == 1
