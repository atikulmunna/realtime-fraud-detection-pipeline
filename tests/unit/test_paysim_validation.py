import pandas as pd

from src.data.prepare_paysim import validate_paysim_dataframe


def test_validate_paysim_dataframe_happy_path():
    df = pd.DataFrame(
        {
            "step": [1, 2, 3, 4, 150],
            "type": ["TRANSFER", "CASH-OUT", "PAYMENT", "DEBIT", "CASH-IN"],
            "amount": [10.0, 25.0, 1.0, 4.0, 2.0],
            "nameOrig": ["C1", "C2", "C3", "C4", "C5"],
            "oldbalanceOrg": [100.0, 120.0, 2.0, 8.0, 6.0],
            "newbalanceOrig": [90.0, 95.0, 1.0, 4.0, 4.0],
            "nameDest": ["C9", "C8", "M1", "M2", "C7"],
            "oldbalanceDest": [0.0, 1.0, 0.0, 0.0, 1.0],
            "newbalanceDest": [10.0, 26.0, 0.0, 0.0, 3.0],
            "isFraud": [1, 1, 0, 0, 0],
            "isFlaggedFraud": [0, 0, 0, 0, 0],
        }
    )
    result = validate_paysim_dataframe(df)
    assert result.ok
    assert result.rows == 5
    assert result.cols == 11


def test_validate_paysim_dataframe_rejects_invalid_fraud_type():
    df = pd.DataFrame(
        {
            "step": [1],
            "type": ["PAYMENT"],
            "amount": [5.0],
            "nameOrig": ["C1"],
            "oldbalanceOrg": [8.0],
            "newbalanceOrig": [3.0],
            "nameDest": ["M1"],
            "oldbalanceDest": [0.0],
            "newbalanceDest": [0.0],
            "isFraud": [1],
            "isFlaggedFraud": [0],
        }
    )
    result = validate_paysim_dataframe(df)
    assert not result.ok
    assert any("outside TRANSFER/CASH-OUT" in e for e in result.errors)


def test_validate_paysim_accepts_underscore_cashout_type():
    df = pd.DataFrame(
        {
            "step": [1],
            "type": ["CASH_OUT"],
            "amount": [5.0],
            "nameOrig": ["C1"],
            "oldbalanceOrg": [8.0],
            "newbalanceOrig": [3.0],
            "nameDest": ["C2"],
            "oldbalanceDest": [0.0],
            "newbalanceDest": [5.0],
            "isFraud": [1],
            "isFlaggedFraud": [0],
        }
    )
    result = validate_paysim_dataframe(df)
    assert result.ok
