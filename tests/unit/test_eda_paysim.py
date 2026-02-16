from pathlib import Path

import pandas as pd

from src.data.eda_paysim import build_summary, run_eda


def test_build_summary_contains_expected_fields():
    df = pd.DataFrame(
        {
            "step": [1, 2, 3, 4],
            "type": ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"],
            "amount": [10.0, 15.0, 1.0, 2.0],
            "isFraud": [1, 1, 0, 0],
        }
    )
    summary = build_summary(df)
    assert summary["rows"] == 4
    assert summary["cols"] == 4
    assert summary["fraud_count"] == 2
    assert "fraud_by_type" in summary
    assert "amount_stats" in summary
    assert summary["invalid_fraud_types"] == []


def test_build_summary_detects_invalid_fraud_types():
    df = pd.DataFrame(
        {
            "step": [1, 2],
            "type": ["PAYMENT", "TRANSFER"],
            "amount": [5.0, 8.0],
            "isFraud": [1, 0],
        }
    )
    summary = build_summary(df)
    assert summary["invalid_fraud_types"] == ["PAYMENT"]


def test_run_eda_writes_output(tmp_path: Path):
    csv_path = tmp_path / "paysim.csv"
    out_path = tmp_path / "summary.json"
    pd.DataFrame(
        {
            "step": [1],
            "type": ["TRANSFER"],
            "amount": [5.0],
            "isFraud": [0],
        }
    ).to_csv(csv_path, index=False)

    summary = run_eda(csv_path, out_path)
    assert summary["rows"] == 1
    assert out_path.exists()
