import json
from pathlib import Path

import pandas as pd
import pytest

from src.common.feature_contract import FEATURES_V1
from src.models.train_sgd import train_sgd_classifier


def _dataset(n: int = 200) -> pd.DataFrame:
    rows = []
    for i in range(n):
        is_fraud = 1 if i % 17 == 0 else 0
        rows.append(
            {
                "amount": 10.0 + (i % 13) + (20 if is_fraud else 0),
                "amount_ratio": 0.1 + (i % 7) * 0.02 + (0.2 if is_fraud else 0.0),
                "balance_diff_orig": ((i % 5) - 2) * 0.03 + (0.1 if is_fraud else 0.0),
                "is_transfer": 1 if i % 2 == 0 else 0,
                "is_cashout": 1 if i % 11 == 0 else 0,
                "hour_of_day": i % 24,
                "txn_velocity_1h": 1 + (i % 4),
                "isFraud": is_fraud,
            }
        )
    return pd.DataFrame(rows)


def test_train_sgd_creates_artifacts(tmp_path: Path):
    df = _dataset()
    input_parquet = tmp_path / "features.parquet"
    model_out = tmp_path / "sgd.joblib"
    metrics_out = tmp_path / "sgd_metrics.json"
    df.to_parquet(input_parquet, index=False)

    metrics = train_sgd_classifier(
        input_parquet=input_parquet,
        output_model=model_out,
        output_metrics=metrics_out,
        random_state=1,
        max_iter=200,
    )

    assert model_out.exists()
    assert metrics_out.exists()
    saved = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert saved["features_order"] == FEATURES_V1
    assert saved["rows_total"] == len(df)
    assert 0.0 <= saved["val_pr_auc"] <= 1.0
    assert metrics["rows_val"] > 0


def test_train_sgd_raises_on_missing_target(tmp_path: Path):
    df = _dataset().drop(columns=["isFraud"])
    input_parquet = tmp_path / "bad.parquet"
    df.to_parquet(input_parquet, index=False)
    with pytest.raises(ValueError, match="Missing required target column"):
        train_sgd_classifier(input_parquet=input_parquet)


def test_train_sgd_raises_on_missing_feature(tmp_path: Path):
    df = _dataset().drop(columns=["hour_of_day"])
    input_parquet = tmp_path / "bad2.parquet"
    df.to_parquet(input_parquet, index=False)
    with pytest.raises(ValueError, match="Missing required features"):
        train_sgd_classifier(input_parquet=input_parquet)
