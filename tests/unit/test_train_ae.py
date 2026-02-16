import json
from pathlib import Path

import pandas as pd
import pytest

from src.common.feature_contract import FEATURES_V1
from src.models.train_ae import train_autoencoder


def _base_df(n: int = 80) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append(
            {
                "amount": 10.0 + (i % 7),
                "amount_ratio": 0.1 + (i % 5) * 0.01,
                "balance_diff_orig": ((i % 3) - 1) * 0.05,
                "is_transfer": 1 if i % 2 == 0 else 0,
                "is_cashout": 1 if i % 11 == 0 else 0,
                "hour_of_day": i % 24,
                "txn_velocity_1h": 1 + (i % 3),
                "isFraud": 0,
            }
        )
    return pd.DataFrame(rows)


def test_train_autoencoder_creates_artifacts(tmp_path: Path):
    df = _base_df()
    input_parquet = tmp_path / "features.parquet"
    model_out = tmp_path / "ae.joblib"
    metrics_out = tmp_path / "ae_metrics.json"
    df.to_parquet(input_parquet, index=False)

    metrics = train_autoencoder(
        input_parquet=input_parquet,
        output_model=model_out,
        output_metrics=metrics_out,
        sample_size=50,
        random_state=1,
        max_iter=10,
    )

    assert model_out.exists()
    assert metrics_out.exists()
    saved = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert saved["features_order"] == FEATURES_V1
    assert saved["rows_sampled"] == 50
    assert metrics["rows_sampled"] == 50
    assert saved["threshold_p99"] >= 0


def test_train_autoencoder_raises_on_missing_feature(tmp_path: Path):
    df = _base_df().drop(columns=["hour_of_day"])
    input_parquet = tmp_path / "bad.parquet"
    df.to_parquet(input_parquet, index=False)

    with pytest.raises(ValueError, match="Missing required features"):
        train_autoencoder(input_parquet=input_parquet)
