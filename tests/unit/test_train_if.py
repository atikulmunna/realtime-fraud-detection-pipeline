import json
from pathlib import Path

import pandas as pd
import pytest

from src.common.feature_contract import FEATURES_V1
from src.models.train_if import train_isolation_forest


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [10.0, 12.0, 9.0, 11.0, 15.0, 8.0],
            "amount_ratio": [0.1, 0.2, 0.08, 0.12, 0.25, 0.07],
            "balance_diff_orig": [0.0, 0.0, 0.1, -0.1, 0.0, 0.0],
            "is_transfer": [1, 1, 0, 1, 0, 1],
            "is_cashout": [0, 1, 0, 0, 1, 0],
            "hour_of_day": [1, 2, 3, 4, 5, 6],
            "txn_velocity_1h": [1, 2, 1, 1, 1, 2],
            "isFraud": [0, 0, 0, 0, 1, 0],
        }
    )


def test_train_isolation_forest_creates_artifacts(tmp_path: Path):
    df = _base_df()
    input_parquet = tmp_path / "features.parquet"
    model_out = tmp_path / "if.joblib"
    metrics_out = tmp_path / "if_metrics.json"
    df.to_parquet(input_parquet, index=False)

    metrics = train_isolation_forest(
        input_parquet=input_parquet,
        output_model=model_out,
        output_metrics=metrics_out,
        sample_size=4,
        contamination=0.1,
        random_state=1,
    )

    assert model_out.exists()
    assert metrics_out.exists()
    saved = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert saved["features_order"] == FEATURES_V1
    assert saved["rows_train"] == 4
    assert metrics["rows_train"] == 4


def test_train_isolation_forest_raises_on_missing_feature(tmp_path: Path):
    df = _base_df().drop(columns=["txn_velocity_1h"])
    input_parquet = tmp_path / "bad.parquet"
    df.to_parquet(input_parquet, index=False)

    with pytest.raises(ValueError, match="Missing required features"):
        train_isolation_forest(input_parquet=input_parquet)
