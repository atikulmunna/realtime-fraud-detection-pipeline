"""Train Isolation Forest baseline on engineered PaySim features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.common.feature_contract import FEATURES_V1


def _validate_features(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURES_V1 if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")


def train_isolation_forest(
    input_parquet: str | Path,
    output_model: str | Path = "models/isolation_forest_v1.joblib",
    output_metrics: str | Path = "models/isolation_forest_v1_metrics.json",
    sample_size: int = 200_000,
    contamination: float = 0.001291,
    random_state: int = 42,
) -> dict[str, Any]:
    df = pd.read_parquet(input_parquet)
    _validate_features(df)

    if "isFraud" in df.columns:
        normal_df = df[df["isFraud"] == 0]
    else:
        normal_df = df

    n_available = int(normal_df.shape[0])
    n_train = min(sample_size, n_available)
    if n_train <= 0:
        raise ValueError("No rows available for training.")

    train_df = normal_df.sample(n=n_train, random_state=random_state)
    X = train_df[FEATURES_V1]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    out_model = Path(output_model)
    out_metrics = Path(output_metrics)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_model)

    metrics = {
        "model": "IsolationForest",
        "input_parquet": str(input_parquet),
        "output_model": str(out_model),
        "rows_total": int(df.shape[0]),
        "rows_normal": n_available,
        "rows_train": n_train,
        "sample_size_requested": sample_size,
        "contamination": contamination,
        "random_state": random_state,
        "features_order": FEATURES_V1,
    }
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IF baseline from engineered PaySim parquet.")
    parser.add_argument("--input", required=True, help="Path to input parquet.")
    parser.add_argument("--output-model", default="models/isolation_forest_v1.joblib")
    parser.add_argument("--output-metrics", default="models/isolation_forest_v1_metrics.json")
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--contamination", type=float, default=0.001291)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    metrics = train_isolation_forest(
        input_parquet=args.input,
        output_model=args.output_model,
        output_metrics=args.output_metrics,
        sample_size=args.sample_size,
        contamination=args.contamination,
        random_state=args.random_state,
    )
    print(
        f"trained_rows={metrics['rows_train']} contamination={metrics['contamination']} "
        f"saved_model={metrics['output_model']}"
    )


if __name__ == "__main__":
    main()
