"""Train AutoEncoder-style baseline using MLP reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from src.common.feature_contract import FEATURES_V1
from src.models.mlflow_logging import log_training_run


def _validate_features(df: pd.DataFrame) -> None:
    missing = [c for c in FEATURES_V1 if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")


def train_autoencoder(
    input_parquet: str | Path,
    output_model: str | Path = "models/autoencoder_v1.joblib",
    output_metrics: str | Path = "models/autoencoder_v1_metrics.json",
    sample_size: int = 200_000,
    random_state: int = 42,
    max_iter: int = 50,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = "realtime-fraud-detection-pipeline",
    mlflow_run_name: str = "ae_baseline_v1",
) -> dict[str, Any]:
    df = pd.read_parquet(input_parquet)
    _validate_features(df)

    if "isFraud" in df.columns:
        normal_df = df[df["isFraud"] == 0]
    else:
        normal_df = df

    n_available = int(normal_df.shape[0])
    n_train_total = min(sample_size, n_available)
    if n_train_total <= 10:
        raise ValueError("Not enough normal rows available for AE training.")

    sampled = normal_df.sample(n=n_train_total, random_state=random_state)
    X = sampled[FEATURES_V1].to_numpy(dtype=float)

    split_idx = int(0.8 * len(X))
    X_train = X[:split_idx]
    X_val = X[split_idx:]

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # 7 -> 5 -> 3 -> 5 -> 7 via hidden layers and reconstruction target.
    model = MLPRegressor(
        hidden_layer_sizes=(5, 3, 5),
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(X_train_s, X_train_s)

    X_val_pred = model.predict(X_val_s)
    val_mse = np.mean((X_val_pred - X_val_s) ** 2, axis=1)
    threshold_p99 = float(np.quantile(val_mse, 0.99))

    payload = {
        "model_type": "mlp_autoencoder",
        "scaler": scaler,
        "model": model,
        "features_order": FEATURES_V1,
        "threshold_p99": threshold_p99,
    }

    out_model = Path(output_model)
    out_metrics = Path(output_metrics)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_model)

    metrics = {
        "model": "AutoEncoder-MLP",
        "input_parquet": str(input_parquet),
        "output_model": str(out_model),
        "rows_total": int(df.shape[0]),
        "rows_normal": n_available,
        "rows_sampled": n_train_total,
        "rows_train": int(X_train.shape[0]),
        "rows_val": int(X_val.shape[0]),
        "random_state": random_state,
        "max_iter": max_iter,
        "threshold_p99": threshold_p99,
        "features_order": FEATURES_V1,
    }
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log_training_run(
        use_mlflow=use_mlflow,
        model_name="AutoEncoder-MLP",
        run_name=mlflow_run_name,
        params={
            "sample_size": sample_size,
            "random_state": random_state,
            "max_iter": max_iter,
            "features_order": FEATURES_V1,
        },
        metrics=metrics,
        artifacts=[out_model, out_metrics],
        tracking_uri=mlflow_tracking_uri,
        experiment_name=mlflow_experiment,
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AE baseline from engineered PaySim parquet.")
    parser.add_argument("--input", required=True, help="Path to input parquet.")
    parser.add_argument("--output-model", default="models/autoencoder_v1.joblib")
    parser.add_argument("--output-metrics", default="models/autoencoder_v1_metrics.json")
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--mlflow", action="store_true", help="Log run to MLflow.")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment", default="realtime-fraud-detection-pipeline")
    parser.add_argument("--mlflow-run-name", default="ae_baseline_v1")
    args = parser.parse_args()

    metrics = train_autoencoder(
        input_parquet=args.input,
        output_model=args.output_model,
        output_metrics=args.output_metrics,
        sample_size=args.sample_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
        use_mlflow=args.mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
    )
    print(
        f"sampled_rows={metrics['rows_sampled']} val_rows={metrics['rows_val']} "
        f"threshold_p99={metrics['threshold_p99']:.8f} saved_model={metrics['output_model']}"
    )


if __name__ == "__main__":
    main()
