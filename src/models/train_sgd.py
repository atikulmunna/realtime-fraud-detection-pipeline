"""Train SGD classifier baseline on engineered PaySim features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.common.feature_contract import FEATURES_V1
from src.models.mlflow_logging import log_training_run


def _validate_columns(df: pd.DataFrame) -> None:
    missing_features = [c for c in FEATURES_V1 if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    if "isFraud" not in df.columns:
        raise ValueError("Missing required target column: isFraud")


def train_sgd_classifier(
    input_parquet: str | Path,
    output_model: str | Path = "models/sgd_classifier_v1.joblib",
    output_metrics: str | Path = "models/sgd_classifier_v1_metrics.json",
    random_state: int = 42,
    max_iter: int = 1000,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment: str = "realtime-fraud-detection-pipeline",
    mlflow_run_name: str = "sgd_baseline_v1",
) -> dict[str, Any]:
    df = pd.read_parquet(input_parquet)
    _validate_columns(df)

    X = df[FEATURES_V1]
    y = df["isFraud"].astype(int)

    split_idx = int(0.8 * len(df))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("Dataset is too small for train/validation split.")

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]

    model = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        random_state=random_state,
        max_iter=max_iter,
        tol=1e-3,
    )
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    metrics = {
        "model": "SGDClassifier",
        "input_parquet": str(input_parquet),
        "output_model": str(output_model),
        "rows_total": int(len(df)),
        "rows_train": int(len(X_train)),
        "rows_val": int(len(X_val)),
        "fraud_count_total": int(y.sum()),
        "fraud_count_train": int(y_train.sum()),
        "fraud_count_val": int(y_val.sum()),
        "random_state": random_state,
        "max_iter": max_iter,
        "features_order": FEATURES_V1,
        "val_pr_auc": float(average_precision_score(y_val, val_proba)),
        "val_precision_at_0_5": float(precision_score(y_val, val_pred, zero_division=0)),
        "val_recall_at_0_5": float(recall_score(y_val, val_pred, zero_division=0)),
    }

    out_model = Path(output_model)
    out_metrics = Path(output_metrics)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    payload = {"model_type": "sgd_classifier", "model": model, "features_order": FEATURES_V1}
    joblib.dump(payload, out_model)
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    log_training_run(
        use_mlflow=use_mlflow,
        model_name="SGDClassifier",
        run_name=mlflow_run_name,
        params={
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
    parser = argparse.ArgumentParser(description="Train SGD baseline from engineered PaySim parquet.")
    parser.add_argument("--input", required=True, help="Path to input parquet.")
    parser.add_argument("--output-model", default="models/sgd_classifier_v1.joblib")
    parser.add_argument("--output-metrics", default="models/sgd_classifier_v1_metrics.json")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--mlflow", action="store_true", help="Log run to MLflow.")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--mlflow-experiment", default="realtime-fraud-detection-pipeline")
    parser.add_argument("--mlflow-run-name", default="sgd_baseline_v1")
    args = parser.parse_args()

    metrics = train_sgd_classifier(
        input_parquet=args.input,
        output_model=args.output_model,
        output_metrics=args.output_metrics,
        random_state=args.random_state,
        max_iter=args.max_iter,
        use_mlflow=args.mlflow,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        mlflow_run_name=args.mlflow_run_name,
    )
    print(
        f"rows_train={metrics['rows_train']} rows_val={metrics['rows_val']} "
        f"pr_auc={metrics['val_pr_auc']:.6f} saved_model={metrics['output_model']}"
    )


if __name__ == "__main__":
    main()
