"""Shared MLflow logging helpers for training scripts."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


def _coerce_params(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif v is None:
            continue
        else:
            out[k] = str(v)
    return out


def _coerce_metrics(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in payload.items():
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def log_training_run(
    *,
    use_mlflow: bool,
    model_name: str,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: list[str | Path],
    tracking_uri: str | None = None,
    experiment_name: str = "realtime-fraud-detection-pipeline",
) -> bool:
    if not use_mlflow:
        return False

    try:
        mlflow = importlib.import_module("mlflow")
    except ImportError as exc:
        raise RuntimeError("MLflow is not installed. Install it or disable --mlflow.") from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(_coerce_params(params))
        mlflow.log_metrics(_coerce_metrics(metrics))
        mlflow.set_tags({"model_name": model_name, "feature_schema_version": "1"})
        for item in artifacts:
            p = Path(item)
            if p.exists():
                mlflow.log_artifact(str(p))
    return True
