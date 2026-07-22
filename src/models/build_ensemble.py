"""Build a calibrated immutable ensemble bundle from trained component artifacts."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.feature_contract import FEATURES_V1
from src.data.splitting import chronological_split
from src.models.ensemble_bundle import build_bundle, check_candidate_quality, save_bundle, select_ensemble_configuration
from src.streaming.ensemble_scoring import load_ensemble_models


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validation_scores(models: Any, x: pd.DataFrame) -> dict[str, np.ndarray]:
    values = x.to_numpy(dtype=float)
    if_raw = np.asarray(models.if_model.decision_function(values), dtype=float)
    if_scores = 1.0 / (1.0 + np.exp(np.clip(if_raw, -40.0, 40.0)))
    scaled = models.ae_scaler.transform(values)
    reconstructed = models.ae_model.predict(scaled)
    mse = np.mean((reconstructed - scaled) ** 2, axis=1)
    ae_scores = np.clip(mse / max(float(models.ae_threshold_p99), 1e-12), 0.0, 1.0)
    sgd_scores = np.asarray(models.sgd_model.predict_proba(values)[:, 1], dtype=float)
    return {"if": if_scores, "ae": ae_scores, "sgd": sgd_scores}


def create_calibrated_bundle(
    *,
    input_parquet: str | Path,
    if_model_path: str | Path,
    ae_model_path: str | Path,
    sgd_model_path: str | Path,
    output_path: str | Path,
    model_version: str,
    alert_budget_ratio: float = 0.005,
    git_revision: str | None = None,
) -> dict[str, Any]:
    df = pd.read_parquet(input_parquet)
    required = [*FEATURES_V1, "isFraud"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Calibration dataset is missing required columns: {missing}")
    validation = chronological_split(df).validation
    x = validation[FEATURES_V1]
    y = validation["isFraud"].astype(int).to_numpy()
    models = load_ensemble_models(
        if_model_path=if_model_path,
        ae_model_path=ae_model_path,
        sgd_model_path=sgd_model_path,
    )
    selection = select_ensemble_configuration(
        _validation_scores(models, x),
        y,
        alert_budget_ratio=alert_budget_ratio,
    )
    accepted, reasons = check_candidate_quality(selection.metrics)
    if not accepted:
        raise ValueError(f"Calibrated ensemble failed quality gates: {list(reasons)}")

    dataset_bytes = pd.util.hash_pandas_object(validation[required], index=False).to_numpy().tobytes()
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()
    bundle = build_bundle(
        models,
        selection,
        model_version=model_version,
        dataset_hash=dataset_hash,
        git_revision=git_revision or _git_revision(),
    )
    manifest = save_bundle(bundle, output_path)
    return {
        "output_path": str(output_path),
        "model_version": model_version,
        "dataset_hash": dataset_hash,
        "metrics": selection.metrics,
        "weights": selection.weights,
        "threshold": selection.threshold,
        "sha256": manifest["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a calibrated immutable ensemble bundle.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--if-model", default="models/isolation_forest_v1.joblib")
    parser.add_argument("--ae-model", default="models/autoencoder_v1.joblib")
    parser.add_argument("--sgd-model", default="models/sgd_classifier_v1.joblib")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--alert-budget-ratio", type=float, default=0.005)
    args = parser.parse_args()
    result = create_calibrated_bundle(
        input_parquet=args.input,
        if_model_path=args.if_model,
        ae_model_path=args.ae_model,
        sgd_model_path=args.sgd_model,
        output_path=args.output,
        model_version=args.model_version,
        alert_budget_ratio=args.alert_budget_ratio,
    )
    print(result)


if __name__ == "__main__":
    main()
