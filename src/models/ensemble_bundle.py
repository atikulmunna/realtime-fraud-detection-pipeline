"""Calibration, quality gates, and immutable ensemble bundle persistence."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.common.feature_contract import FEATURES_V1, FEATURE_CONTRACT_VERSION
from src.streaming.ensemble_scoring import EnsembleModels


@dataclass(frozen=True)
class ScoreCalibrator:
    slope: float
    intercept: float

    def transform(self, scores: np.ndarray) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        logits = np.clip((self.slope * values) + self.intercept, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass(frozen=True)
class EnsembleSelection:
    calibrators: dict[str, ScoreCalibrator]
    weights: tuple[float, float, float]
    threshold: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class BundleMetadata:
    model_version: str
    dataset_hash: str
    git_revision: str
    metrics: dict[str, float]
    weights: tuple[float, float, float]
    threshold: float
    feature_contract_version: str = FEATURE_CONTRACT_VERSION
    features_order: tuple[str, ...] = tuple(FEATURES_V1)


@dataclass(frozen=True)
class EnsembleBundle:
    models: EnsembleModels
    metadata: BundleMetadata


def fit_score_calibrator(scores: np.ndarray, labels: np.ndarray) -> ScoreCalibrator:
    x = np.asarray(scores, dtype=float).reshape(-1, 1)
    y = np.asarray(labels, dtype=int)
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Scores and labels must be non-empty and have equal length.")
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration labels must contain both classes.")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(x, y)
    return ScoreCalibrator(slope=float(model.coef_[0][0]), intercept=float(model.intercept_[0]))


def _quality_at_budget(labels: np.ndarray, scores: np.ndarray, budget_ratio: float) -> dict[str, float]:
    if not 0.0 < budget_ratio <= 1.0:
        raise ValueError("budget_ratio must be within (0, 1].")
    budget_size = max(1, int(round(len(labels) * budget_ratio)))
    ranked = np.argsort(-scores, kind="stable")[:budget_size]
    positives = int(labels.sum())
    true_positives = int(labels[ranked].sum())
    return {
        "pr_auc": float(average_precision_score(labels, scores)),
        "precision_at_budget": float(true_positives / budget_size),
        "recall_at_budget": float(true_positives / positives) if positives else 0.0,
    }


def select_ensemble_configuration(
    component_scores: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    alert_budget_ratio: float = 0.005,
) -> EnsembleSelection:
    required = {"if", "ae", "sgd"}
    if set(component_scores) != required:
        raise ValueError(f"component_scores must contain exactly {sorted(required)}")
    y = np.asarray(labels, dtype=int)
    calibrators = {name: fit_score_calibrator(values, y) for name, values in component_scores.items()}
    calibrated = {name: calibrators[name].transform(values) for name, values in component_scores.items()}

    best: tuple[tuple[float, float, float], tuple[float, float, float], np.ndarray] | None = None
    for if_units in range(11):
        for ae_units in range(11 - if_units):
            sgd_units = 10 - if_units - ae_units
            weights = (if_units / 10.0, ae_units / 10.0, sgd_units / 10.0)
            combined = (
                weights[0] * calibrated["if"]
                + weights[1] * calibrated["ae"]
                + weights[2] * calibrated["sgd"]
            )
            metrics = _quality_at_budget(y, combined, alert_budget_ratio)
            objective = (
                metrics["recall_at_budget"],
                metrics["precision_at_budget"],
                metrics["pr_auc"],
            )
            if best is None or objective > best[0]:
                best = (objective, weights, combined)

    assert best is not None
    _, weights, combined = best
    budget_size = max(1, int(round(len(y) * alert_budget_ratio)))
    threshold = float(np.sort(combined)[-budget_size])
    metrics = _quality_at_budget(y, combined, alert_budget_ratio)
    return EnsembleSelection(calibrators=calibrators, weights=weights, threshold=threshold, metrics=metrics)


def check_candidate_quality(
    candidate: dict[str, float],
    champion: dict[str, float] | None = None,
    *,
    max_regression: float = 0.02,
) -> tuple[bool, tuple[str, ...]]:
    absolute_gates = {"pr_auc": 0.10, "precision_at_budget": 0.10, "recall_at_budget": 0.60}
    reasons = [f"{name}={candidate.get(name, 0.0):.6f} below {minimum:.6f}" for name, minimum in absolute_gates.items() if candidate.get(name, 0.0) < minimum]
    if champion is not None:
        for name in absolute_gates:
            if candidate.get(name, 0.0) < champion.get(name, 0.0) - max_regression:
                reasons.append(f"{name} regressed by more than {max_regression:.6f}")
    return not reasons, tuple(reasons)


def build_bundle(
    models: EnsembleModels,
    selection: EnsembleSelection,
    *,
    model_version: str,
    dataset_hash: str,
    git_revision: str,
) -> EnsembleBundle:
    calibrated_models = EnsembleModels(
        if_model=models.if_model,
        ae_model=models.ae_model,
        ae_scaler=models.ae_scaler,
        ae_threshold_p99=models.ae_threshold_p99,
        sgd_model=models.sgd_model,
        model_source="trained_artifacts",
        model_version=model_version,
        calibrators=selection.calibrators,
        weights=selection.weights,
        threshold=selection.threshold,
    )
    metadata = BundleMetadata(
        model_version=model_version,
        dataset_hash=dataset_hash,
        git_revision=git_revision,
        metrics=selection.metrics,
        weights=selection.weights,
        threshold=selection.threshold,
    )
    return EnsembleBundle(models=calibrated_models, metadata=metadata)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_bundle(bundle: EnsembleBundle, output_path: str | Path) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    joblib.dump({"bundle_type": "calibrated_ensemble_v1", "bundle": bundle}, temporary)
    os.replace(temporary, output)
    checksum = _sha256(output)
    manifest = {"sha256": checksum, "metadata": asdict(bundle.metadata)}
    output.with_suffix(output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_bundle(path: str | Path) -> EnsembleBundle:
    bundle_path = Path(path)
    manifest_path = bundle_path.with_suffix(bundle_path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["sha256"] != _sha256(bundle_path):
        raise ValueError("Ensemble bundle checksum validation failed.")
    payload = joblib.load(bundle_path)
    if payload.get("bundle_type") != "calibrated_ensemble_v1" or not isinstance(payload.get("bundle"), EnsembleBundle):
        raise ValueError("Unsupported ensemble bundle payload.")
    return payload["bundle"]
