"""Ensemble scoring utilities for streaming inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np

from src.common.feature_contract import FEATURE_CONTRACT_VERSION, FEATURES_V1


@dataclass(frozen=True)
class EnsembleModels:
    if_model: Any
    ae_model: Any
    ae_scaler: Any
    ae_threshold_p99: float
    sgd_model: Any
    model_source: str = "trained_artifacts"
    model_version: str = "v1"
    feature_contract_version: str = FEATURE_CONTRACT_VERSION
    calibrators: dict[str, Any] | None = None
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    threshold: float = 0.5


def load_ensemble_models(
    *,
    if_model_path: str | Path = "models/isolation_forest_v1.joblib",
    ae_model_path: str | Path = "models/autoencoder_v1.joblib",
    sgd_model_path: str | Path = "models/sgd_classifier_v1.joblib",
) -> EnsembleModels:
    if_model = joblib.load(if_model_path)

    ae_payload = joblib.load(ae_model_path)
    ae_model = ae_payload["model"]
    ae_scaler = ae_payload["scaler"]
    ae_threshold = float(ae_payload["threshold_p99"])

    sgd_payload = joblib.load(sgd_model_path)
    sgd_model = sgd_payload["model"]

    ae_features = ae_payload.get("features_order", FEATURES_V1)
    sgd_features = sgd_payload.get("features_order", FEATURES_V1)
    if list(ae_features) != FEATURES_V1 or list(sgd_features) != FEATURES_V1:
        raise ValueError("Model artifact feature order does not match FEATURES_V1.")

    model_version = str(sgd_payload.get("model_version", ae_payload.get("model_version", "v1")))
    contract_version = str(
        sgd_payload.get(
            "feature_contract_version",
            ae_payload.get("feature_contract_version", FEATURE_CONTRACT_VERSION),
        )
    )
    if contract_version != FEATURE_CONTRACT_VERSION:
        raise ValueError(
            f"Model feature contract version {contract_version!r} does not match {FEATURE_CONTRACT_VERSION!r}."
        )

    return EnsembleModels(
        if_model=if_model,
        ae_model=ae_model,
        ae_scaler=ae_scaler,
        ae_threshold_p99=ae_threshold,
        sgd_model=sgd_model,
        model_source="trained_artifacts",
        model_version=model_version,
        feature_contract_version=contract_version,
    )


def _to_vector(features: dict[str, Any]) -> np.ndarray:
    missing = [f for f in FEATURES_V1 if f not in features]
    if missing:
        raise ValueError(f"Missing required feature keys: {missing}")
    return np.array([[float(features[f]) for f in FEATURES_V1]], dtype=float)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _normalize_weights(weights: tuple[float, float, float]) -> tuple[float, float, float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Ensemble weights must have positive sum.")
    normalized = tuple(float(w / total) for w in weights)
    return cast(tuple[float, float, float], normalized)


def score_event_features(
    features: dict[str, Any],
    models: EnsembleModels,
    *,
    weights: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    selected_weights = weights or cast(
        tuple[float, float, float],
        getattr(models, "weights", (0.4, 0.3, 0.3)),
    )
    w_if, w_ae, w_sgd = _normalize_weights(selected_weights)
    x = _to_vector(features)

    raw_if = float(models.if_model.decision_function(x)[0])
    if_score = _sigmoid(-raw_if)

    x_scaled = models.ae_scaler.transform(x)
    x_recon = models.ae_model.predict(x_scaled)
    mse = float(np.mean((x_recon - x_scaled) ** 2))
    ae_score = float(min(max(mse / max(models.ae_threshold_p99, 1e-12), 0.0), 1.0))

    sgd_score = float(models.sgd_model.predict_proba(x)[0][1])

    raw_scores = {"if": if_score, "ae": ae_score, "sgd": sgd_score}
    calibrators = getattr(models, "calibrators", None) or {}
    calibrated_scores = {
        name: float(calibrators[name].transform(np.array([score], dtype=float))[0]) if name in calibrators else score
        for name, score in raw_scores.items()
    }
    ensemble_score = float(
        (w_if * calibrated_scores["if"]) + (w_ae * calibrated_scores["ae"]) + (w_sgd * calibrated_scores["sgd"])
    )
    result = {
        "if_score": calibrated_scores["if"],
        "ae_score": calibrated_scores["ae"],
        "sgd_score": calibrated_scores["sgd"],
        "ensemble_score": ensemble_score,
    }
    if calibrators:
        result.update({f"raw_{name}_score": score for name, score in raw_scores.items()})
    return result


def route_score_to_topic(
    ensemble_score: float,
    *,
    threshold: float = 0.5,
    anomaly_topic: str = "anomalies",
    normal_topic: str = "metrics",
) -> str:
    return anomaly_topic if float(ensemble_score) >= float(threshold) else normal_topic
