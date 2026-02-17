"""Ensemble scoring utilities for streaming inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np

from src.common.feature_contract import FEATURES_V1


@dataclass(frozen=True)
class EnsembleModels:
    if_model: Any
    ae_model: Any
    ae_scaler: Any
    ae_threshold_p99: float
    sgd_model: Any


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

    return EnsembleModels(
        if_model=if_model,
        ae_model=ae_model,
        ae_scaler=ae_scaler,
        ae_threshold_p99=ae_threshold,
        sgd_model=sgd_model,
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
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> dict[str, float]:
    w_if, w_ae, w_sgd = _normalize_weights(weights)
    x = _to_vector(features)

    raw_if = float(models.if_model.decision_function(x)[0])
    if_score = _sigmoid(-raw_if)

    x_scaled = models.ae_scaler.transform(x)
    x_recon = models.ae_model.predict(x_scaled)
    mse = float(np.mean((x_recon - x_scaled) ** 2))
    ae_score = float(min(max(mse / max(models.ae_threshold_p99, 1e-12), 0.0), 1.0))

    sgd_score = float(models.sgd_model.predict_proba(x)[0][1])

    ensemble_score = float((w_if * if_score) + (w_ae * ae_score) + (w_sgd * sgd_score))
    return {
        "if_score": if_score,
        "ae_score": ae_score,
        "sgd_score": sgd_score,
        "ensemble_score": ensemble_score,
    }


def route_score_to_topic(
    ensemble_score: float,
    *,
    threshold: float = 0.5,
    anomaly_topic: str = "anomalies",
    normal_topic: str = "metrics",
) -> str:
    return anomaly_topic if float(ensemble_score) >= float(threshold) else normal_topic
