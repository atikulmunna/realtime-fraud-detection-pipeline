"""Minimal Flink job wrapper scaffold for integration handoff."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.metrics_stub import MetricsRegistry
from src.streaming.pipeline_skeleton import process_stream_batch


@dataclass(frozen=True)
class FlinkJobConfig:
    input_topic: str = "raw-events"
    anomaly_topic: str = "anomalies"
    metrics_topic: str = "metrics"
    dlq_topic: str = "dead-letter"
    threshold: float = 0.5
    if_model_path: str = "models/isolation_forest_v1.joblib"
    ae_model_path: str = "models/autoencoder_v1.joblib"
    sgd_model_path: str = "models/sgd_classifier_v1.joblib"


def load_config_from_env(env: dict[str, str] | None = None) -> FlinkJobConfig:
    import os

    src = env or os.environ
    return FlinkJobConfig(
        input_topic=src.get("RAW_EVENTS_TOPIC", "raw-events"),
        anomaly_topic=src.get("ANOMALIES_TOPIC", "anomalies"),
        metrics_topic=src.get("METRICS_TOPIC", "metrics"),
        dlq_topic=src.get("DLQ_TOPIC", "dead-letter"),
        threshold=float(src.get("ANOMALY_THRESHOLD", "0.5")),
        if_model_path=src.get("IF_MODEL_PATH", "models/isolation_forest_v1.joblib"),
        ae_model_path=src.get("AE_MODEL_PATH", "models/autoencoder_v1.joblib"),
        sgd_model_path=src.get("SGD_MODEL_PATH", "models/sgd_classifier_v1.joblib"),
    )


def operator_wiring_contract() -> list[str]:
    return [
        "EventParser",
        "FeatureExtractor",
        "EnsembleScoring",
        "ThresholdRouter",
    ]


def validate_model_paths(config: FlinkJobConfig) -> dict[str, bool]:
    return {
        "if_model_exists": Path(config.if_model_path).exists(),
        "ae_model_exists": Path(config.ae_model_path).exists(),
        "sgd_model_exists": Path(config.sgd_model_path).exists(),
    }


def run_local_wrapper_batch(
    payloads: list[str | bytes | dict[str, Any]],
    *,
    models: Any,
    config: FlinkJobConfig | None = None,
    metrics: MetricsRegistry | None = None,
) -> dict[str, list[dict[str, Any]]]:
    cfg = config or load_config_from_env()
    return process_stream_batch(
        payloads,
        models=models,
        score_threshold=cfg.threshold,
        anomaly_topic=cfg.anomaly_topic,
        normal_topic=cfg.metrics_topic,
        dlq_topic=cfg.dlq_topic,
        metrics=metrics,
    )
