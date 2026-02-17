"""Local demo flow runner for parse -> score -> feedback -> online-update."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import SGDClassifier

from src.api.main import create_app
from src.common.feature_contract import FEATURES_V1
from src.common.metrics_stub import MetricsRegistry
from src.online.online_sgd_updater import (
    InMemoryModelUpdatePublisher,
    OnlineSGDUpdater,
    process_feedback_messages,
)
from src.streaming.ensemble_scoring import EnsembleModels
from src.streaming.pipeline_skeleton import process_stream_batch


class DemoIFModel:
    def decision_function(self, x):
        amount = x[:, 0]
        return -(amount - 100.0) / 100.0


class DemoScaler:
    def transform(self, x):
        return x / 1000.0


class DemoAEModel:
    def predict(self, x):
        out = np.array(x, copy=True)
        # Create reconstruction error for larger amounts.
        out[:, 0] = np.minimum(out[:, 0], 0.12)
        return out


class DemoSGDModel:
    def predict_proba(self, x):
        amount = np.clip(x[:, 0] / 1000.0, 0.0, 1.0)
        p1 = np.clip(0.1 + (0.8 * amount), 0.0, 1.0)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T


class _RecorderPublisher:
    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def publish(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)


def _seed_sgd_model(path: Path) -> None:
    model = SGDClassifier(loss="log_loss", random_state=1, max_iter=200, tol=1e-3)
    x = np.array([[0.1] * len(FEATURES_V1), [0.9] * len(FEATURES_V1)], dtype=float)
    y = np.array([0, 1], dtype=int)
    model.fit(x, y)
    payload = {"model_type": "sgd_classifier", "model": model, "features_order": FEATURES_V1}
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def _demo_payloads() -> list[str | bytes | dict[str, Any]]:
    return [
        {
            "event_id": "evt-high-1",
            "timestamp": "2026-02-16T08:15:00Z",
            "user_id": "C100",
            "type": "TRANSFER",
            "amount": 900.0,
            "old_balance_orig": 1000.0,
            "new_balance_orig": 100.0,
        },
        {
            "event_id": "evt-low-1",
            "timestamp": "2026-02-16T08:16:00Z",
            "user_id": "C101",
            "type": "PAYMENT",
            "amount": 20.0,
            "old_balance_orig": 500.0,
            "new_balance_orig": 480.0,
        },
        "{bad json}",
    ]


def run_demo_flow(*, model_path: str | Path = "models/sgd_classifier_demo.joblib") -> dict[str, Any]:
    models = EnsembleModels(
        if_model=DemoIFModel(),
        ae_model=DemoAEModel(),
        ae_scaler=DemoScaler(),
        ae_threshold_p99=0.01,
        sgd_model=DemoSGDModel(),
    )

    parsed = process_stream_batch(
        _demo_payloads(),
        models=models,
        score_threshold=0.65,
        anomaly_topic="anomalies",
        normal_topic="metrics",
        dlq_topic="dead-letter",
        txn_velocity_1h=1,
    )

    anomaly_events = parsed["anomalies"]
    metrics_events = parsed["metrics"]
    dlq_events = parsed["dead-letter"]

    feedback_publisher = _RecorderPublisher()
    api_metrics = MetricsRegistry()
    app = create_app(publisher=feedback_publisher, metrics=api_metrics)
    client = TestClient(app)

    for ev in anomaly_events:
        client.post(
            "/feedback",
            json={
                "anomaly_id": ev["event_id"],
                "label": "true_positive",
                "analyst_id": "demo-analyst",
                "features": ev["features"],
            },
        )

    model_path = Path(model_path)
    if not model_path.exists():
        _seed_sgd_model(model_path)

    update_publisher = InMemoryModelUpdatePublisher(events=[])
    online_metrics = MetricsRegistry()
    updater = OnlineSGDUpdater(
        model_path=model_path,
        batch_size=max(1, len(feedback_publisher.events)),
        publisher=update_publisher,
        metrics=online_metrics,
    )
    update_result = process_feedback_messages(feedback_publisher.events, updater=updater, force_flush=True)

    return {
        "events_in": len(_demo_payloads()),
        "anomalies": len(anomaly_events),
        "metrics": len(metrics_events),
        "dlq": len(dlq_events),
        "feedback_published": len(feedback_publisher.events),
        "online_updated": update_result.updated,
        "online_update_count": (
            int(update_result.signal["online_update_count"]) if update_result.signal else 0
        ),
        "signals_emitted": len(update_publisher.events),
        "api_feedback_requests_total": api_metrics.get_counter("feedback_requests_total"),
        "online_updates_total": online_metrics.get_counter("online_updates_total"),
    }
