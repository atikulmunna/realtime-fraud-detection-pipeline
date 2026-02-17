from pathlib import Path

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import SGDClassifier

from src.common.feature_contract import FEATURES_V1
from src.common.metrics_stub import MetricsRegistry
from src.online.online_sgd_updater import OnlineSGDUpdater
from src.online.updater_service import create_online_service_app


def _seed_model(path: Path) -> None:
    model = SGDClassifier(loss="log_loss", random_state=1, max_iter=200, tol=1e-3)
    x = np.array([[0.1] * len(FEATURES_V1), [0.9] * len(FEATURES_V1)], dtype=float)
    y = np.array([0, 1], dtype=int)
    model.fit(x, y)
    joblib.dump({"model_type": "sgd_classifier", "model": model, "features_order": FEATURES_V1}, path)


class _IfModel:
    def decision_function(self, x):
        return np.array([0.0] * len(x))


class _Scaler:
    def transform(self, x):
        return x


class _AeModel:
    def predict(self, x):
        return np.zeros_like(x)


class _SgdModel:
    def predict_proba(self, x):
        return np.array([[0.1, 0.9] for _ in range(len(x))])


class _Models:
    if_model = _IfModel()
    ae_model = _AeModel()
    ae_scaler = _Scaler()
    ae_threshold_p99 = 1.0
    sgd_model = _SgdModel()


def _feedback_payload() -> dict:
    return {
        "label": "true_positive",
        "features": {k: 1.0 for k in FEATURES_V1},
    }


def _stream_event_payload() -> dict:
    return {
        "event_id": "evt-1",
        "timestamp": "2026-02-17T12:00:00Z",
        "user_id": "C100",
        "type": "TRANSFER",
        "amount": 600.0,
        "old_balance_orig": 1000.0,
        "new_balance_orig": 400.0,
    }


def test_online_service_emits_feedback_and_stream_metrics(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    metrics = MetricsRegistry()
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=1, metrics=metrics)
    app = create_online_service_app(updater=updater, metrics=metrics, stream_models=_Models())
    client = TestClient(app)

    fb = client.post("/feedback", json=_feedback_payload())
    assert fb.status_code == 200
    assert fb.json()["accepted"] is True
    assert fb.json()["updated"] is True

    st = client.post("/stream/event", json={"event": _stream_event_payload(), "txn_velocity_1h": 1})
    assert st.status_code == 200
    assert st.json()["topic"] in {"anomalies", "metrics"}

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    text = metrics_resp.text
    assert "online_feedback_received_total" in text
    assert "online_updates_total" in text
    assert "stream_events_in_total" in text
    assert "stream_process_latency_ms_total" in text


def test_online_service_stream_sample_endpoint(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    metrics = MetricsRegistry()
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=2, metrics=metrics)
    app = create_online_service_app(updater=updater, metrics=metrics, stream_models=_Models())
    client = TestClient(app)

    resp = client.post("/stream/sample")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["anomalies"] + payload["metrics"] + payload["dead-letter"] == 2

    fb = client.post("/feedback/sample")
    assert fb.status_code == 200
    assert fb.json()["updated"] is True

    metrics_resp = client.get("/metrics")
    assert "online_updates_total" in metrics_resp.text
