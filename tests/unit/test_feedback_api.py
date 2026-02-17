from fastapi.testclient import TestClient

from src.api.main import create_app
from src.common.metrics_stub import MetricsRegistry


class _RecorderPublisher:
    def __init__(self):
        self.events = []

    def publish(self, payload):
        self.events.append(payload)


class _FailingPublisher:
    def publish(self, payload):
        raise RuntimeError("kafka down")


def test_feedback_api_happy_path_accepts_and_publishes():
    publisher = _RecorderPublisher()
    metrics = MetricsRegistry()
    app = create_app(publisher=publisher, metrics=metrics)
    client = TestClient(app)

    resp = client.post(
        "/feedback",
        json={
            "anomaly_id": "a-1",
            "label": "true_positive",
            "analyst_id": "analyst-1",
            "features": {"amount": 10.0},
        },
    )
    assert resp.status_code == 202
    payload = resp.json()
    assert payload["status"] == "accepted"
    assert "published_at" in payload
    assert len(publisher.events) == 1
    assert publisher.events[0]["label"] == "true_positive"
    assert "received_at" in publisher.events[0]
    assert metrics.get_counter("feedback_requests_total") == 1.0
    assert metrics.get_counter("feedback_published_total") == 1.0


def test_feedback_api_invalid_label_rejected():
    app = create_app(publisher=_RecorderPublisher())
    client = TestClient(app)
    resp = client.post(
        "/feedback",
        json={
            "anomaly_id": "a-1",
            "label": "wrong",
            "analyst_id": "analyst-1",
        },
    )
    assert resp.status_code == 422


def test_feedback_api_extra_field_rejected():
    app = create_app(publisher=_RecorderPublisher())
    client = TestClient(app)
    resp = client.post(
        "/feedback",
        json={
            "anomaly_id": "a-1",
            "label": "false_positive",
            "analyst_id": "analyst-1",
            "unexpected": 123,
        },
    )
    assert resp.status_code == 422


def test_feedback_api_publisher_failure_returns_503():
    metrics = MetricsRegistry()
    app = create_app(publisher=_FailingPublisher(), metrics=metrics)
    client = TestClient(app)
    resp = client.post(
        "/feedback",
        json={
            "anomaly_id": "a-1",
            "label": "false_positive",
            "analyst_id": "analyst-1",
        },
    )
    assert resp.status_code == 503
    assert "Publisher unavailable" in resp.json()["detail"]
    assert metrics.get_counter("feedback_requests_total") == 1.0
    assert metrics.get_counter("feedback_publish_errors_total") == 1.0


def test_feedback_api_metrics_endpoint():
    metrics = MetricsRegistry()
    app = create_app(publisher=_RecorderPublisher(), metrics=metrics)
    client = TestClient(app)
    client.post(
        "/feedback",
        json={
            "anomaly_id": "a-1",
            "label": "false_positive",
            "analyst_id": "analyst-1",
        },
    )
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "feedback_requests_total" in resp.text
