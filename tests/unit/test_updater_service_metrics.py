from fastapi.testclient import TestClient

from src.common.metrics_stub import MetricsRegistry
from src.online.updater_service import create_updater_metrics_app


def test_updater_metrics_app_health_and_metrics():
    metrics = MetricsRegistry()
    metrics.inc("online_feedback_received_total", 3)
    metrics.set_gauge("online_updater_buffer_size", 1)

    app = create_updater_metrics_app(metrics=metrics)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    prom = client.get("/metrics")
    assert prom.status_code == 200
    assert "online_feedback_received_total" in prom.text
    assert "online_updater_buffer_size" in prom.text
