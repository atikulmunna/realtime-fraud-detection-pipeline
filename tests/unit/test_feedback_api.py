from fastapi.testclient import TestClient

from src.api.app_entry import app_factory
from src.api.feedback_publisher import (
    ConfluentFeedbackPublisher,
    KafkaFeedbackPublisher,
    build_kafka_feedback_publisher,
    build_reliable_feedback_publisher,
)
from src.api.feedback_store import SqlFeedbackStore
from src.api.main import create_app
from src.api.outbox_relay import relay_once
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


def test_kafka_feedback_publisher_sends_and_flushes():
    class _Producer:
        def __init__(self):
            self.sent = []
            self.flush_count = 0

        def send(self, topic, payload):
            self.sent.append((topic, payload))

        def flush(self):
            self.flush_count += 1

    producer = _Producer()
    publisher = KafkaFeedbackPublisher(producer=producer, topic="feedback")
    publisher.publish({"label": "true_positive"})

    assert producer.sent == [("feedback", {"label": "true_positive"})]
    assert producer.flush_count == 1


def test_build_kafka_feedback_publisher_configures_producer(monkeypatch):
    captured = {}

    def fake_producer(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("src.api.feedback_publisher.KafkaProducer", fake_producer)
    publisher = build_kafka_feedback_publisher(bootstrap_servers="broker:9092", topic="labels")

    assert publisher.topic == "labels"
    assert captured["bootstrap_servers"] == ["broker:9092"]
    assert captured["value_serializer"]({"x": 1}) == b'{"x": 1}'


def test_reliable_feedback_publisher_uses_idempotent_acknowledged_delivery(monkeypatch):
    captured = {}

    class _Producer:
        def __init__(self, config):
            captured["config"] = config
            captured["produced"] = []

        def produce(self, topic, *, value, key):
            captured["produced"].append((topic, value, key))

        def flush(self, timeout):
            captured["flush_timeout"] = timeout
            return 0

    monkeypatch.setattr("src.api.feedback_publisher.Producer", _Producer)
    publisher = build_reliable_feedback_publisher(bootstrap_servers="broker:9092", topic="feedback")
    publisher.publish({"feedback_id": "fb-1", "label": "true_positive"})

    assert captured["config"]["enable.idempotence"] is True
    assert captured["config"]["acks"] == "all"
    assert captured["produced"][0][2] == "fb-1"


def test_reliable_publisher_supports_an_explicit_message_key(monkeypatch):
    captured = {}

    class _Producer:
        def __init__(self, config):
            captured["config"] = config

        def produce(self, topic, *, value, key):
            captured["produced"] = (topic, value, key)

        def flush(self, timeout):
            return 0

    monkeypatch.setattr("src.api.feedback_publisher.Producer", _Producer)
    publisher = build_reliable_feedback_publisher(
        bootstrap_servers="broker:9092",
        topic="model-updates",
        key_field="model_type",
    )
    publisher.publish({"model_type": "sgd_classifier", "online_update_count": 1})

    assert captured["produced"][2] == "sgd_classifier"


def test_reliable_feedback_publisher_rejects_unacknowledged_delivery():
    class _Producer:
        def produce(self, topic, *, value, key):
            return None

        def flush(self, timeout):
            return 1

    publisher = ConfluentFeedbackPublisher(producer=_Producer(), topic="feedback")
    try:
        publisher.publish({"feedback_id": "fb-1"})
    except RuntimeError as exc:
        assert "did not acknowledge" in str(exc)
    else:
        raise AssertionError("Expected unacknowledged delivery to fail")


def test_app_factory_uses_environment_configuration(monkeypatch):
    publisher = _RecorderPublisher()
    captured = {}
    monkeypatch.setenv("APP_ENV", "development")

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return publisher

    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker:19092")
    monkeypatch.setenv("FEEDBACK_TOPIC", "analyst-feedback")
    monkeypatch.setattr("src.api.app_entry.build_kafka_feedback_publisher", fake_builder)

    app = app_factory()

    assert app.title == "Realtime Fraud Feedback API"
    assert captured == {"bootstrap_servers": "broker:19092", "topic": "analyst-feedback"}


def test_feedback_api_requires_configured_api_key():
    client = TestClient(create_app(publisher=_RecorderPublisher(), api_key="secret"))
    payload = {"anomaly_id": "a-1", "label": "true_positive", "analyst_id": "analyst-1"}

    assert client.post("/feedback", json=payload).status_code == 401
    assert client.post("/feedback", json=payload, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/feedback", json=payload, headers={"X-API-Key": "secret"}).status_code == 202


def test_feedback_store_is_idempotent_and_outbox_relay_publishes_once(tmp_path):
    store = SqlFeedbackStore(f"sqlite:///{tmp_path / 'feedback.db'}", create_schema=True)
    client = TestClient(create_app(feedback_store=store))
    payload = {"anomaly_id": "a-1", "label": "true_positive", "analyst_id": "analyst-1"}
    headers = {"Idempotency-Key": "request-123"}

    first = client.post("/feedback", json=payload, headers=headers)
    duplicate = client.post("/feedback", json=payload, headers=headers)

    assert first.status_code == 202
    assert first.json()["duplicate"] is False
    assert duplicate.status_code == 202
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["feedback_id"] == first.json()["feedback_id"]
    assert len(store.pending()) == 1

    publisher = _RecorderPublisher()
    metrics = MetricsRegistry()
    result = relay_once(store=store, publisher=publisher, metrics=metrics)
    second_result = relay_once(store=store, publisher=publisher, metrics=metrics)

    assert result.published == 1
    assert second_result.published == 0
    assert len(publisher.events) == 1
    assert metrics.get_counter("outbox_published_total") == 1
    assert metrics.get_gauge("outbox_backlog") == 0
