from src.common.metrics_stub import MetricsRegistry
from src.streaming.pipeline_skeleton import process_stream_batch, process_stream_payload


def _valid_event(event_id: str = "evt-1") -> dict:
    return {
        "event_id": event_id,
        "timestamp": "2026-02-16T08:15:00Z",
        "user_id": "C100",
        "type": "TRANSFER",
        "amount": 120.0,
        "old_balance_orig": 500.0,
        "new_balance_orig": 380.0,
    }


def test_process_stream_payload_happy_path():
    metrics = MetricsRegistry()
    topic, payload = process_stream_payload(_valid_event(), txn_velocity_1h=4, metrics=metrics)
    assert topic == "feature-events"
    assert "features" in payload
    assert payload["features"]["txn_velocity_1h"] == 4
    assert payload["features"]["is_transfer"] == 1
    assert metrics.get_counter("stream_events_in_total") == 1.0
    assert metrics.get_counter("stream_events_valid_total") == 1.0
    assert metrics.get_gauge("stream_last_process_latency_ms") >= 0.0


def test_process_stream_payload_invalid_routes_dlq():
    bad = _valid_event()
    bad["type"] = "INVALID"
    topic, payload = process_stream_payload(bad)
    assert topic == "dead-letter"
    assert "Invalid value for 'type'" in payload["error"]


def test_process_stream_payload_strips_leakage_before_features():
    event = _valid_event()
    event["is_fraud"] = 1
    topic, payload = process_stream_payload(event)
    assert topic == "feature-events"
    assert "is_fraud" not in payload
    assert "features" in payload


def test_process_stream_batch_mixed_flow():
    batch = [
        _valid_event("evt-1"),
        {**_valid_event("evt-2"), "type": "BAD_TYPE"},
        "{bad json}",
        {**_valid_event("evt-3"), "type": "CASH-OUT"},
    ]
    out = process_stream_batch(batch, txn_velocity_1h=2)
    assert len(out["feature-events"]) == 2
    assert len(out["dead-letter"]) == 2
    assert out["feature-events"][1]["features"]["is_cashout"] == 1


class _IfModel:
    def __init__(self, decision=0.0):
        self._d = decision

    def decision_function(self, x):
        import numpy as np

        return np.array([self._d] * len(x))


class _Scaler:
    def transform(self, x):
        return x


class _AeModelZero:
    def predict(self, x):
        import numpy as np

        return np.zeros_like(x)


class _AeModelIdentity:
    def predict(self, x):
        return x


class _SgdModel:
    def __init__(self, p1):
        self._p1 = p1

    def predict_proba(self, x):
        import numpy as np

        return np.array([[1.0 - self._p1, self._p1] for _ in range(len(x))])


class _Models:
    def __init__(self, high=True):
        self.if_model = _IfModel(0.0)
        self.ae_scaler = _Scaler()
        self.ae_threshold_p99 = 1.0
        if high:
            self.ae_model = _AeModelZero()
            self.sgd_model = _SgdModel(0.9)
        else:
            self.ae_model = _AeModelIdentity()
            self.sgd_model = _SgdModel(0.1)


def test_process_stream_payload_with_scoring_routes_anomaly():
    topic, payload = process_stream_payload(
        _valid_event("evt-high"),
        models=_Models(high=True),
        score_threshold=0.7,
    )
    assert topic == "anomalies"
    assert "scores" in payload
    assert payload["scores"]["ensemble_score"] >= 0.7


def test_process_stream_payload_with_scoring_routes_normal():
    topic, payload = process_stream_payload(
        _valid_event("evt-low"),
        models=_Models(high=False),
        score_threshold=0.5,
    )
    assert topic == "metrics"
    assert "scores" in payload
    assert payload["scores"]["ensemble_score"] < 0.5


def test_process_stream_batch_with_scoring_mixed_topics_and_dlq():
    batch = [
        _valid_event("evt-high"),
        {**_valid_event("evt-bad"), "type": "BAD_TYPE"},
    ]
    out = process_stream_batch(
        batch,
        models=_Models(high=True),
        score_threshold=0.7,
        anomaly_topic="anomalies",
        normal_topic="metrics",
        dlq_topic="dead-letter",
    )
    assert len(out["anomalies"]) == 1
    assert len(out["metrics"]) == 0
    assert len(out["dead-letter"]) == 1


def test_process_stream_batch_with_metrics_counts_all_routes():
    batch = [
        _valid_event("evt-high"),
        _valid_event("evt-low"),
        {**_valid_event("evt-bad"), "type": "BAD_TYPE"},
    ]
    metrics = MetricsRegistry()
    out = process_stream_batch(
        batch,
        models=_Models(high=True),
        score_threshold=0.7,
        anomaly_topic="anomalies",
        normal_topic="metrics",
        dlq_topic="dead-letter",
        metrics=metrics,
    )
    assert len(out["anomalies"]) == 2
    assert len(out["metrics"]) == 0
    assert len(out["dead-letter"]) == 1
    assert metrics.get_counter("stream_events_in_total") == 3.0
    assert metrics.get_counter("stream_events_anomaly_total") == 2.0
    assert metrics.get_counter("stream_events_dlq_total") == 1.0
    assert metrics.get_counter("stream_process_latency_ms_total") > 0.0
