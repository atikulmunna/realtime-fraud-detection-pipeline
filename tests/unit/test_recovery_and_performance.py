import os
from pathlib import Path
from time import perf_counter

import numpy as np

from src.api.feedback_store import SqlFeedbackStore
from src.api.outbox_relay import relay_once
from src.demo.demo_flow import build_demo_models
from src.streaming.flink_job import is_late_event
from src.streaming.pipeline_skeleton import process_stream_payload


class _FailingPublisher:
    def publish(self, payload):
        raise RuntimeError("kafka unavailable")


class _RecorderPublisher:
    def __init__(self):
        self.events = []

    def publish(self, payload):
        self.events.append(payload)


def test_acknowledged_outbox_record_survives_publish_failure_and_relay_restart(tmp_path: Path):
    store = SqlFeedbackStore(f"sqlite:///{tmp_path / 'feedback.db'}", create_schema=True)
    payload = {"feedback_id": "feedback-1", "label": "true_positive"}
    stored = store.accept(feedback_id="feedback-1", idempotency_key="request-1", payload=payload)

    failed = relay_once(store=store, publisher=_FailingPublisher())
    assert stored.duplicate is False
    assert failed.failed == 1
    assert store.pending_count() == 1

    publisher = _RecorderPublisher()
    recovered = relay_once(store=store, publisher=publisher)
    assert recovered.published == 1
    assert store.pending_count() == 0
    assert publisher.events == [payload]


def test_late_event_boundary_matches_two_minute_watermark_contract():
    watermark = 1_000_000
    assert is_late_event(watermark - 1, watermark) is True
    assert is_late_event(watermark, watermark) is False
    assert is_late_event(watermark - 500_000, -1) is False


def test_local_scoring_p95_meets_configurable_latency_slo():
    models = build_demo_models()
    samples_ms = []
    event = {
        "event_id": "perf-event",
        "timestamp": "2026-07-20T08:00:00Z",
        "user_id": "customer-1",
        "type": "TRANSFER",
        "amount": 900.0,
        "old_balance_orig": 1000.0,
        "new_balance_orig": 100.0,
    }
    for _ in range(100):
        started = perf_counter()
        topic, _ = process_stream_payload(event, models=models)
        samples_ms.append((perf_counter() - started) * 1000.0)
        assert topic in {"anomalies", "metrics"}

    p95_ms = float(np.percentile(samples_ms, 95))
    slo_ms = float(os.getenv("LOCAL_E2E_P95_SLO_MS", "500"))
    assert p95_ms <= slo_ms, f"Local scoring p95 {p95_ms:.2f} ms exceeded {slo_ms:.2f} ms."
