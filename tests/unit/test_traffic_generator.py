import json
import random
from collections import Counter
from datetime import UTC, datetime

import pytest

from src.demo.traffic_generator import (
    DRAINING_TYPES,
    ConfluentTransactionPublisher,
    RecordingTransactionPublisher,
    build_confluent_publisher,
    generate_transaction,
    main,
    run_generator,
)
from src.streaming.event_parser import parse_and_validate_event

FIXED_NOW = datetime(2026, 2, 16, 8, 15, tzinfo=UTC)


class FakeProducer:
    def __init__(self, unacked: int = 0):
        self.produced: list[tuple[str, bytes, str]] = []
        self.polls = 0
        self.flushed: list[float] = []
        self._unacked = unacked

    def produce(self, topic, value, key):
        self.produced.append((topic, value, key))

    def poll(self, timeout):
        self.polls += 1

    def flush(self, timeout):
        self.flushed.append(timeout)
        return self._unacked


class FakeClock:
    """Monotonic clock that advances only when the generator sleeps."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_generated_events_pass_the_real_event_contract():
    rng = random.Random(7)
    for _ in range(200):
        event = generate_transaction(rng, now=FIXED_NOW, fraud_ratio=0.5)
        result = parse_and_validate_event(event)
        assert result.ok, result.dlq


def test_benign_events_preserve_balance_arithmetic():
    rng = random.Random(11)
    for _ in range(200):
        event = generate_transaction(rng, now=FIXED_NOW, fraud_ratio=0.0)
        assert event["type"] not in DRAINING_TYPES
        expected = round(event["old_balance_orig"] - event["amount"], 2)
        assert event["new_balance_orig"] == pytest.approx(expected)
        assert event["new_balance_orig"] >= 0.0


def test_draining_events_zero_the_originating_balance():
    rng = random.Random(11)
    for _ in range(50):
        event = generate_transaction(rng, now=FIXED_NOW, fraud_ratio=1.0)
        assert event["type"] in DRAINING_TYPES
        assert event["new_balance_orig"] == 0.0
        assert event["amount"] == event["old_balance_orig"]
        assert event["amount"] >= 5_000.0


def test_same_seed_reproduces_the_same_event_stream():
    first = [generate_transaction(random.Random(3), now=FIXED_NOW) for _ in range(1)]
    second = [generate_transaction(random.Random(3), now=FIXED_NOW) for _ in range(1)]
    assert first == second


def test_user_ids_stay_within_the_requested_population():
    rng = random.Random(5)
    ids = {generate_transaction(rng, now=FIXED_NOW, users=4)["user_id"] for _ in range(100)}
    assert ids <= {"C0000000", "C0000001", "C0000002", "C0000003"}


def test_default_population_keeps_repeat_users_rare():
    """The models were trained on data where txn_velocity_1h is ~always 1.

    A small population makes the Flink job's per-user counter climb and flags
    nearly all traffic, so the default has to keep repeats within an hour rare.
    """
    rng = random.Random(9)
    hourly_events = 4 * 3600
    seen = Counter(generate_transaction(rng, now=FIXED_NOW)["user_id"] for _ in range(hourly_events // 4))
    repeat_rate = 1 - (len(seen) / sum(seen.values()))
    assert repeat_rate < 0.02, f"repeat-user rate {repeat_rate:.2%} would inflate txn_velocity_1h"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"users": 0}, "users must be at least 1"),
        ({"fraud_ratio": 1.5}, "fraud_ratio must be between"),
        ({"fraud_ratio": -0.1}, "fraud_ratio must be between"),
    ],
)
def test_generate_transaction_rejects_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate_transaction(random.Random(1), now=FIXED_NOW, **kwargs)


def test_run_generator_publishes_at_the_requested_rate():
    publisher = RecordingTransactionPublisher()
    clock = FakeClock()

    summary = run_generator(
        publisher,
        rate=10.0,
        duration=1.0,
        seed=1,
        clock=clock,
        sleeper=clock.sleep,
        now_factory=lambda: FIXED_NOW,
    )

    assert summary["published"] == 10
    assert len(publisher.events) == 10
    assert summary["elapsed_seconds"] == pytest.approx(1.0)


def test_run_generator_counts_draining_events():
    publisher = RecordingTransactionPublisher()
    clock = FakeClock()

    summary = run_generator(
        publisher,
        rate=5.0,
        duration=1.0,
        fraud_ratio=1.0,
        seed=1,
        clock=clock,
        sleeper=clock.sleep,
        now_factory=lambda: FIXED_NOW,
    )

    assert summary["published"] == 5
    assert summary["draining"] == 5


def test_run_generator_with_zero_duration_publishes_nothing():
    publisher = RecordingTransactionPublisher()
    clock = FakeClock()

    summary = run_generator(
        publisher,
        duration=0.0,
        clock=clock,
        sleeper=clock.sleep,
        now_factory=lambda: FIXED_NOW,
    )

    assert summary["published"] == 0
    assert publisher.events == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rate": 0.0}, "rate must be greater than 0"),
        ({"rate": -1.0}, "rate must be greater than 0"),
        ({"duration": -1.0}, "duration must not be negative"),
    ],
)
def test_run_generator_rejects_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_generator(RecordingTransactionPublisher(), **kwargs)


def test_confluent_publisher_keys_by_user_and_polls_without_blocking():
    producer = FakeProducer()
    publisher = ConfluentTransactionPublisher(producer=producer, topic="raw-events")

    publisher.publish({"user_id": "C00007", "event_id": "e1", "amount": 5.0})

    topic, value, key = producer.produced[0]
    assert topic == "raw-events"
    assert key == "C00007"
    assert json.loads(value.decode("utf-8"))["event_id"] == "e1"
    assert producer.polls == 1
    assert producer.flushed == []


def test_confluent_publisher_close_flushes_and_raises_on_unacknowledged():
    ok = ConfluentTransactionPublisher(producer=FakeProducer())
    ok.close()

    stuck = ConfluentTransactionPublisher(producer=FakeProducer(unacked=3))
    with pytest.raises(RuntimeError, match="did not acknowledge 3"):
        stuck.close()


def test_build_confluent_publisher_requests_idempotent_delivery(monkeypatch):
    captured: dict[str, object] = {}

    def fake_producer(config):
        captured.update(config)
        return FakeProducer()

    monkeypatch.setattr("src.demo.traffic_generator.Producer", fake_producer)
    publisher = build_confluent_publisher(bootstrap_servers="kafka:29092", topic="raw-events")

    assert publisher.topic == "raw-events"
    assert captured["bootstrap.servers"] == "kafka:29092"
    assert captured["enable.idempotence"] is True
    assert captured["acks"] == "all"


def test_recording_publisher_close_is_a_noop():
    publisher = RecordingTransactionPublisher()
    assert publisher.close() is None


def test_main_dry_run_reports_a_summary(capsys):
    main(["--dry-run", "--rate", "50", "--duration", "0.2", "--seed", "1"])

    summary = json.loads(capsys.readouterr().out)
    assert summary["published"] > 0
    assert summary["rate"] == 50.0


def test_main_reports_interruption(monkeypatch, capsys):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("src.demo.traffic_generator.run_generator", interrupt)
    main(["--dry-run", "--duration", "0.1"])

    summary = json.loads(capsys.readouterr().out)
    assert summary["interrupted"] is True
