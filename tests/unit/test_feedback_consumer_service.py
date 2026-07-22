from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from src.common.feature_contract import FEATURES_V1
from src.common.metrics_stub import MetricsRegistry
from src.online.feedback_consumer_service import _consumer_lag, run_feedback_consumer_loop
from src.online.model_promotion import PromotionThresholds
from src.online.online_sgd_updater import OnlineSGDUpdater


class _Record:
    def __init__(self, value):
        self.value = value


class _FakeConsumer:
    def __init__(self, polls):
        self._polls = list(polls)
        self.closed = False
        self.commits = 0

    def poll(self, timeout_ms=0, max_records=None):
        if not self._polls:
            return {}
        return self._polls.pop(0)

    def close(self):
        self.closed = True

    def commit(self):
        self.commits += 1


class _FakeRegistry:
    def __init__(self):
        self.registered = []
        self.promoted = []

    def register_candidate(self, candidate_path, metadata):
        assert Path(candidate_path).is_file()
        self.registered.append((Path(candidate_path), metadata))
        return "7"

    def promote_candidate(self, version):
        self.promoted.append(version)


def test_consumer_lag_sums_assigned_partition_distance():
    class _LagConsumer:
        def assignment(self):
            return {"p0", "p1"}

        def end_offsets(self, partitions):
            return {"p0": 12, "p1": 7}

        def position(self, partition):
            return {"p0": 9, "p1": 2}[partition]

    assert _consumer_lag(_LagConsumer()) == 8


def _seed_model(path: Path) -> None:
    model = SGDClassifier(loss="log_loss", random_state=1, max_iter=200, tol=1e-3)
    x = np.array([[0.1] * len(FEATURES_V1), [0.9] * len(FEATURES_V1)], dtype=float)
    y = np.array([0, 1], dtype=int)
    model.fit(x, y)
    joblib.dump({"model_type": "sgd_classifier", "model": model, "features_order": FEATURES_V1}, path)


def _feedback(label: str, value: float) -> dict:
    return {"label": label, "features": {k: value for k in FEATURES_V1}}


def test_feedback_consumer_loop_processes_messages_and_updates(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    metrics = MetricsRegistry()
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=2, metrics=metrics)
    consumer = _FakeConsumer(
        [
            {"tp": [_Record(_feedback("true_positive", 0.9)), _Record(_feedback("false_positive", 0.2))]},
            {},
        ]
    )

    out = run_feedback_consumer_loop(
        updater=updater,
        consumer=consumer,
        metrics=metrics,
        poll_timeout_ms=1,
        flush_interval_s=100.0,
        max_messages=2,
        max_idle_polls=1,
        force_flush_on_exit=False,
    )

    assert out["messages_seen"] == 2
    assert out["accepted"] == 2
    assert out["updates"] == 1
    assert consumer.closed is True
    assert metrics.get_counter("online_consumer_messages_total") == 2.0
    assert metrics.get_counter("online_updates_total") == 1.0
    assert consumer.commits == 1
    assert out["commits"] == 1


def test_feedback_consumer_registers_then_promotes_before_commit(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=1)
    consumer = _FakeConsumer([{"tp": [_Record(_feedback("true_positive", 0.9))]}])
    registry = _FakeRegistry()

    out = run_feedback_consumer_loop(
        updater=updater,
        consumer=consumer,
        poll_timeout_ms=1,
        flush_interval_s=100.0,
        max_messages=1,
        force_flush_on_exit=False,
        candidate_registry=registry,
    )

    assert out["commits"] == 1
    assert len(registry.registered) == 1
    assert registry.promoted == ["7"]
    assert not updater.candidate_path.exists()


def test_feedback_consumer_loop_idle_exit_force_flush(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=50)
    consumer = _FakeConsumer(
        [{"tp": [_Record(_feedback("true_positive", 0.8)), _Record(_feedback("false_positive", 0.1))]}, {}, {}]
    )

    out = run_feedback_consumer_loop(
        updater=updater,
        consumer=consumer,
        poll_timeout_ms=1,
        flush_interval_s=100.0,
        max_idle_polls=2,
        force_flush_on_exit=True,
    )
    assert out["messages_seen"] == 2
    assert out["updates"] == 1
    assert consumer.closed is True


def test_feedback_consumer_interval_flushes_partial_batch_and_commits(tmp_path: Path, monkeypatch):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=50)
    consumer = _FakeConsumer([{"tp": [_Record(_feedback("true_positive", 0.8))]}])
    ticks = iter([0.0, 2.0, 3.0])
    monkeypatch.setattr("src.online.feedback_consumer_service.time.monotonic", lambda: next(ticks))

    out = run_feedback_consumer_loop(
        updater=updater,
        consumer=consumer,
        poll_timeout_ms=1,
        flush_interval_s=1.0,
        max_messages=1,
        force_flush_on_exit=False,
    )

    assert out["updates"] == 1
    assert consumer.commits == 1


def test_feedback_consumer_loop_promotion_guardrail_failure_rolls_back(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)

    holdout_rows = []
    for i in range(10):
        y = 1 if i % 2 == 0 else 0
        v = 0.9 if y == 1 else 0.1
        row = {k: v for k in FEATURES_V1}
        row["isFraud"] = y
        holdout_rows.append(row)
    holdout = tmp_path / "holdout.parquet"
    import pandas as pd

    pd.DataFrame(holdout_rows).to_parquet(holdout, index=False)

    updater = OnlineSGDUpdater(model_path=model_path, batch_size=2)
    consumer = _FakeConsumer(
        [{"tp": [_Record(_feedback("true_positive", 0.9)), _Record(_feedback("false_positive", 0.2))]}]
    )

    out = run_feedback_consumer_loop(
        updater=updater,
        consumer=consumer,
        poll_timeout_ms=1,
        flush_interval_s=100.0,
        max_messages=2,
        force_flush_on_exit=False,
        promotion_holdout_parquet=holdout,
        promotion_thresholds=PromotionThresholds(min_precision=1.1),
    )
    assert out["updates"] == 1
    assert out["promotion_failed"] == 1
    assert out["rollbacks"] == 1
