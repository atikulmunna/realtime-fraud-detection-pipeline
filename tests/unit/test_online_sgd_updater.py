from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from src.common.feature_contract import FEATURES_V1
from src.common.metrics_stub import MetricsRegistry
from src.online.online_sgd_updater import (
    InMemoryModelUpdatePublisher,
    OnlineSGDUpdater,
    process_feedback_messages,
)


def _seed_model(path: Path) -> None:
    model = SGDClassifier(loss="log_loss", random_state=1, max_iter=200, tol=1e-3)
    x = np.array(
        [
            [0.1] * len(FEATURES_V1),
            [0.9] * len(FEATURES_V1),
            [0.2] * len(FEATURES_V1),
            [0.8] * len(FEATURES_V1),
        ],
        dtype=float,
    )
    y = np.array([0, 1, 0, 1], dtype=int)
    model.fit(x, y)
    joblib.dump({"model_type": "sgd_classifier", "model": model, "features_order": FEATURES_V1}, path)


def _feedback(label: str, v: float) -> dict:
    return {
        "anomaly_id": f"a-{v}",
        "label": label,
        "analyst_id": "u1",
        "features": {k: v for k in FEATURES_V1},
    }


def test_online_updater_buffers_and_flushes_at_batch_size(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    publisher = InMemoryModelUpdatePublisher(events=[])
    metrics = MetricsRegistry()
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=2, publisher=publisher, metrics=metrics)

    assert updater.add_feedback(_feedback("true_positive", 0.95)) is True
    pre = updater.flush()
    assert pre.updated is False
    assert pre.batch_size == 1

    updater.add_feedback(_feedback("false_positive", 0.2))
    res = updater.flush()
    assert res.updated is True
    assert res.batch_size == 2
    assert res.signal is not None
    assert res.signal["online_update_count"] == 1
    assert len(publisher.events) == 1
    assert metrics.get_counter("online_feedback_received_total") == 2.0
    assert metrics.get_counter("online_feedback_accepted_total") == 2.0
    assert metrics.get_counter("online_updates_total") == 1.0
    assert metrics.get_gauge("online_updater_buffer_size") == 0.0


def test_online_updater_skips_invalid_feedback(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    metrics = MetricsRegistry()
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=2, metrics=metrics)

    assert updater.add_feedback({"label": "bad", "features": {}}) is False
    assert updater.add_feedback({"label": "true_positive"}) is False
    res = updater.flush(force=True)
    assert res.updated is False
    assert res.skipped == 2
    assert metrics.get_counter("online_feedback_skipped_total") == 2.0


def test_online_updater_persists_online_update_metadata(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=2)
    updater.add_feedback(_feedback("true_positive", 0.9))
    updater.add_feedback(_feedback("false_positive", 0.1))
    res = updater.flush()
    assert res.updated is True

    saved = joblib.load(model_path)
    assert saved["online_update_count"] == 1
    assert "last_updated_at" in saved


def test_process_feedback_messages_wrapper(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=3)

    msgs = [
        _feedback("true_positive", 0.8),
        _feedback("false_positive", 0.3),
        _feedback("true_positive", 0.9),
    ]
    res = process_feedback_messages(msgs, updater=updater, force_flush=False)
    assert res.updated is True
    assert res.batch_size == 3


def test_online_updater_persists_feedback_ids_and_skips_redelivery(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    feedback = {
        "feedback_id": "feedback-1",
        "label": "true_positive",
        "features": {k: 0.9 for k in FEATURES_V1},
    }
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=1)

    assert updater.add_feedback(feedback) is True
    assert updater.flush().updated is True
    assert not model_path.with_suffix(".joblib.tmp").exists()

    reloaded = OnlineSGDUpdater(model_path=model_path, batch_size=1)
    assert reloaded.add_feedback(feedback) is False
    payload = joblib.load(model_path)
    assert payload["processed_feedback_ids"] == ["feedback-1"]
    assert payload["update_history"][-1]["feedback_ids"] == ["feedback-1"]


def test_staged_candidate_does_not_replace_active_until_promoted(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    original = model_path.read_bytes()
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=1)
    feedback = {
        "feedback_id": "feedback-1",
        "label": "true_positive",
        "features": {k: 0.9 for k in FEATURES_V1},
    }

    updater.add_feedback(feedback)
    result = updater.flush(stage_candidate=True)

    assert result.updated is True
    assert updater.candidate_path.is_file()
    assert model_path.read_bytes() == original

    updater.promote_candidate()
    assert not updater.candidate_path.exists()
    promoted = joblib.load(model_path)
    assert promoted["online_update_count"] == 1
    assert promoted["processed_feedback_ids"] == ["feedback-1"]


def test_staged_candidate_signal_is_published_only_after_promotion(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    publisher = InMemoryModelUpdatePublisher(events=[])
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=1, publisher=publisher)

    updater.add_feedback(_feedback("true_positive", 0.9))
    updater.flush(stage_candidate=True)
    assert publisher.events == []

    updater.promote_candidate()
    assert publisher.events[0]["model_path"] == str(model_path)
    assert len(publisher.events[0]["artifact_sha256"]) == 64
    assert publisher.events[0]["staged_candidate"] is False
