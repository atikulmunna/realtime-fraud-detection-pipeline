import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from src.common.feature_contract import FEATURES_V1
from src.online.updater_service import run_updater_service_once


def _seed_model(path: Path) -> None:
    model = SGDClassifier(loss="log_loss", random_state=1, max_iter=200, tol=1e-3)
    x = np.array([[0.1] * len(FEATURES_V1), [0.9] * len(FEATURES_V1)], dtype=float)
    y = np.array([0, 1], dtype=int)
    model.fit(x, y)
    joblib.dump({"model_type": "sgd_classifier", "model": model, "features_order": FEATURES_V1}, path)


def _feedback(label: str, value: float) -> dict:
    return {
        "anomaly_id": f"a-{value}",
        "label": label,
        "analyst_id": "u1",
        "features": {k: value for k in FEATURES_V1},
    }


def test_run_updater_service_once_reads_jsonl_and_updates(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    feedback_file = tmp_path / "feedback.jsonl"
    _seed_model(model_path)

    lines = [json.dumps(_feedback("true_positive", 0.9)), json.dumps(_feedback("false_positive", 0.2))]
    feedback_file.write_text("\n".join(lines), encoding="utf-8")

    summary = run_updater_service_once(
        model_path=model_path,
        batch_size=2,
        feedback_file=feedback_file,
        force_flush=True,
    )
    assert summary["messages_in"] == 2
    assert summary["updated"] is True
    assert summary["batch_size"] == 2


def test_run_updater_service_once_no_force_flush_respects_batch_size(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    feedback_file = tmp_path / "feedback.jsonl"
    _seed_model(model_path)
    feedback_file.write_text(json.dumps(_feedback("true_positive", 0.9)), encoding="utf-8")

    summary = run_updater_service_once(
        model_path=model_path,
        batch_size=3,
        feedback_file=feedback_file,
        force_flush=False,
    )
    assert summary["messages_in"] == 1
    assert summary["updated"] is False
