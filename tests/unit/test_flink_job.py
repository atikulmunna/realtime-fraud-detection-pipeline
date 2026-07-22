import hashlib
import json
from pathlib import Path

import joblib

from src.common.feature_contract import FEATURES_V1
from src.streaming.flink_job import (
    envelope_user_key,
    event_timestamp_ms,
    event_user_key,
    utc_hour_key,
    validate_model_update_signal,
)


def test_event_time_helpers_are_utc_and_invalid_keys_are_safe():
    timestamp_ms = event_timestamp_ms({"timestamp": "2026-07-20T09:42:00+06:00"})

    assert utc_hour_key(timestamp_ms) == "2026-07-20T03:00:00+00:00"
    assert event_user_key(json.dumps({"user_id": "customer-7"})) == "customer-7"
    assert event_user_key("{bad json}") == "__invalid__"
    assert envelope_user_key(json.dumps({"payload": {"user_id": "customer-7"}})) == "customer-7"


def test_model_update_signal_accepts_only_promoted_contract_compatible_artifact(tmp_path: Path):
    artifact = tmp_path / "sgd.joblib"
    joblib.dump({"model": object(), "features_order": FEATURES_V1}, artifact)

    resolved = validate_model_update_signal(
        {
            "model_type": "sgd_classifier",
            "model_path": f"/app/models/{artifact.name}",
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        },
        tmp_path,
    )

    assert resolved == artifact.resolve()


def test_model_update_signal_rejects_staged_candidate(tmp_path: Path):
    try:
        validate_model_update_signal(
            {"model_type": "sgd_classifier", "model_path": "/app/models/sgd.joblib.candidate"},
            tmp_path,
        )
    except ValueError as exc:
        assert "promoted artifact" in str(exc)
    else:
        raise AssertionError("Expected staged candidate signal to fail")
