"""Opt-in tests against `scripts/smoke-compose.ps1` infrastructure."""

import os
import re
import time
from uuid import uuid4

import httpx
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_COMPOSE_INTEGRATION") != "1", reason="Compose integration is opt-in."),
]


def _metrics(url: str) -> str | None:
    try:
        return httpx.get(url, timeout=5.0).raise_for_status().text
    except httpx.HTTPError:
        return None


def _metric_value(metrics: str | None, name: str) -> float | None:
    if metrics is None:
        return None
    match = re.search(rf"^{re.escape(name)}\s+([0-9.eE+-]+)$", metrics, re.MULTILINE)
    return float(match.group(1)) if match else None


def _model_version_count() -> int | None:
    try:
        response = httpx.get(
            "http://127.0.0.1:5000/api/2.0/mlflow/model-versions/search",
            params={"filter": "name='fraud-online-sgd'"},
            timeout=5.0,
        )
        if response.status_code == 404:
            return 0
        response.raise_for_status()
        return len(response.json().get("model_versions", []))
    except httpx.HTTPError:
        return None


def test_feedback_outbox_kafka_updater_mlflow_chain():
    run_id = uuid4().hex
    initial_model_versions = _model_version_count()
    assert initial_model_versions is not None
    headers = {
        "X-API-Key": os.getenv("FEEDBACK_API_KEY", "development-only-change-me"),
        "Idempotency-Key": f"integration-request-{run_id}",
    }
    payload = {
        "feedback_id": f"integration-feedback-{run_id}",
        "anomaly_id": f"integration-anomaly-{run_id}",
        "label": "true_positive",
        "analyst_id": "integration-test",
        "features": {
            "amount": 900.0,
            "amount_ratio": 0.9,
            "balance_diff_orig": 0.0,
            "is_transfer": 1.0,
            "is_cashout": 0.0,
            "hour_of_day": 8.0,
            "txn_velocity_1h": 1.0,
        },
    }

    first = httpx.post("http://127.0.0.1:8000/feedback", headers=headers, json=payload, timeout=5.0)
    duplicate = httpx.post("http://127.0.0.1:8000/feedback", headers=headers, json=payload, timeout=5.0)
    assert first.raise_for_status().json()["duplicate"] is False
    assert duplicate.raise_for_status().json()["duplicate"] is True

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        relay = _metrics("http://127.0.0.1:8003/metrics")
        updater = _metrics("http://127.0.0.1:8002/metrics")
        registered = httpx.get(
            "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/get",
            params={"name": "fraud-online-sgd"},
            timeout=5.0,
        )
        if (
            _metric_value(relay, "outbox_backlog") == 0.0
            and (_metric_value(updater, "online_updates_total") or 0.0) >= 1.0
            and registered.status_code == 200
            and (_model_version_count() or 0) > initial_model_versions
        ):
            break
        time.sleep(1.0)
    else:
        raise AssertionError("Feedback did not complete the durable update chain within 60 seconds.")

    assert registered.raise_for_status().json()["registered_model"]["name"] == "fraud-online-sgd"
