from pathlib import Path

from src.demo.demo_flow import run_demo_flow


def test_run_demo_flow_smoke(tmp_path: Path):
    model_path = tmp_path / "sgd_demo.joblib"
    summary = run_demo_flow(model_path=model_path)

    assert summary["events_in"] == 3
    assert summary["anomalies"] >= 1
    assert summary["metrics"] >= 1
    assert summary["dlq"] >= 1
    assert summary["feedback_published"] == summary["anomalies"]
    assert summary["online_updated"] is True
    assert summary["online_update_count"] >= 1
    assert summary["signals_emitted"] == 1
    assert summary["api_feedback_requests_total"] == float(summary["feedback_published"])
    assert summary["online_updates_total"] == 1.0
