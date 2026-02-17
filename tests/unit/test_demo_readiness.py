from pathlib import Path

from src.demo.readiness_check import run_demo_readiness_check, save_readiness_report


def test_demo_readiness_check_returns_pass_report(tmp_path: Path):
    model_path = tmp_path / "sgd_demo.joblib"
    report = run_demo_readiness_check(
        model_path=model_path,
        benchmark_events=200,
        benchmark_alert_budget_ratio=0.1,
        latency_slo_ms=500.0,
    )

    assert report["overall_ok"] is True
    assert report["checks"]["demo_has_anomalies"] is True
    assert report["checks"]["demo_online_updated"] is True
    assert report["checks"]["benchmark_latency_slo_met"] is True
    assert "precision" in report["benchmark"]["quality_at_budget"]
    assert "recall" in report["benchmark"]["quality_at_budget"]


def test_save_readiness_report_writes_file(tmp_path: Path):
    report = {"overall_ok": True, "checks": {"x": True}}
    out = save_readiness_report(report, output_path=tmp_path / "readiness.json")
    assert out.exists()
    assert '"overall_ok": true' in out.read_text(encoding="utf-8")

