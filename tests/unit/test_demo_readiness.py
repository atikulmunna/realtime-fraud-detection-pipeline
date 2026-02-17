from pathlib import Path

import joblib
import numpy as np

from src.demo.readiness_check import run_demo_readiness_check, save_readiness_report


class _DummyIFModel:
    def decision_function(self, x):
        return np.array([0.0] * len(x))


class _DummyScaler:
    def transform(self, x):
        return x


class _DummyAEModel:
    def predict(self, x):
        return np.zeros_like(x)


class _DummySGDModel:
    def predict_proba(self, x):
        return np.array([[0.2, 0.8] for _ in range(len(x))])


def _write_dummy_ensemble_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    if_path = tmp_path / "if.joblib"
    ae_path = tmp_path / "ae.joblib"
    sgd_path = tmp_path / "sgd.joblib"
    joblib.dump(_DummyIFModel(), if_path)
    joblib.dump({"model": _DummyAEModel(), "scaler": _DummyScaler(), "threshold_p99": 1.0}, ae_path)
    joblib.dump({"model": _DummySGDModel()}, sgd_path)
    return if_path, ae_path, sgd_path


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


def test_demo_readiness_check_with_trained_models(tmp_path: Path):
    model_path = tmp_path / "sgd_demo.joblib"
    if_path, ae_path, sgd_path = _write_dummy_ensemble_artifacts(tmp_path)
    report = run_demo_readiness_check(
        model_path=model_path,
        benchmark_events=120,
        benchmark_alert_budget_ratio=0.1,
        latency_slo_ms=500.0,
        use_trained_models=True,
        if_model_path=if_path,
        ae_model_path=ae_path,
        sgd_model_path=sgd_path,
    )
    assert report["overall_ok"] is True
    assert report["benchmark"]["model_source"] == "trained_artifacts"
    assert report["benchmark"]["model_paths"]["if_model_path"] == str(if_path)


def test_save_readiness_report_writes_file(tmp_path: Path):
    report = {"overall_ok": True, "checks": {"x": True}}
    out = save_readiness_report(report, output_path=tmp_path / "readiness.json")
    assert out.exists()
    assert '"overall_ok": true' in out.read_text(encoding="utf-8")
