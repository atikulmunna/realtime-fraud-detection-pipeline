from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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
        p1 = np.clip(x[:, 0] / 1000.0, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def partial_fit(self, x, y, classes=None):
        return self


class _NoAlertIFModel:
    def decision_function(self, x):
        return np.array([10.0] * len(x))


class _NoAlertAEModel:
    def predict(self, x):
        return x


class _NoAlertSGDModel:
    def predict_proba(self, x):
        return np.array([[1.0, 0.0] for _ in range(len(x))])

    def partial_fit(self, x, y, classes=None):
        return self


def _write_dummy_ensemble_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    if_path = tmp_path / "if.joblib"
    ae_path = tmp_path / "ae.joblib"
    sgd_path = tmp_path / "sgd.joblib"
    joblib.dump(_DummyIFModel(), if_path)
    joblib.dump({"model": _DummyAEModel(), "scaler": _DummyScaler(), "threshold_p99": 1.0}, ae_path)
    joblib.dump({"model": _DummySGDModel()}, sgd_path)
    return if_path, ae_path, sgd_path


def _write_no_alert_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    if_path = tmp_path / "no-alert-if.joblib"
    ae_path = tmp_path / "no-alert-ae.joblib"
    sgd_path = tmp_path / "no-alert-sgd.joblib"
    joblib.dump(_NoAlertIFModel(), if_path)
    joblib.dump({"model": _NoAlertAEModel(), "scaler": _DummyScaler(), "threshold_p99": 1.0}, ae_path)
    joblib.dump({"model": _NoAlertSGDModel()}, sgd_path)
    return if_path, ae_path, sgd_path


def _write_evaluation_parquet(tmp_path: Path) -> Path:
    rows = []
    base = datetime(2024, 1, 1)
    for step in range(1, 31):
        for offset in range(30):
            is_fraud = int(offset % 10 == 0)
            amount = 900.0 if is_fraud else 20.0
            rows.append(
                {
                    "step": step,
                    "timestamp": base + timedelta(hours=step),
                    "nameOrig": f"C-{step}-{offset}",
                    "type": "TRANSFER" if is_fraud else "PAYMENT",
                    "amount": amount,
                    "oldbalanceOrg": 1000.0,
                    "newbalanceOrig": 1000.0 - amount,
                    "txn_velocity_1h": 1,
                    "isFraud": is_fraud,
                }
            )
    path = tmp_path / "evaluation.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_demo_readiness_check_returns_pass_report(tmp_path: Path):
    model_path = tmp_path / "sgd_demo.joblib"
    report = run_demo_readiness_check(
        model_path=model_path,
        benchmark_events=200,
        benchmark_alert_budget_ratio=0.1,
        latency_slo_ms=500.0,
        use_trained_models=False,
        allow_demo_mode=True,
    )

    assert report["overall_ok"] is True
    assert report["checks"]["demo_has_anomalies"] is True
    assert report["checks"]["demo_online_updated"] is True
    assert report["checks"]["benchmark_latency_slo_met"] is True
    assert report["checks"]["model_sources_match"] is True
    assert report["checks"]["benchmark_routes_anomalies"] is True
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
        evaluation_parquet=_write_evaluation_parquet(tmp_path),
    )
    assert report["overall_ok"] is True
    assert report["benchmark"]["model_source"] == "trained_artifacts"
    assert report["demo"]["model_source"] == "trained_artifacts"
    assert report["benchmark"]["model_paths"]["if_model_path"] == str(if_path)


def test_save_readiness_report_writes_file(tmp_path: Path):
    report = {"overall_ok": True, "checks": {"x": True}}
    out = save_readiness_report(report, output_path=tmp_path / "readiness.json")
    assert out.exists()
    assert '"overall_ok": true' in out.read_text(encoding="utf-8")


def test_trained_readiness_fails_when_model_routes_no_anomalies(tmp_path: Path):
    if_path, ae_path, sgd_path = _write_no_alert_artifacts(tmp_path)

    report = run_demo_readiness_check(
        benchmark_events=50,
        use_trained_models=True,
        if_model_path=if_path,
        ae_model_path=ae_path,
        sgd_model_path=sgd_path,
        evaluation_parquet=_write_evaluation_parquet(tmp_path),
    )

    assert report["overall_ok"] is False
    assert report["checks"]["demo_has_anomalies"] is False
    assert report["checks"]["benchmark_routes_anomalies"] is False
