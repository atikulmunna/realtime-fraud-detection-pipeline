from pathlib import Path

import joblib
import numpy as np
import pytest

from src.evaluation.benchmark_report import BenchmarkConfig, run_benchmark, save_benchmark_report


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


def test_run_benchmark_returns_latency_and_quality_metrics():
    report = run_benchmark(
        BenchmarkConfig(
            n_events=200,
            fraud_ratio=0.15,
            alert_budget_ratio=0.1,
            score_threshold=0.65,
            seed=7,
        )
    )

    assert report["events_total"] == 200
    assert report["events_scored"] > 0
    assert report["routed_dlq"] == 0
    assert report["alerts_sent"] == int(round(report["events_scored"] * 0.1))

    latency = report["latency_ms"]
    assert latency["p50"] >= 0.0
    assert latency["p95"] >= latency["p50"]
    assert latency["max"] >= latency["p95"]

    quality = report["quality_at_budget"]
    assert 0.0 <= quality["precision"] <= 1.0
    assert 0.0 <= quality["recall"] <= 1.0
    assert report["model_source"] == "demo_models"


def test_save_benchmark_report_writes_json(tmp_path: Path):
    report = {"latency_ms": {"p95": 10.0}, "quality_at_budget": {"precision": 0.5, "recall": 0.4}}
    out = save_benchmark_report(report, output_path=tmp_path / "bench.json")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert '"p95": 10.0' in text


def test_run_benchmark_with_trained_model_artifacts(tmp_path: Path):
    if_path, ae_path, sgd_path = _write_dummy_ensemble_artifacts(tmp_path)
    report = run_benchmark(
        BenchmarkConfig(
            n_events=100,
            use_trained_models=True,
            if_model_path=if_path,
            ae_model_path=ae_path,
            sgd_model_path=sgd_path,
        )
    )
    assert report["events_total"] == 100
    assert report["model_source"] == "trained_artifacts"
    assert report["model_paths"]["if_model_path"] == str(if_path)


def test_run_benchmark_trained_models_missing_artifact_raises(tmp_path: Path):
    if_path, ae_path, sgd_path = _write_dummy_ensemble_artifacts(tmp_path)
    if_path.unlink()
    with pytest.raises(FileNotFoundError, match="Missing trained model artifact"):
        run_benchmark(
            BenchmarkConfig(
                n_events=20,
                use_trained_models=True,
                if_model_path=if_path,
                ae_model_path=ae_path,
                sgd_model_path=sgd_path,
            )
        )
