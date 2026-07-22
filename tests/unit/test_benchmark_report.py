from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
    assert report["model_source"] == "demo"


def test_save_benchmark_report_writes_json(tmp_path: Path):
    report = {"latency_ms": {"p95": 10.0}, "quality_at_budget": {"precision": 0.5, "recall": 0.4}}
    out = save_benchmark_report(report, output_path=tmp_path / "bench.json")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert '"p95": 10.0' in text


def test_run_benchmark_with_trained_model_artifacts(tmp_path: Path):
    if_path, ae_path, sgd_path = _write_dummy_ensemble_artifacts(tmp_path)
    evaluation_path = _write_evaluation_parquet(tmp_path)
    report = run_benchmark(
        BenchmarkConfig(
            n_events=100,
            use_trained_models=True,
            if_model_path=if_path,
            ae_model_path=ae_path,
            sgd_model_path=sgd_path,
            evaluation_parquet=evaluation_path,
        )
    )
    assert report["events_total"] == 100
    assert report["model_source"] == "trained_artifacts"
    assert report["model_paths"]["if_model_path"] == str(if_path)
    assert report["dataset"]["split"] == "chronological_test"
    assert len(report["dataset"]["dataset_hash"]) == 64
    assert "pr_auc" in report["quality_at_budget"]
    assert set(report["quality_at_threshold"]) >= {
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    }


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
                evaluation_parquet=_write_evaluation_parquet(tmp_path),
            )
        )


def test_trained_benchmark_requires_representative_evaluation_data(tmp_path: Path):
    if_path, ae_path, sgd_path = _write_dummy_ensemble_artifacts(tmp_path)
    with pytest.raises(ValueError, match="require evaluation_parquet"):
        run_benchmark(
            BenchmarkConfig(
                use_trained_models=True,
                if_model_path=if_path,
                ae_model_path=ae_path,
                sgd_model_path=sgd_path,
            )
        )
