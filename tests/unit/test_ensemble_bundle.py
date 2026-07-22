from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.build_ensemble import create_calibrated_bundle
from src.models.ensemble_bundle import (
    ScoreCalibrator,
    build_bundle,
    check_candidate_quality,
    fit_score_calibrator,
    load_bundle,
    save_bundle,
    select_ensemble_configuration,
)
from src.streaming.ensemble_scoring import EnsembleModels, score_event_features


class _IfModel:
    def decision_function(self, x):
        return np.zeros(len(x))


class _Scaler:
    def transform(self, x):
        return x


class _AeModel:
    def predict(self, x):
        return np.zeros_like(x)


class _SgdModel:
    def predict_proba(self, x):
        p1 = np.clip(x[:, 0] / 1000.0, 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])


def _models() -> EnsembleModels:
    return EnsembleModels(
        if_model=_IfModel(),
        ae_model=_AeModel(),
        ae_scaler=_Scaler(),
        ae_threshold_p99=1.0,
        sgd_model=_SgdModel(),
    )


def test_fit_score_calibrator_returns_bounded_probabilities():
    calibrator = fit_score_calibrator(np.array([0.0, 0.1, 0.9, 1.0]), np.array([0, 0, 1, 1]))
    calibrated = calibrator.transform(np.array([-10.0, 0.5, 10.0]))
    assert np.all(calibrated > 0.0)
    assert np.all(calibrated < 1.0)
    assert calibrated[0] < calibrated[1] < calibrated[2]


def test_select_configuration_is_deterministic_and_reports_budget_quality():
    labels = np.array([0, 0, 0, 0, 1, 1])
    component_scores = {
        "if": np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9]),
        "ae": np.array([0.2, 0.1, 0.4, 0.3, 0.9, 0.8]),
        "sgd": np.array([0.05, 0.1, 0.2, 0.3, 0.85, 0.95]),
    }

    first = select_ensemble_configuration(component_scores, labels, alert_budget_ratio=1 / 3)
    second = select_ensemble_configuration(component_scores, labels, alert_budget_ratio=1 / 3)

    assert first.weights == second.weights
    assert first.threshold == pytest.approx(second.threshold)
    assert first.metrics["precision_at_budget"] == 1.0
    assert first.metrics["recall_at_budget"] == 1.0


def test_candidate_quality_requires_absolute_gates_and_limits_regression():
    champion = {"pr_auc": 0.5, "precision_at_budget": 0.4, "recall_at_budget": 0.8}
    passing = {"pr_auc": 0.49, "precision_at_budget": 0.39, "recall_at_budget": 0.79}
    failing = {"pr_auc": 0.3, "precision_at_budget": 0.2, "recall_at_budget": 0.5}

    assert check_candidate_quality(passing, champion)[0] is True
    accepted, reasons = check_candidate_quality(failing, champion)
    assert accepted is False
    assert any("recall_at_budget" in reason for reason in reasons)


def test_bundle_round_trip_is_checksum_verified_and_immutable(tmp_path: Path):
    labels = np.array([0, 0, 1, 1])
    scores = {
        "if": np.array([0.1, 0.2, 0.8, 0.9]),
        "ae": np.array([0.2, 0.1, 0.9, 0.8]),
        "sgd": np.array([0.05, 0.1, 0.85, 0.95]),
    }
    selection = select_ensemble_configuration(scores, labels, alert_budget_ratio=0.5)
    bundle = build_bundle(
        _models(),
        selection,
        model_version="candidate-1",
        dataset_hash="a" * 64,
        git_revision="deadbeef",
    )
    path = tmp_path / "ensemble.joblib"

    manifest = save_bundle(bundle, path)
    loaded = load_bundle(path)

    assert loaded.metadata.model_version == "candidate-1"
    assert manifest["sha256"]
    with pytest.raises(FileExistsError, match="immutable bundle"):
        save_bundle(bundle, path)

    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(path)


def test_scoring_applies_bundle_calibrators_and_weights():
    models = _models()
    models = EnsembleModels(
        if_model=models.if_model,
        ae_model=models.ae_model,
        ae_scaler=models.ae_scaler,
        ae_threshold_p99=models.ae_threshold_p99,
        sgd_model=models.sgd_model,
        calibrators={name: ScoreCalibrator(0.0, 0.0) for name in ("if", "ae", "sgd")},
        weights=(0.2, 0.3, 0.5),
        threshold=0.7,
    )
    features = {
        "amount": 0.8,
        "amount_ratio": 0.1,
        "balance_diff_orig": 0.0,
        "is_transfer": 1.0,
        "is_cashout": 0.0,
        "hour_of_day": 1.0,
        "txn_velocity_1h": 1.0,
    }

    result = score_event_features(features, models)

    assert result["ensemble_score"] == pytest.approx(0.5)
    assert result["if_score"] == pytest.approx(0.5)
    assert "raw_if_score" in result


def test_create_calibrated_bundle_uses_validation_split_and_writes_manifest(tmp_path: Path):
    rows = []
    for step in range(1, 31):
        for offset in range(30):
            fraud = int(offset % 10 == 0)
            amount = 900.0 if fraud else 20.0
            rows.append(
                {
                    "step": step,
                    "amount": amount,
                    "amount_ratio": amount / 1001.0,
                    "balance_diff_orig": 0.0,
                    "is_transfer": fraud,
                    "is_cashout": 0,
                    "hour_of_day": step % 24,
                    "txn_velocity_1h": 1,
                    "isFraud": fraud,
                }
            )
    parquet = tmp_path / "features.parquet"
    pd.DataFrame(rows).to_parquet(parquet, index=False)
    if_path = tmp_path / "if.joblib"
    ae_path = tmp_path / "ae.joblib"
    sgd_path = tmp_path / "sgd.joblib"
    joblib.dump(_IfModel(), if_path)
    joblib.dump({"model": _AeModel(), "scaler": _Scaler(), "threshold_p99": 1.0}, ae_path)
    joblib.dump({"model": _SgdModel()}, sgd_path)
    output = tmp_path / "bundle.joblib"

    result = create_calibrated_bundle(
        input_parquet=parquet,
        if_model_path=if_path,
        ae_model_path=ae_path,
        sgd_model_path=sgd_path,
        output_path=output,
        model_version="candidate-2",
        alert_budget_ratio=0.1,
        git_revision="cafebabe",
    )

    assert output.exists()
    assert output.with_suffix(".joblib.manifest.json").exists()
    assert result["model_version"] == "candidate-2"
    assert result["metrics"]["recall_at_budget"] >= 0.60
