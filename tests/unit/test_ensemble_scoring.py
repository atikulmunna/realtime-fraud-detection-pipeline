from pathlib import Path

import joblib
import numpy as np
import pytest

from src.common.feature_contract import FEATURES_V1
from src.streaming.ensemble_scoring import (
    load_ensemble_models,
    route_score_to_topic,
    score_event_features,
)


class DummyIFModel:
    def decision_function(self, x):
        return np.array([0.0] * len(x))


class DummyScaler:
    def transform(self, x):
        return x


class DummyAEModel:
    def predict(self, x):
        return np.zeros_like(x)


class DummySGDModel:
    def predict_proba(self, x):
        return np.array([[0.2, 0.8] for _ in range(len(x))])


def _feature_row(value: float = 1.0) -> dict:
    return {k: value for k in FEATURES_V1}


def test_load_ensemble_models_reads_artifacts(tmp_path: Path):
    if_path = tmp_path / "if.joblib"
    ae_path = tmp_path / "ae.joblib"
    sgd_path = tmp_path / "sgd.joblib"

    joblib.dump(DummyIFModel(), if_path)
    joblib.dump(
        {
            "model": DummyAEModel(),
            "scaler": DummyScaler(),
            "threshold_p99": 2.0,
        },
        ae_path,
    )
    joblib.dump({"model": DummySGDModel()}, sgd_path)

    models = load_ensemble_models(
        if_model_path=if_path,
        ae_model_path=ae_path,
        sgd_model_path=sgd_path,
    )

    assert isinstance(models.if_model, DummyIFModel)
    assert isinstance(models.ae_model, DummyAEModel)
    assert isinstance(models.ae_scaler, DummyScaler)
    assert isinstance(models.sgd_model, DummySGDModel)
    assert models.ae_threshold_p99 == 2.0


def test_score_event_features_expected_weighted_score():
    class _Models:
        if_model = DummyIFModel()
        ae_model = DummyAEModel()
        ae_scaler = DummyScaler()
        ae_threshold_p99 = 2.0
        sgd_model = DummySGDModel()

    scores = score_event_features(_feature_row(1.0), _Models())
    assert pytest.approx(scores["if_score"], rel=1e-6) == 0.5
    assert pytest.approx(scores["ae_score"], rel=1e-6) == 0.5
    assert pytest.approx(scores["sgd_score"], rel=1e-6) == 0.8
    assert pytest.approx(scores["ensemble_score"], rel=1e-6) == 0.59


def test_score_event_features_normalizes_weights():
    class _Models:
        if_model = DummyIFModel()
        ae_model = DummyAEModel()
        ae_scaler = DummyScaler()
        ae_threshold_p99 = 2.0
        sgd_model = DummySGDModel()

    a = score_event_features(_feature_row(1.0), _Models(), weights=(0.4, 0.3, 0.3))
    b = score_event_features(_feature_row(1.0), _Models(), weights=(4, 3, 3))
    assert pytest.approx(a["ensemble_score"], rel=1e-9) == b["ensemble_score"]


def test_score_event_features_raises_on_missing_feature():
    class _Models:
        if_model = DummyIFModel()
        ae_model = DummyAEModel()
        ae_scaler = DummyScaler()
        ae_threshold_p99 = 2.0
        sgd_model = DummySGDModel()

    bad = _feature_row(1.0)
    bad.pop(FEATURES_V1[0])
    with pytest.raises(ValueError, match="Missing required feature keys"):
        score_event_features(bad, _Models())


def test_route_score_to_topic_threshold():
    assert route_score_to_topic(0.8, threshold=0.5) == "anomalies"
    assert route_score_to_topic(0.2, threshold=0.5) == "metrics"
