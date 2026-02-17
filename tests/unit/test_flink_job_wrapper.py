from pathlib import Path

from src.streaming.flink_job_wrapper import (
    FlinkJobConfig,
    load_config_from_env,
    operator_wiring_contract,
    run_local_wrapper_batch,
    validate_model_paths,
)


class _IfModel:
    def decision_function(self, x):
        import numpy as np

        return np.array([0.0] * len(x))


class _Scaler:
    def transform(self, x):
        return x


class _AeModel:
    def predict(self, x):
        import numpy as np

        return np.zeros_like(x)


class _SgdModel:
    def predict_proba(self, x):
        import numpy as np

        return np.array([[0.2, 0.8] for _ in range(len(x))])


class _Models:
    if_model = _IfModel()
    ae_model = _AeModel()
    ae_scaler = _Scaler()
    ae_threshold_p99 = 1.0
    sgd_model = _SgdModel()


def _valid_event(event_id: str = "evt-1") -> dict:
    return {
        "event_id": event_id,
        "timestamp": "2026-02-16T10:00:00Z",
        "user_id": "C1",
        "type": "TRANSFER",
        "amount": 100.0,
        "old_balance_orig": 500.0,
        "new_balance_orig": 400.0,
    }


def test_load_config_from_env_overrides_defaults():
    cfg = load_config_from_env(
        {
            "RAW_EVENTS_TOPIC": "raw",
            "ANOMALIES_TOPIC": "anom",
            "METRICS_TOPIC": "met",
            "DLQ_TOPIC": "dlq",
            "ANOMALY_THRESHOLD": "0.7",
        }
    )
    assert cfg.input_topic == "raw"
    assert cfg.anomaly_topic == "anom"
    assert cfg.metrics_topic == "met"
    assert cfg.dlq_topic == "dlq"
    assert cfg.threshold == 0.7


def test_operator_wiring_contract_order():
    assert operator_wiring_contract() == [
        "EventParser",
        "FeatureExtractor",
        "EnsembleScoring",
        "ThresholdRouter",
    ]


def test_validate_model_paths(tmp_path: Path):
    if_path = tmp_path / "if.joblib"
    ae_path = tmp_path / "ae.joblib"
    sgd_path = tmp_path / "sgd.joblib"
    if_path.write_text("x", encoding="utf-8")
    ae_path.write_text("x", encoding="utf-8")
    cfg = FlinkJobConfig(
        if_model_path=str(if_path),
        ae_model_path=str(ae_path),
        sgd_model_path=str(sgd_path),
    )
    res = validate_model_paths(cfg)
    assert res["if_model_exists"] is True
    assert res["ae_model_exists"] is True
    assert res["sgd_model_exists"] is False


def test_run_local_wrapper_batch_routes_topics():
    cfg = FlinkJobConfig(anomaly_topic="anomalies", metrics_topic="metrics", dlq_topic="dead-letter")
    out = run_local_wrapper_batch(
        [_valid_event("evt-1"), "{bad json}"],
        models=_Models(),
        config=cfg,
    )
    assert len(out["anomalies"]) == 1
    assert len(out["metrics"]) == 0
    assert len(out["dead-letter"]) == 1
