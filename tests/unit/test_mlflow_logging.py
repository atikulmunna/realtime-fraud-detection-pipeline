import sys
from pathlib import Path

from src.models.mlflow_logging import log_training_run


class _DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyMLflow:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    def set_tracking_uri(self, uri):
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name):
        self.calls.append(("set_experiment", name))

    def start_run(self, run_name=None):
        self.calls.append(("start_run", run_name))
        return _DummyRun()

    def log_params(self, params):
        self.calls.append(("log_params", params))

    def log_metrics(self, metrics):
        self.calls.append(("log_metrics", metrics))

    def set_tags(self, tags):
        self.calls.append(("set_tags", tags))

    def log_artifact(self, path):
        self.calls.append(("log_artifact", path))


def test_log_training_run_noop_when_disabled(tmp_path: Path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("x", encoding="utf-8")
    ok = log_training_run(
        use_mlflow=False,
        model_name="x",
        run_name="r",
        params={},
        metrics={},
        artifacts=[artifact],
    )
    assert ok is False


def test_log_training_run_uses_mlflow_module(tmp_path: Path, monkeypatch):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("x", encoding="utf-8")
    dummy = _DummyMLflow()
    monkeypatch.setitem(sys.modules, "mlflow", dummy)

    ok = log_training_run(
        use_mlflow=True,
        model_name="SGDClassifier",
        run_name="sgd_baseline_v1",
        params={"max_iter": 1000, "features_order": ["a", "b"]},
        metrics={"val_pr_auc": 0.1, "rows_total": 100, "ignored": "x"},
        artifacts=[artifact],
        tracking_uri="http://localhost:5000",
        experiment_name="exp1",
    )
    assert ok is True
    names = [c[0] for c in dummy.calls]
    assert "set_tracking_uri" in names
    assert "set_experiment" in names
    assert "start_run" in names
    assert "log_params" in names
    assert "log_metrics" in names
    assert "set_tags" in names
    assert "log_artifact" in names
