from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd

from src.models.registry import (
    FraudEnsemblePyFunc,
    MlflowOnlineCandidateRegistry,
    load_champion_bundle,
    promote_candidate,
    register_candidate_bundle,
    rollback_champion,
)


class _RegistryClient:
    def __init__(self, *, champion="3"):
        self.aliases = {"candidate": "4", "champion": champion}
        self.calls = []

    def get_model_version_by_alias(self, name, alias):
        return SimpleNamespace(version=self.aliases[alias])

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = str(version)
        self.calls.append(("alias", name, alias, str(version)))

    def set_model_version_tag(self, name, version, key, value):
        self.calls.append(("tag", name, str(version), key, value))


def test_promote_and_rollback_use_mlflow_aliases():
    client = _RegistryClient()

    result = promote_candidate(client=client)
    rollback_champion(result["previous_champion_version"], client=client)

    assert result == {"champion_version": "4", "previous_champion_version": "3"}
    assert client.aliases["champion"] == "3"


def test_register_candidate_logs_pyfunc_and_sets_candidate_alias(monkeypatch, tmp_path: Path):
    bundle_path = tmp_path / "bundle.joblib"
    bundle_path.write_bytes(b"bundle")
    bundle_path.with_suffix(".joblib.manifest.json").write_text("{}", encoding="utf-8")
    metadata = SimpleNamespace(
        model_version="candidate-7",
        dataset_hash="a" * 64,
        git_revision="deadbeef",
        feature_contract_version="1",
    )
    monkeypatch.setattr("src.models.registry.load_bundle", lambda path: SimpleNamespace(metadata=metadata))
    monkeypatch.setattr("src.models.registry.mlflow.set_tracking_uri", lambda value: None)
    monkeypatch.setattr("src.models.registry.mlflow.set_experiment", lambda value: None)
    logged = {}
    monkeypatch.setattr("src.models.registry.mlflow.pyfunc.log_model", lambda **kwargs: logged.update(kwargs))

    @contextmanager
    def fake_run(**kwargs):
        yield SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

    monkeypatch.setattr("src.models.registry.mlflow.start_run", fake_run)
    monkeypatch.setattr(
        "src.models.registry.mlflow.register_model",
        lambda **kwargs: SimpleNamespace(version="7"),
    )
    client = _RegistryClient()
    monkeypatch.setattr("src.models.registry.MlflowClient", lambda **kwargs: client)

    result = register_candidate_bundle(bundle_path=bundle_path, tracking_uri="http://mlflow:5000")

    assert result["version"] == "7"
    assert logged["artifacts"]["bundle"] == str(bundle_path.resolve())
    assert client.aliases["candidate"] == "7"


def test_pyfunc_requires_feature_contract(monkeypatch):
    model = FraudEnsemblePyFunc()
    model.bundle = SimpleNamespace(models=object())
    try:
        model.predict(None, pd.DataFrame({"amount": [1.0]}))
    except ValueError as exc:
        assert "missing required features" in str(exc)
    else:
        raise AssertionError("Expected missing features to fail")


def test_champion_loader_uses_only_verified_cache_on_registry_failure(monkeypatch, tmp_path: Path):
    cache = tmp_path / "champion.joblib"
    cache.write_bytes(b"cached")
    expected = SimpleNamespace(metadata=SimpleNamespace(model_version="cached-v1"))
    monkeypatch.setattr(
        "src.models.registry.mlflow.artifacts.download_artifacts",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("registry down")),
    )
    monkeypatch.setattr("src.models.registry.load_bundle", lambda path: expected)

    loaded = load_champion_bundle(cache_path=cache, allow_verified_cache=True)

    assert loaded is expected


def test_online_candidate_registry_registers_and_promotes_exact_version(monkeypatch, tmp_path: Path):
    candidate = tmp_path / "candidate.joblib"
    joblib.dump({"model": object()}, candidate)
    monkeypatch.setattr("src.models.registry.mlflow.set_tracking_uri", lambda value: None)
    monkeypatch.setattr("src.models.registry.mlflow.set_experiment", lambda value: None)
    logged = {}
    monkeypatch.setattr("src.models.registry.mlflow_sklearn.log_model", lambda **kwargs: logged.update(kwargs))

    @contextmanager
    def fake_run(**kwargs):
        yield SimpleNamespace(info=SimpleNamespace(run_id="online-run-1"))

    monkeypatch.setattr("src.models.registry.mlflow.start_run", fake_run)
    monkeypatch.setattr(
        "src.models.registry.mlflow.register_model",
        lambda **kwargs: SimpleNamespace(version="8"),
    )
    client = _RegistryClient()
    monkeypatch.setattr("src.models.registry.MlflowClient", lambda **kwargs: client)
    registry = MlflowOnlineCandidateRegistry(tracking_uri="http://mlflow:5000")

    version = registry.register_candidate(candidate, {"online_update_count": 2, "batch_size": 4})
    registry.promote_candidate(version)

    assert version == "8"
    assert logged["artifact_path"] == "model"
    assert client.aliases["candidate"] == "8"
    assert client.aliases["champion"] == "8"
