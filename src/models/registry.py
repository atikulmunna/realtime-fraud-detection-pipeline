"""MLflow registration, alias promotion, rollback, and verified champion caching."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow import sklearn as mlflow_sklearn
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PythonModel

from src.common.feature_contract import FEATURES_V1
from src.models.ensemble_bundle import EnsembleBundle, load_bundle
from src.streaming.ensemble_scoring import score_event_features


class MlflowOnlineCandidateRegistry:
    """Register staged online SGD candidates and promote only approved versions."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        registered_model_name: str = "fraud-online-sgd",
        experiment_name: str = "realtime-fraud-online-updates",
    ) -> None:
        self.tracking_uri = tracking_uri
        self.registered_model_name = registered_model_name
        self.experiment_name = experiment_name

    def register_candidate(self, candidate_path: str | Path, metadata: dict[str, Any]) -> str:
        payload = joblib.load(candidate_path)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=f"online-update-{metadata['online_update_count']}") as run:
            mlflow_sklearn.log_model(
                sk_model=payload["model"],
                artifact_path="model",
                metadata=metadata,
            )
            version = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=self.registered_model_name,
            )

        client = MlflowClient(tracking_uri=self.tracking_uri)
        client.set_registered_model_alias(self.registered_model_name, "candidate", version.version)
        client.set_model_version_tag(
            self.registered_model_name,
            version.version,
            "online_update_count",
            str(metadata["online_update_count"]),
        )
        return str(version.version)

    def promote_candidate(self, version: str) -> None:
        client = MlflowClient(tracking_uri=self.tracking_uri)
        candidate = client.get_model_version_by_alias(self.registered_model_name, "candidate")
        if str(candidate.version) != str(version):
            raise RuntimeError(
                f"Candidate alias changed during evaluation: expected {version}, found {candidate.version}."
            )
        client.set_registered_model_alias(self.registered_model_name, "champion", version)
        client.set_model_version_tag(self.registered_model_name, version, "promotion_status", "champion")


class FraudEnsemblePyFunc(PythonModel):
    def load_context(self, context: Any) -> None:
        self.bundle = load_bundle(context.artifacts["bundle"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None) -> pd.DataFrame:
        missing = [name for name in FEATURES_V1 if name not in model_input.columns]
        if missing:
            raise ValueError(f"Model input is missing required features: {missing}")
        rows = [score_event_features(row, self.bundle.models) for row in model_input[FEATURES_V1].to_dict("records")]
        return pd.DataFrame(rows)


def register_candidate_bundle(
    *,
    bundle_path: str | Path,
    tracking_uri: str,
    registered_model_name: str = "fraud-ensemble",
    experiment_name: str = "realtime-fraud-detection-pipeline",
) -> dict[str, str]:
    bundle = load_bundle(bundle_path)
    path = Path(bundle_path).resolve()
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"candidate-{bundle.metadata.model_version}") as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=FraudEnsemblePyFunc(),
            artifacts={"bundle": str(path), "manifest": str(manifest)},
            pip_requirements=["mlflow>=2.16,<3.0", "numpy>=1.26,<3.0", "pandas>=2.2,<3.0", "scikit-learn>=1.5,<2.0"],
            metadata={
                "model_version": bundle.metadata.model_version,
                "dataset_hash": bundle.metadata.dataset_hash,
                "git_revision": bundle.metadata.git_revision,
                "feature_contract_version": bundle.metadata.feature_contract_version,
            },
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(registered_model_name, "candidate", version.version)
    client.set_model_version_tag(registered_model_name, version.version, "dataset_hash", bundle.metadata.dataset_hash)
    client.set_model_version_tag(
        registered_model_name,
        version.version,
        "feature_contract_version",
        bundle.metadata.feature_contract_version,
    )
    return {"run_id": run.info.run_id, "model_uri": model_uri, "version": str(version.version)}


def promote_candidate(
    *,
    registered_model_name: str = "fraud-ensemble",
    client: Any | None = None,
) -> dict[str, str | None]:
    registry = client or MlflowClient()
    candidate = registry.get_model_version_by_alias(registered_model_name, "candidate")
    previous: str | None = None
    try:
        previous = str(registry.get_model_version_by_alias(registered_model_name, "champion").version)
    except MlflowException:
        pass
    registry.set_registered_model_alias(registered_model_name, "champion", candidate.version)
    registry.set_model_version_tag(registered_model_name, candidate.version, "promotion_status", "champion")
    return {"champion_version": str(candidate.version), "previous_champion_version": previous}


def rollback_champion(
    previous_version: str,
    *,
    registered_model_name: str = "fraud-ensemble",
    client: Any | None = None,
) -> None:
    registry = client or MlflowClient()
    registry.set_registered_model_alias(registered_model_name, "champion", previous_version)
    registry.set_model_version_tag(registered_model_name, previous_version, "promotion_status", "rollback_champion")


def load_champion_bundle(
    *,
    cache_path: str | Path,
    registered_model_name: str = "fraud-ensemble",
    tracking_uri: str | None = None,
    allow_verified_cache: bool = True,
) -> EnsembleBundle:
    cache = Path(cache_path)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    try:
        with TemporaryDirectory(prefix="fraud-champion-") as temp_dir:
            downloaded = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{registered_model_name}@champion",
                    dst_path=temp_dir,
                )
            )
            candidates = list(downloaded.rglob("*.joblib"))
            if len(candidates) != 1:
                raise ValueError("Champion model must contain exactly one ensemble bundle.")
            source = candidates[0]
            source_manifest = source.with_suffix(source.suffix + ".manifest.json")
            load_bundle(source)
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache.with_suffix(cache.suffix + ".tmp")
            temporary_manifest = temporary.with_suffix(temporary.suffix + ".manifest.json")
            shutil.copy2(source, temporary)
            shutil.copy2(source_manifest, temporary_manifest)
            os.replace(temporary, cache)
            os.replace(temporary_manifest, cache.with_suffix(cache.suffix + ".manifest.json"))
            return load_bundle(cache)
    except Exception:
        if allow_verified_cache and cache.exists():
            return load_bundle(cache)
        raise
