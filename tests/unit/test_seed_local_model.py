from pathlib import Path

import joblib

from src.common.feature_contract import FEATURES_V1
from src.demo.seed_local_model import seed_local_sgd_model


def test_seed_local_model_is_deterministic_and_does_not_overwrite(tmp_path: Path):
    path = tmp_path / "sgd.joblib"

    assert seed_local_sgd_model(path) is True
    first = path.read_bytes()
    assert seed_local_sgd_model(path) is False
    assert path.read_bytes() == first

    payload = joblib.load(path)
    assert payload["features_order"] == FEATURES_V1
    assert payload["model_version"] == "local-seed-v1"
