"""Create the deterministic local SGD seed required by the Compose updater."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from src.common.feature_contract import FEATURES_V1


def seed_local_sgd_model(path: str | Path, *, overwrite: bool = False) -> bool:
    output = Path(path)
    if output.exists() and not overwrite:
        return False
    model = SGDClassifier(loss="log_loss", random_state=1, max_iter=200, tol=1e-3)
    x = np.array([[0.1] * len(FEATURES_V1), [0.9] * len(FEATURES_V1)], dtype=float)
    model.fit(x, np.array([0, 1], dtype=int))
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "sgd_classifier",
            "model": model,
            "features_order": FEATURES_V1,
            "model_version": "local-seed-v1",
        },
        output,
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic local-only SGD seed model.")
    parser.add_argument("--output", default="models/sgd_classifier_v1.joblib")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    created = seed_local_sgd_model(args.output, overwrite=args.overwrite)
    print(f"created={str(created).lower()} path={args.output}")


if __name__ == "__main__":
    main()
