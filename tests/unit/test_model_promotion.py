from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import SGDClassifier

from src.common.feature_contract import FEATURES_V1
from src.online.model_promotion import (
    PromotionThresholds,
    check_promotion_thresholds,
    evaluate_and_maybe_rollback,
    evaluate_model_on_holdout,
)
from src.online.online_sgd_updater import OnlineSGDUpdater


def _seed_model(path: Path) -> None:
    model = SGDClassifier(loss="log_loss", random_state=1, max_iter=300, tol=1e-3)
    x = pd.DataFrame([[0.1] * len(FEATURES_V1), [0.9] * len(FEATURES_V1)], columns=FEATURES_V1)
    y = [0, 1]
    model.fit(x, y)
    joblib.dump({"model_type": "sgd_classifier", "model": model, "features_order": FEATURES_V1}, path)


def _holdout(path: Path) -> Path:
    rows = []
    for i in range(20):
        label = 1 if i % 2 == 0 else 0
        v = 0.9 if label == 1 else 0.1
        row = {k: v for k in FEATURES_V1}
        row["isFraud"] = label
        rows.append(row)
    out = pd.DataFrame(rows)
    holdout_path = path / "holdout.parquet"
    out.to_parquet(holdout_path, index=False)
    return holdout_path


def test_evaluate_model_on_holdout_returns_metrics(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    payload = joblib.load(model_path)
    holdout = _holdout(tmp_path)

    metrics = evaluate_model_on_holdout(model=payload["model"], holdout_parquet=holdout)
    assert metrics["rows_holdout"] == 20.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_check_promotion_thresholds_detects_failures():
    passed, reasons = check_promotion_thresholds(
        metrics={"precision_at_0_5": 0.1, "recall_at_0_5": 0.2, "pr_auc": 0.3},
        thresholds=PromotionThresholds(min_precision=0.2, min_recall=0.2, min_pr_auc=0.4),
    )
    assert passed is False
    assert len(reasons) == 2


def test_evaluate_and_maybe_rollback_restores_previous_payload(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    holdout = _holdout(tmp_path)
    updater = OnlineSGDUpdater(model_path=model_path, batch_size=2)

    backup = joblib.load(model_path)
    # Force changed online update count to verify rollback.
    updater.online_update_count = 999
    decision = evaluate_and_maybe_rollback(
        updater=updater,
        backup_payload=backup,
        holdout_parquet=holdout,
        thresholds=PromotionThresholds(min_precision=1.1),
    )
    assert decision.passed is False
    assert decision.rolled_back is True
    restored = joblib.load(model_path)
    assert restored.get("online_update_count", 0) == int(backup.get("online_update_count", 0))


def test_evaluate_model_on_holdout_requires_target_column(tmp_path: Path):
    model_path = tmp_path / "sgd.joblib"
    _seed_model(model_path)
    payload = joblib.load(model_path)

    bad = pd.DataFrame([{k: 0.1 for k in FEATURES_V1}])
    holdout_path = tmp_path / "bad.parquet"
    bad.to_parquet(holdout_path, index=False)
    with pytest.raises(ValueError, match="must include `isFraud`"):
        evaluate_model_on_holdout(model=payload["model"], holdout_parquet=holdout_path)

