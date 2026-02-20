"""Model promotion guardrails for online updates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

from src.common.feature_contract import FEATURES_V1
from src.online.online_sgd_updater import OnlineSGDUpdater


@dataclass(frozen=True)
class PromotionThresholds:
    min_precision: float = 0.0
    min_recall: float = 0.0
    min_pr_auc: float = 0.0


@dataclass(frozen=True)
class PromotionDecision:
    passed: bool
    metrics: dict[str, float]
    reasons: tuple[str, ...]
    rolled_back: bool


def _score_model(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.array(model.predict_proba(x)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        raw = np.array(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    raise ValueError("Model must expose predict_proba or decision_function for guardrail evaluation.")


def evaluate_model_on_holdout(
    *,
    model: Any,
    holdout_parquet: str | Path,
    threshold: float = 0.5,
) -> dict[str, float]:
    df = pd.read_parquet(holdout_parquet)
    missing = [c for c in FEATURES_V1 if c not in df.columns]
    if missing:
        raise ValueError(f"Holdout data is missing required feature columns: {missing}")
    if "isFraud" not in df.columns:
        raise ValueError("Holdout data must include `isFraud` target column.")

    x = df[FEATURES_V1]
    y = df["isFraud"].astype(int).to_numpy()
    scores = _score_model(model, x)
    pred = (scores >= float(threshold)).astype(int)

    return {
        "rows_holdout": float(len(df)),
        "pr_auc": float(average_precision_score(y, scores)),
        "precision_at_0_5": float(precision_score(y, pred, zero_division=0)),
        "recall_at_0_5": float(recall_score(y, pred, zero_division=0)),
    }


def check_promotion_thresholds(
    *,
    metrics: dict[str, float],
    thresholds: PromotionThresholds,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if metrics["precision_at_0_5"] < thresholds.min_precision:
        reasons.append(
            f"precision_at_0_5={metrics['precision_at_0_5']:.6f} below min_precision={thresholds.min_precision:.6f}"
        )
    if metrics["recall_at_0_5"] < thresholds.min_recall:
        reasons.append(f"recall_at_0_5={metrics['recall_at_0_5']:.6f} below min_recall={thresholds.min_recall:.6f}")
    if metrics["pr_auc"] < thresholds.min_pr_auc:
        reasons.append(f"pr_auc={metrics['pr_auc']:.6f} below min_pr_auc={thresholds.min_pr_auc:.6f}")
    return (len(reasons) == 0), tuple(reasons)


def evaluate_and_maybe_rollback(
    *,
    updater: OnlineSGDUpdater,
    backup_payload: dict[str, Any],
    holdout_parquet: str | Path,
    thresholds: PromotionThresholds,
) -> PromotionDecision:
    metrics = evaluate_model_on_holdout(model=updater.model, holdout_parquet=holdout_parquet, threshold=0.5)
    passed, reasons = check_promotion_thresholds(metrics=metrics, thresholds=thresholds)
    if passed:
        return PromotionDecision(passed=True, metrics=metrics, reasons=reasons, rolled_back=False)

    # Roll back both in-memory updater state and persisted model artifact.
    updater.model = backup_payload["model"]
    updater.model_type = backup_payload.get("model_type", "sgd_classifier")
    updater.features_order = backup_payload.get("features_order", FEATURES_V1)
    updater.online_update_count = int(backup_payload.get("online_update_count", 0))
    joblib.dump(backup_payload, updater.model_path)
    return PromotionDecision(passed=False, metrics=metrics, reasons=reasons, rolled_back=True)

