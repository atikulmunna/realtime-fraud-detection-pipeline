"""Reproducible local benchmark for latency and alert-budget quality."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data.splitting import chronological_split
from src.streaming.ensemble_scoring import EnsembleModels, load_ensemble_models
from src.streaming.pipeline_skeleton import process_stream_payload


class _IfModel:
    def decision_function(self, x):
        amount = x[:, 0]
        return -(amount - 100.0) / 100.0


class _Scaler:
    def transform(self, x):
        return x / 1000.0


class _AeModel:
    def predict(self, x):
        out = np.array(x, copy=True)
        out[:, 0] = np.minimum(out[:, 0], 0.12)
        return out


class _SgdModel:
    def predict_proba(self, x):
        amount = np.clip(x[:, 0] / 1000.0, 0.0, 1.0)
        p1 = np.clip(0.1 + (0.8 * amount), 0.0, 1.0)
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T


@dataclass(frozen=True)
class BenchmarkConfig:
    n_events: int = 1000
    fraud_ratio: float = 0.1
    alert_budget_ratio: float = 0.005
    score_threshold: float = 0.65
    seed: int = 42
    txn_velocity_1h: int = 1
    use_trained_models: bool = False
    if_model_path: str | Path = "models/isolation_forest_v1.joblib"
    ae_model_path: str | Path = "models/autoencoder_v1.joblib"
    sgd_model_path: str | Path = "models/sgd_classifier_v1.joblib"
    evaluation_parquet: str | Path | None = None


def _build_models() -> EnsembleModels:
    return EnsembleModels(
        if_model=_IfModel(),
        ae_model=_AeModel(),
        ae_scaler=_Scaler(),
        ae_threshold_p99=0.01,
        sgd_model=_SgdModel(),
        model_source="demo",
        model_version="demo-v1",
    )


def _load_models_from_artifacts(config: BenchmarkConfig) -> EnsembleModels:
    paths = {
        "if_model_path": Path(config.if_model_path),
        "ae_model_path": Path(config.ae_model_path),
        "sgd_model_path": Path(config.sgd_model_path),
    }
    missing = [f"{name}={path}" for name, path in paths.items() if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing trained model artifact(s): {joined}")

    return load_ensemble_models(
        if_model_path=paths["if_model_path"],
        ae_model_path=paths["ae_model_path"],
        sgd_model_path=paths["sgd_model_path"],
    )


def _make_event(*, idx: int, amount: float) -> dict[str, Any]:
    old_balance = float(max(0.0, amount + 200.0))
    new_balance = float(max(0.0, old_balance - amount))
    return {
        "event_id": f"bench-{idx}",
        "timestamp": "2026-02-17T12:00:00Z",
        "user_id": f"C{1000 + idx}",
        "type": "TRANSFER" if amount >= 120.0 else "PAYMENT",
        "amount": float(amount),
        "old_balance_orig": old_balance,
        "new_balance_orig": new_balance,
    }


def _generate_labeled_events(config: BenchmarkConfig) -> list[tuple[dict[str, Any], int, int]]:
    if config.n_events <= 0:
        raise ValueError("n_events must be > 0")
    if not 0.0 < config.fraud_ratio < 1.0:
        raise ValueError("fraud_ratio must be between 0 and 1")
    if not 0.0 < config.alert_budget_ratio <= 1.0:
        raise ValueError("alert_budget_ratio must be within (0, 1]")

    rng = np.random.default_rng(config.seed)
    fraud_n = int(round(config.n_events * config.fraud_ratio))
    fraud_n = max(1, min(config.n_events - 1, fraud_n))
    normal_n = config.n_events - fraud_n

    normal_amounts = rng.uniform(5.0, 90.0, size=normal_n)
    fraud_amounts = rng.uniform(300.0, 1100.0, size=fraud_n)

    rows: list[tuple[dict[str, Any], int, int]] = []
    idx = 0
    for amount in normal_amounts:
        rows.append((_make_event(idx=idx, amount=float(amount)), 0, config.txn_velocity_1h))
        idx += 1
    for amount in fraud_amounts:
        rows.append((_make_event(idx=idx, amount=float(amount)), 1, config.txn_velocity_1h))
        idx += 1
    order = rng.permutation(len(rows))
    return [rows[int(i)] for i in order]


def _timestamp_to_wire(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.isoformat().replace("+00:00", "Z")


def _dataset_hash(df: pd.DataFrame, columns: list[str]) -> str:
    hashed = pd.util.hash_pandas_object(df[columns], index=False).to_numpy().tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _load_representative_test_events(
    config: BenchmarkConfig,
) -> tuple[list[tuple[dict[str, Any], int, int]], dict[str, Any]]:
    if config.evaluation_parquet is None:
        raise ValueError("Trained-model benchmarks require evaluation_parquet.")

    path = Path(config.evaluation_parquet)
    df = pd.read_parquet(path)
    required = {
        "step",
        "timestamp",
        "nameOrig",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "txn_velocity_1h",
        "isFraud",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Evaluation parquet is missing required columns: {missing}")

    test_df = chronological_split(df).test
    if len(test_df) > config.n_events:
        selected = (
            test_df.sample(n=config.n_events, random_state=config.seed).sort_values("step").reset_index(drop=True)
        )
    else:
        selected = test_df.reset_index(drop=True)

    rows: list[tuple[dict[str, Any], int, int]] = []
    for idx, row in selected.iterrows():
        event = {
            "event_id": f"eval-{int(row['step'])}-{idx}",
            "timestamp": _timestamp_to_wire(row["timestamp"]),
            "user_id": str(row["nameOrig"]),
            "type": str(row["type"]),
            "amount": float(row["amount"]),
            "old_balance_orig": float(row["oldbalanceOrg"]),
            "new_balance_orig": float(row["newbalanceOrig"]),
        }
        rows.append((event, int(row["isFraud"]), int(row["txn_velocity_1h"])))

    hash_columns = sorted(required)
    metadata = {
        "source": str(path),
        "split": "chronological_test",
        "rows_available": int(len(test_df)),
        "rows_selected": int(len(selected)),
        "fraud_count": int(selected["isFraud"].sum()),
        "dataset_hash": _dataset_hash(selected, hash_columns),
    }
    return rows, metadata


def _precision_recall_at_budget(
    *,
    y_true: list[int],
    scores: list[float],
    alert_budget_ratio: float,
) -> tuple[float, float, int]:
    n = len(y_true)
    if n == 0:
        return 0.0, 0.0, 0
    budget_n = max(1, int(round(n * alert_budget_ratio)))
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    alert_idx = set(ranked[:budget_n])

    tp = sum(1 for i in alert_idx if y_true[i] == 1)
    fp = budget_n - tp
    positives = sum(1 for y in y_true if y == 1)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / positives) if positives > 0 else 0.0
    return precision, recall, budget_n


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    models = _load_models_from_artifacts(config) if config.use_trained_models else _build_models()
    if config.use_trained_models:
        labeled, dataset_metadata = _load_representative_test_events(config)
    else:
        labeled = _generate_labeled_events(config)
        synthetic_payload = json.dumps(labeled, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        dataset_metadata = {
            "source": "deterministic_synthetic_demo",
            "split": "demo",
            "rows_available": len(labeled),
            "rows_selected": len(labeled),
            "fraud_count": sum(label for _, label, _ in labeled),
            "dataset_hash": hashlib.sha256(synthetic_payload).hexdigest(),
        }

    latencies_ms: list[float] = []
    valid_scores: list[float] = []
    valid_labels: list[int] = []
    routed_anomalies = 0
    routed_normal = 0
    routed_dlq = 0
    threshold_tp = 0
    threshold_fp = 0
    threshold_fn = 0
    threshold_tn = 0

    for event, label, txn_velocity in labeled:
        t0 = perf_counter()
        topic, payload = process_stream_payload(
            event,
            models=models,
            score_threshold=config.score_threshold,
            anomaly_topic="anomalies",
            normal_topic="metrics",
            dlq_topic="dead-letter",
            txn_velocity_1h=txn_velocity,
        )
        latencies_ms.append((perf_counter() - t0) * 1000.0)

        if topic == "anomalies":
            routed_anomalies += 1
            if label == 1:
                threshold_tp += 1
            else:
                threshold_fp += 1
        elif topic == "metrics":
            routed_normal += 1
            if label == 1:
                threshold_fn += 1
            else:
                threshold_tn += 1
        else:
            routed_dlq += 1

        if topic != "dead-letter" and "scores" in payload:
            valid_scores.append(float(payload["scores"]["ensemble_score"]))
            valid_labels.append(int(label))

    p95_latency_ms = float(np.percentile(np.array(latencies_ms, dtype=float), 95))
    precision, recall, alerts_sent = _precision_recall_at_budget(
        y_true=valid_labels,
        scores=valid_scores,
        alert_budget_ratio=config.alert_budget_ratio,
    )
    pr_auc = (
        float(average_precision_score(np.array(valid_labels, dtype=int), np.array(valid_scores, dtype=float)))
        if len(set(valid_labels)) > 1
        else 0.0
    )
    threshold_precision = (
        float(threshold_tp / (threshold_tp + threshold_fp)) if threshold_tp + threshold_fp > 0 else 0.0
    )
    threshold_recall = float(threshold_tp / (threshold_tp + threshold_fn)) if threshold_tp + threshold_fn > 0 else 0.0

    return {
        "model_source": "trained_artifacts" if config.use_trained_models else "demo",
        "model_paths": (
            {
                "if_model_path": str(config.if_model_path),
                "ae_model_path": str(config.ae_model_path),
                "sgd_model_path": str(config.sgd_model_path),
            }
            if config.use_trained_models
            else {}
        ),
        "events_total": len(labeled),
        "events_scored": len(valid_scores),
        "dataset": dataset_metadata,
        "score_threshold": config.score_threshold,
        "alert_budget_ratio": config.alert_budget_ratio,
        "alerts_sent": int(alerts_sent),
        "routed_anomalies": int(routed_anomalies),
        "routed_normal": int(routed_normal),
        "routed_dlq": int(routed_dlq),
        "routing_rate": float(routed_anomalies / len(valid_scores)) if valid_scores else 0.0,
        "latency_ms": {
            "p50": float(np.percentile(np.array(latencies_ms, dtype=float), 50)),
            "p95": p95_latency_ms,
            "max": float(np.max(np.array(latencies_ms, dtype=float))),
        },
        "quality_at_budget": {
            "precision": precision,
            "recall": recall,
            "pr_auc": pr_auc,
        },
        "quality_at_threshold": {
            "true_positive": int(threshold_tp),
            "false_positive": int(threshold_fp),
            "false_negative": int(threshold_fn),
            "true_negative": int(threshold_tn),
            "precision": threshold_precision,
            "recall": threshold_recall,
        },
    }


def save_benchmark_report(report: dict[str, Any], *, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local benchmark and write JSON report.")
    parser.add_argument("--n-events", type=int, default=1000)
    parser.add_argument("--fraud-ratio", type=float, default=0.1)
    parser.add_argument("--alert-budget-ratio", type=float, default=0.005)
    parser.add_argument("--score-threshold", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-trained-models", action="store_true")
    parser.add_argument("--if-model-path", default="models/isolation_forest_v1.joblib")
    parser.add_argument("--ae-model-path", default="models/autoencoder_v1.joblib")
    parser.add_argument("--sgd-model-path", default="models/sgd_classifier_v1.joblib")
    parser.add_argument("--evaluation-parquet", default="data/processed/paysim_features.parquet")
    parser.add_argument("--output", default="reports/benchmark_report.json")
    args = parser.parse_args()

    config = BenchmarkConfig(
        n_events=args.n_events,
        fraud_ratio=args.fraud_ratio,
        alert_budget_ratio=args.alert_budget_ratio,
        score_threshold=args.score_threshold,
        seed=args.seed,
        use_trained_models=bool(args.use_trained_models),
        if_model_path=args.if_model_path,
        ae_model_path=args.ae_model_path,
        sgd_model_path=args.sgd_model_path,
        evaluation_parquet=args.evaluation_parquet,
    )
    report = run_benchmark(config)
    out = save_benchmark_report(report, output_path=args.output)
    print(json.dumps({"output_path": str(out), "summary": report}, indent=2))


if __name__ == "__main__":
    main()
