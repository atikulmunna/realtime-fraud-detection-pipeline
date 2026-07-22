"""End-to-end local demo readiness checks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.demo.demo_flow import build_demo_models, run_demo_flow
from src.evaluation.benchmark_report import BenchmarkConfig, run_benchmark
from src.streaming.ensemble_scoring import load_ensemble_models


def run_demo_readiness_check(
    *,
    model_path: str | Path = "models/sgd_classifier_demo.joblib",
    benchmark_events: int = 500,
    benchmark_alert_budget_ratio: float = 0.1,
    latency_slo_ms: float = 500.0,
    use_trained_models: bool = True,
    allow_demo_mode: bool = False,
    if_model_path: str | Path = "models/isolation_forest_v1.joblib",
    ae_model_path: str | Path = "models/autoencoder_v1.joblib",
    sgd_model_path: str | Path = "models/sgd_classifier_v1.joblib",
    evaluation_parquet: str | Path = "data/processed/paysim_features.parquet",
) -> dict[str, Any]:
    if use_trained_models:
        stream_models = load_ensemble_models(
            if_model_path=if_model_path,
            ae_model_path=ae_model_path,
            sgd_model_path=sgd_model_path,
        )
    elif allow_demo_mode:
        stream_models = build_demo_models()
    else:
        raise ValueError("Demo readiness requires allow_demo_mode=True when trained models are disabled.")

    with TemporaryDirectory(prefix="fraud-readiness-") as tmp_dir:
        update_model_path = Path(tmp_dir) / "sgd-readiness.joblib"
        if use_trained_models:
            shutil.copy2(sgd_model_path, update_model_path)
        demo = run_demo_flow(model_path=update_model_path, stream_models=stream_models)

    benchmark = run_benchmark(
        BenchmarkConfig(
            n_events=int(benchmark_events),
            alert_budget_ratio=float(benchmark_alert_budget_ratio),
            use_trained_models=bool(use_trained_models),
            if_model_path=if_model_path,
            ae_model_path=ae_model_path,
            sgd_model_path=sgd_model_path,
            evaluation_parquet=evaluation_parquet if use_trained_models else None,
        )
    )

    checks = {
        "model_sources_match": bool(demo["model_source"] == benchmark["model_source"]),
        "production_model_active": bool(benchmark["model_source"] == "trained_artifacts" or allow_demo_mode),
        "demo_has_anomalies": bool(demo.get("anomalies", 0) >= 1),
        "demo_has_feedback": bool(demo.get("feedback_published", 0) >= 1),
        "demo_online_updated": bool(demo.get("online_updated", False)),
        "benchmark_latency_slo_met": bool(benchmark["latency_ms"]["p95"] <= latency_slo_ms),
        "benchmark_routes_anomalies": bool(benchmark.get("routed_anomalies", 0) >= 1),
        "benchmark_precision_gate_met": bool(benchmark["quality_at_budget"].get("precision", 0.0) >= 0.10),
        "benchmark_recall_gate_met": bool(benchmark["quality_at_budget"].get("recall", 0.0) >= 0.60),
        "benchmark_pr_auc_gate_met": bool(benchmark["quality_at_budget"].get("pr_auc", 0.0) >= 0.10),
    }
    overall_ok = all(checks.values())

    return {
        "overall_ok": overall_ok,
        "latency_slo_ms": float(latency_slo_ms),
        "checks": checks,
        "demo": demo,
        "benchmark": benchmark,
    }


def save_readiness_report(report: dict[str, Any], *, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end demo readiness checks.")
    parser.add_argument("--model-path", default="models/sgd_classifier_demo.joblib")
    parser.add_argument("--benchmark-events", type=int, default=500)
    parser.add_argument("--benchmark-alert-budget-ratio", type=float, default=0.1)
    parser.add_argument("--latency-slo-ms", type=float, default=500.0)
    parser.add_argument("--use-trained-models", action="store_true", help="Deprecated; trained models are the default.")
    parser.add_argument("--allow-demo-mode", action="store_true")
    parser.add_argument("--if-model-path", default="models/isolation_forest_v1.joblib")
    parser.add_argument("--ae-model-path", default="models/autoencoder_v1.joblib")
    parser.add_argument("--sgd-model-path", default="models/sgd_classifier_v1.joblib")
    parser.add_argument("--evaluation-parquet", default="data/processed/paysim_features.parquet")
    parser.add_argument("--output", default="reports/demo_readiness_report.json")
    args = parser.parse_args()

    report = run_demo_readiness_check(
        model_path=args.model_path,
        benchmark_events=args.benchmark_events,
        benchmark_alert_budget_ratio=args.benchmark_alert_budget_ratio,
        latency_slo_ms=args.latency_slo_ms,
        use_trained_models=not bool(args.allow_demo_mode),
        allow_demo_mode=bool(args.allow_demo_mode),
        if_model_path=args.if_model_path,
        ae_model_path=args.ae_model_path,
        sgd_model_path=args.sgd_model_path,
        evaluation_parquet=args.evaluation_parquet,
    )
    out = save_readiness_report(report, output_path=args.output)
    print(json.dumps({"output_path": str(out), "summary": report}, indent=2))

    if not report["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
