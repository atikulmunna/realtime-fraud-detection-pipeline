"""End-to-end local demo readiness checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.demo.demo_flow import run_demo_flow
from src.evaluation.benchmark_report import BenchmarkConfig, run_benchmark


def run_demo_readiness_check(
    *,
    model_path: str | Path = "models/sgd_classifier_demo.joblib",
    benchmark_events: int = 500,
    benchmark_alert_budget_ratio: float = 0.1,
    latency_slo_ms: float = 500.0,
    use_trained_models: bool = False,
    if_model_path: str | Path = "models/isolation_forest_v1.joblib",
    ae_model_path: str | Path = "models/autoencoder_v1.joblib",
    sgd_model_path: str | Path = "models/sgd_classifier_v1.joblib",
) -> dict[str, Any]:
    demo = run_demo_flow(model_path=model_path)
    benchmark = run_benchmark(
        BenchmarkConfig(
            n_events=int(benchmark_events),
            alert_budget_ratio=float(benchmark_alert_budget_ratio),
            use_trained_models=bool(use_trained_models),
            if_model_path=if_model_path,
            ae_model_path=ae_model_path,
            sgd_model_path=sgd_model_path,
        )
    )

    checks = {
        "demo_has_anomalies": bool(demo.get("anomalies", 0) >= 1),
        "demo_has_feedback": bool(demo.get("feedback_published", 0) >= 1),
        "demo_online_updated": bool(demo.get("online_updated", False)),
        "benchmark_latency_slo_met": bool(benchmark["latency_ms"]["p95"] <= latency_slo_ms),
        "benchmark_quality_fields_present": bool(
            "precision" in benchmark["quality_at_budget"] and "recall" in benchmark["quality_at_budget"]
        ),
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
    parser.add_argument("--use-trained-models", action="store_true")
    parser.add_argument("--if-model-path", default="models/isolation_forest_v1.joblib")
    parser.add_argument("--ae-model-path", default="models/autoencoder_v1.joblib")
    parser.add_argument("--sgd-model-path", default="models/sgd_classifier_v1.joblib")
    parser.add_argument("--output", default="reports/demo_readiness_report.json")
    args = parser.parse_args()

    report = run_demo_readiness_check(
        model_path=args.model_path,
        benchmark_events=args.benchmark_events,
        benchmark_alert_budget_ratio=args.benchmark_alert_budget_ratio,
        latency_slo_ms=args.latency_slo_ms,
        use_trained_models=bool(args.use_trained_models),
        if_model_path=args.if_model_path,
        ae_model_path=args.ae_model_path,
        sgd_model_path=args.sgd_model_path,
    )
    out = save_readiness_report(report, output_path=args.output)
    print(json.dumps({"output_path": str(out), "summary": report}, indent=2))

    if not report["overall_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
