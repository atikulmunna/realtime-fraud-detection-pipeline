from pathlib import Path

from src.evaluation.benchmark_report import BenchmarkConfig, run_benchmark, save_benchmark_report


def test_run_benchmark_returns_latency_and_quality_metrics():
    report = run_benchmark(
        BenchmarkConfig(
            n_events=200,
            fraud_ratio=0.15,
            alert_budget_ratio=0.1,
            score_threshold=0.65,
            seed=7,
        )
    )

    assert report["events_total"] == 200
    assert report["events_scored"] > 0
    assert report["routed_dlq"] == 0
    assert report["alerts_sent"] == int(round(report["events_scored"] * 0.1))

    latency = report["latency_ms"]
    assert latency["p50"] >= 0.0
    assert latency["p95"] >= latency["p50"]
    assert latency["max"] >= latency["p95"]

    quality = report["quality_at_budget"]
    assert 0.0 <= quality["precision"] <= 1.0
    assert 0.0 <= quality["recall"] <= 1.0


def test_save_benchmark_report_writes_json(tmp_path: Path):
    report = {"latency_ms": {"p95": 10.0}, "quality_at_budget": {"precision": 0.5, "recall": 0.4}}
    out = save_benchmark_report(report, output_path=tmp_path / "bench.json")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert '"p95": 10.0' in text

