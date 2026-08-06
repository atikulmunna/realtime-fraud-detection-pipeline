# Local Demo Runbook

## Objective
Run an end-to-end local validation for demo readiness and capture a machine-readable report.

## Prerequisites
- Locked environment synced with `uv sync --locked`.
- Run from repository root.

## Commands
1. Optional infra startup:
   - `docker compose -f infra/docker-compose.yml up -d`
2. Run demo readiness check:
   - `uv run python -m src.demo.readiness_check --allow-demo-mode --output reports/demo_readiness_report.json`
3. Run benchmark-only report:
   - `uv run python -m src.evaluation.benchmark_report --output reports/benchmark_report.json`

Drop `--allow-demo-mode` to require real trained artifacts, and add `--use-trained-models` to the benchmark for the trained-model variant.

## Expected outputs
- `reports/demo_readiness_report.json`
- `reports/benchmark_report.json`

## Pass criteria
- `overall_ok = true` in `reports/demo_readiness_report.json`
- `checks.benchmark_latency_slo_met = true`
- `checks.demo_has_anomalies = true`
- `checks.demo_online_updated = true`

## Troubleshooting
- Missing model artifacts:
  - Run the training sequence in the [README](../README.md) to generate model files in `models/`.
- Port conflicts:
  - Ensure ports `8000`, `9090`, `3000` are available for API/Prometheus/Grafana workflows.

