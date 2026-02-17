# Local Demo Runbook

## Objective
Run an end-to-end local validation for demo readiness and capture a machine-readable report.

## Prerequisites
- Conda env `realtime-fraud` exists with dependencies installed.
- Run from repository root.

## Commands
1. Optional infra startup:
   - `docker compose -f infra/docker-compose.yml up -d`
2. Run demo readiness check:
   - `scripts/tasks.ps1 -Task demo-readiness`
3. Run benchmark-only report:
   - `scripts/tasks.ps1 -Task benchmark-report`

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
  - Run baseline training tasks to generate model files in `models/`.
- Conda command issues:
  - Re-run commands with `conda run -n realtime-fraud ...`.
- Port conflicts:
  - Ensure ports `8000`, `9090`, `3000` are available for API/Prometheus/Grafana workflows.

