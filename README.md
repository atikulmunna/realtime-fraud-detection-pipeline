# Realtime Fraud Detection Pipeline

Real-time fraud detection workflow with:
- offline training on PaySim
- streaming scoring pipeline
- feedback API
- online SGD updater
- Prometheus/Grafana observability

## Requirements
- Python 3.11+
- Conda (`realtime-fraud` env recommended)
- Docker Desktop (for Kafka/Flink/Prometheus/Grafana)

## Setup
```powershell
conda create -n realtime-fraud python=3.11 -y
conda activate realtime-fraud
pip install -r requirements.txt
```

## Run
From repo root:

```powershell
# Start infra
docker compose -f infra/docker-compose.yml up -d

# Run tests
scripts/tasks.ps1 -Task test

# Local demo flow
scripts/tasks.ps1 -Task local-demo

# Start feedback API
scripts/tasks.ps1 -Task start-api

# Start always-on online service metrics endpoint (for Prometheus online_updater target)
scripts/tasks.ps1 -Task start-online-service

# API healthcheck
scripts/tasks.ps1 -Task healthcheck-api

# Run online updater once
scripts/tasks.ps1 -Task start-updater

# Run updater with sample feedback
scripts/tasks.ps1 -Task start-updater-sample

# Benchmark report
scripts/tasks.ps1 -Task benchmark-report

# End-to-end readiness check
scripts/tasks.ps1 -Task demo-readiness

# Benchmark with trained model artifacts in models/
scripts/tasks.ps1 -Task benchmark-report-trained

# Readiness check with trained model artifacts in models/
scripts/tasks.ps1 -Task demo-readiness-trained

# Seed online service counters so Prometheus/Grafana panels show data quickly
scripts/tasks.ps1 -Task seed-online-metrics
```

## Outputs
- `reports/benchmark_report.json`
- `reports/demo_readiness_report.json`

## Endpoints
- API: `http://127.0.0.1:8000`
- API metrics: `http://127.0.0.1:8000/metrics`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

## Common Issues
- `NoBrokersAvailable`: start Kafka via Docker Compose.
- Missing model files in updater: generate baseline models first.
- Port conflict (`8000`, `9090`, `3000`): stop the process using the port.
