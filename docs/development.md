# Development and Clean-Machine Verification

## Prerequisites

- Git
- Python 3.11
- uv
- Docker Desktop with Compose
- PowerShell 7 for repository automation on Windows

## Reproduce

```powershell
git clone <repository-url>
Set-Location realtime-fraud-detection-pipeline
uv sync --locked
uv run pre-commit run --all-files
uv run pytest
docker compose -f infra/docker-compose.yml config --quiet
```

Then run the real service-chain qualification:

```powershell
.\scripts\smoke-compose.ps1
$env:RUN_COMPOSE_INTEGRATION = '1'
uv run pytest tests/integration -m integration --no-cov
```

The first command may download images and build the app/Flink images. The integration test uses unique identifiers and verifies duplicate acceptance, outbox drain, online update, and MLflow registration. Stop services without deleting durable volumes:

```powershell
docker compose -f infra/docker-compose.yml stop
```

## Data and training

PaySim download and model training are intentionally separate from environment bootstrap because the dataset is large and externally licensed. Follow the training sequence in the [README](../README.md). Never commit downloaded data, trained artifacts, reports, logs, credentials, or checkpoint data.
