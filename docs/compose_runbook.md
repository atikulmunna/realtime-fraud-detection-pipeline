# Compose Stack Runbook

The local stack builds the feedback API, durable outbox relay, online updater, MLflow server, and Flink runtime from repository Dockerfiles. Kafka topics are created by the one-shot `topic-init` service; Kafka auto-creation is disabled. Postgres, Kafka, MLflow artifacts, checkpoints, Prometheus data, and Grafana data use named volumes.

## Start and verify

```powershell
Copy-Item infra/.env.example infra/.env
# Replace both placeholder values in infra/.env and the development API-key secret file for shared environments.
docker compose --env-file infra/.env -f infra/docker-compose.yml up -d --build
.\scripts\smoke-compose.ps1
```

The smoke command creates a deterministic local SGD seed only when the expected artifact is absent. It then submits uniquely identified feedback, waits for the Postgres outbox to publish it, and waits for the updater to register and promote an MLflow candidate.

Endpoints:

- Feedback API: `http://localhost:8000/health`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Flink: `http://localhost:8081`

## Recovery

Use `docker compose -f infra/docker-compose.yml restart <service>` for a stateless process. Do not delete Kafka, Postgres, MLflow, or checkpoint volumes during incident recovery. Feedback already accepted by the API remains in Postgres until the relay marks it published, and the updater commits Kafka offsets only after promotion or rollback is durable.

The checked-in API-key file is development-only. Production deployments must replace it with an external secret manager and must not reuse the local Postgres password default.
