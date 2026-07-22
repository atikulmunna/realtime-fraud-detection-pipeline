# Contracts, Guarantees, and Security

## Event and feature contracts

`schemas/event_v1.json` is the transaction boundary. Legacy `CASH-OUT` and `CASH-IN` values normalize to `CASH_OUT` and `CASH_IN`. Timestamps must be timezone-aware and numbers finite. `FEATURES_V1` is ordered and shared by training, online feedback, and streaming scoring.

Every failed parse, feature extraction, scoring operation, or late event becomes `schemas/dlq_v1.json` with `stage` and `error_code`. PyFlink uses event timestamps, a two-minute bounded-out-of-order watermark, and per-user UTC-hour managed state for `txn_velocity_1h`.

## Delivery guarantees

- API acknowledgement means feedback and its outbox row committed together in Postgres.
- `feedback_id` and `Idempotency-Key` provide application-level exactly-once acceptance.
- The relay uses an idempotent Kafka producer with `acks=all`; a record remains pending until publish acknowledgement.
- The updater disables auto-commit, persists processed feedback IDs and update manifests, and commits offsets only after durable promotion/rollback.
- Flink sources participate in checkpoints; anomaly, metrics, and DLQ Kafka sinks use exactly-once transactions.
- Redelivery can occur across failures, but persisted identifiers prevent duplicate online training.

These guarantees depend on correctly configured external Kafka replication, Postgres durability, object storage, and checkpoint retention.

## Readiness and quality

Liveness reports process availability. Readiness fails closed for missing/corrupt production artifacts, contract mismatch, demo models outside development, zero routed anomalies, and failed quality gates. Promotion rejects candidates below absolute gates or more than two percentage points behind the champion.

## Security

The feedback API requires `X-API-Key` outside development. Payloads forbid unknown fields. Compose mounts the local API key as a secret; Kubernetes expects an externally managed `fraud-secrets` object. Logs are JSON and correlate identifiers without logging API keys. Kubernetes containers drop capabilities, prohibit privilege escalation, run as non-root, and use default-deny network policy.

API-key authentication is a reference minimum, not a complete internet-facing identity system. Production should add TLS, workload identity, analyst authorization, rate limiting, audit retention, secret rotation, encryption keys, and private network endpoints.

## Recovery

Do not reset Kafka offsets or delete Postgres/outbox, MLflow, checkpoint, or savepoint data during incident response. Restore dependencies and allow replay. Roll models back by moving the MLflow `champion` alias. Roll Flink code through an operator savepoint upgrade. See the service-specific runbooks for commands.
