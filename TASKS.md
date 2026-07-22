# Realtime Fraud Detection Pipeline Improvement Tasks

This file is the execution ledger for hardening the repository into a cloud-neutral production reference.
Tasks are completed in order unless a blocking dependency is recorded here.

## Workflow

- Status values: `TODO`, `IN PROGRESS`, `BLOCKED`, `DONE`.
- Exactly one task may be `IN PROGRESS`.
- Each task is intended to be one reviewable pull request.
- A task is `DONE` only after its acceptance commands pass and evidence is recorded.
- Preserve local data, model artifacts, reports, and runtime logs unless a task explicitly migrates them.

## Universal quality gates

```powershell
uv run ruff check src tests scripts
uv run mypy src
uv run pytest
docker compose -f infra/docker-compose.yml config --quiet
```

## Tasks

### T01 — Repository foundations — DONE

Goal: establish reproducible dependencies and enforce consistent local and CI quality checks.

- [x] Add `pyproject.toml`, `uv.lock`, Ruff, mypy, pytest coverage, and pre-commit configuration.
- [x] Retain `requirements.txt` as a generated compatibility export.
- [x] Update CI to use the locked Python 3.11 environment.
- [x] Ignore runtime logs and OS metadata without deleting existing local files.
- [x] Pass lint, typing, compilation, tests, 85% source coverage, and Compose validation.

Evidence: 103 tests passed with 85.03% coverage; Ruff, mypy, compileall, uv lock validation, pre-commit config validation, and Compose config validation passed on 2026-07-20.

### T02 — Truthful readiness and fail-closed model loading — DONE

Goal: ensure a healthy service always represents a valid production model and meaningful evaluation.

- [x] Make demo fallback opt-in through `ALLOW_DEMO_MODE=true`.
- [x] Separate liveness `/health` from dependency/model readiness `/ready`.
- [x] Report model source, model version, and feature-contract version.
- [x] Evaluate anomaly routing and quality with the same model source.
- [x] Fail readiness for demo models outside development, invalid artifacts, zero routed anomalies, or failed quality gates.

Acceptance: API tests cover trained, missing, corrupt, and explicitly enabled demo models; readiness reports cannot pass with zero trained-model anomalies.

Evidence: 108 tests passed with 86.05% coverage; fail-closed, explicit demo, metadata readiness, contract mismatch, and zero-anomaly regression cases passed on 2026-07-20.

### T03 — Event, feature, and DLQ contracts — DONE

Goal: make ingestion compatible, strict, and consistent across every processing stage.

- [x] Accept legacy hyphenated transaction types and normalize to `CASH_OUT`/`CASH_IN`.
- [x] Apply complete JSON Schema validation, finite-number checks, and timezone-aware timestamps.
- [x] Emit schema-valid DLQ records with `stage` and `error_code` for parse, feature, scoring, and late-event failures.
- [x] Add backward-compatibility and offline/online feature-parity tests.

Evidence: 117 tests passed with 86.75% coverage; JSON Schema, normalization, finite/timezone validation, structured DLQ, and feature parity cases passed on 2026-07-20. Late-event production is reserved for T11's event-time Flink job.

### T04 — Representative data splitting and evaluation — DONE

Goal: replace misleading synthetic readiness with chronological, reproducible evaluation.

- [x] Split PaySim by `step`: 70% train, 15% promotion validation, 15% untouched test.
- [x] Build deterministic representative fixtures from the test distribution.
- [x] Report dataset hash, routing rate, confusion counts, PR-AUC, alert-budget metrics, and latency.
- [x] Enforce PR-AUC >= 0.10, precision@0.5% >= 0.10, and recall@0.5% >= 0.60.

Evidence: 122 tests passed with 86.96% coverage; chronological split, source-order preservation, representative trained evaluation, dataset hash, confusion metrics, and quality gates passed on 2026-07-20.

### T05 — Model improvement and calibration — DONE

Goal: produce an immutable, calibrated ensemble that meets operational quality gates.

- [x] Establish reproducible supervised and anomaly baselines.
- [x] Calibrate component scores and select weights/thresholds only on promotion validation data.
- [x] Bundle preprocessing, feature order, thresholds, metrics, dataset hash, and Git revision.
- [x] Reject challengers that fail absolute gates or regress a promotion metric by more than two percentage points.

Evidence: 128 tests passed with 87.43% coverage; deterministic calibration/selection, absolute and regression gates, immutable persistence, checksum verification, and bundle creation passed on 2026-07-20.

### T06 — MLflow registry lifecycle — DONE

Goal: make MLflow/Postgres the authoritative source for immutable candidate and champion models.

- [x] Add persistent MLflow tracking and registry services.
- [x] Register signed model bundles with contract and dataset metadata.
- [x] Implement `candidate` and `champion` aliases, promotion, and alias-based rollback.
- [x] Support only checksum-verified last-known champion caches when MLflow is unavailable.

Evidence: 132 tests passed with 86.69% coverage; registry logging, alias promotion/rollback, verified cache fallback, MLflow pyfunc validation, and Compose configuration passed on 2026-07-20.

### T07 — Durable and authenticated feedback ingestion — DONE

Goal: accept analyst feedback exactly once at the application level and durably queue it.

- [x] Add `feedback_id`, `Idempotency-Key`, canonical duplicate responses, and API-key authentication outside development.
- [x] Store feedback and an outbox record in one Postgres transaction.
- [x] Add a reliable `confluent-kafka` outbox relay with acknowledgements and retries.
- [x] Prove repeated feedback IDs are never republished.

Evidence: 136 tests passed with 86.24% coverage; authentication, durable acceptance, duplicate replay, outbox relay, idempotent producer configuration, and acknowledgement failure cases passed on 2026-07-20.

### T08 — Reliable online updates and promotion — DONE

Goal: prevent acknowledged feedback loss, duplicate training, and active-model corruption.

- [x] Use manual Kafka commits after durable processing and promotion/rollback.
- [x] Persist processed feedback IDs and update-batch manifests.
- [x] Flush partial batches on the configured interval.
- [x] Build candidates separately and promote them through MLflow without in-place champion overwrites.
- [x] Add concurrency control around update and reload operations.

Evidence: 141 tests passed with 87.41% coverage; manual offset commits, interval and exit flushes, persisted feedback deduplication, atomic candidate staging, MLflow candidate/champion alias promotion, rejection rollback, and updater locking passed on 2026-07-20.

### T09 — Production observability — DONE

Goal: expose actionable metrics and logs for data flow, model state, and reliability.

- [x] Replace the metrics stub with `prometheus_client` while preserving dashboard-compatible names.
- [x] Add structured correlated JSON logs.
- [x] Measure Kafka lag, outbox backlog, DLQ rate, model age, promotions, latency, and alert rate.
- [x] Correct alerts and update Grafana dashboards and runbooks.

Evidence: 145 tests passed with 87.39% coverage; real Prometheus collectors and latency histograms, correlated JSON logs, lag/backlog/model-age metrics, corrected traffic-aware alerts, expanded Grafana panels, and the observability runbook passed on 2026-07-20.

### T10 — Fully containerized local stack — DONE

Goal: run the complete application and infrastructure reproducibly through Compose.

- [x] Build images for the API, outbox relay, updater, MLflow, and Flink job.
- [x] Add health checks, persistent volumes, internal networks, dependency conditions, and secret injection.
- [x] Bootstrap Kafka topics automatically and remove genuinely unused services.
- [x] Provide a one-command model registration, traffic generation, and readiness smoke test.

Evidence: 147 tests passed with 87.66% coverage; Ruff, mypy, Compose validation, the Python 3.11 application image build/import, and the Python 3.10 PyFlink image build/import passed on 2026-07-20. The repeatable full-stack workflow is `scripts/smoke-compose.ps1`.

### T11 — Real checkpointed PyFlink pipeline — DONE

Goal: replace the in-process wrapper with Kafka-to-Flink event-time scoring.

- [x] Add checkpoint-managed Kafka source and exactly-once anomaly, metrics, and DLQ sinks.
- [x] Implement validation, feature extraction, calibrated scoring, and versioned outputs.
- [x] Implement per-user UTC-hour velocity state with a two-minute out-of-order allowance.
- [x] Reload validated champion versions through broadcast model-update state.

Evidence: 151 tests passed with 87.55% coverage; the rebuilt Flink image successfully constructed an execution graph containing transaction/model-update Kafka sources, keyed and broadcast processing, and three exactly-once writer/committer sink pairs on 2026-07-20. Model updates are emitted only after atomic promotion and are contract/checksum validated before reload.

### T12 — Integration, recovery, and performance tests — DONE

Goal: prove correctness under dependency failure, redelivery, and process restart.

- [x] Cover API -> outbox -> Kafka -> updater -> MLflow promotion through Compose-backed tests.
- [x] Test duplicates, outages, corrupt models, rollback, late events, and Flink checkpoint recovery.
- [x] Prove acknowledged feedback is not lost and redelivery does not duplicate updates.
- [x] Enforce quality gates and a configurable 500 ms local end-to-end p95 latency SLO.

Evidence: the opt-in Compose test passed against fresh isolated service volumes on 2026-07-22. It proved authenticated idempotent feedback acceptance, outbox drain, Kafka consumption and commit, a nonzero online update, artifact upload through MLflow, creation of a new ready model version, and post-promotion updater health. The recovery, redelivery, rollback, late-event, checkpoint, and configurable local p95 SLO cases also pass in the unit/recovery suite.

### T13 — Kubernetes deployment — DONE

Goal: provide a cloud-neutral production deployment reference.

- [x] Add Kustomize base and development overlays.
- [x] Deploy the streaming job through the Flink Kubernetes Operator.
- [x] Add probes, resources, autoscaling, disruption budgets, network policies, and rollback instructions.
- [x] Treat Kafka, Postgres, object storage, and secrets management as external production dependencies.

Evidence: 155 non-integration tests passed; both the base and development overlay rendered successfully with `kubectl kustomize` on 2026-07-20. Manifests include restricted security contexts, probes, resources, HPAs, PDBs, default-deny policies, an operator-managed savepoint-upgrade Flink job, and external dependency placeholders.

### T14 — Documentation and release qualification — DONE

Goal: make the architecture, guarantees, and operating procedures accurate and reproducible.

- [x] Document development, Compose, and Kubernetes topologies.
- [x] Document contracts, delivery guarantees, model promotion, security, recovery, and operations.
- [x] Complete a clean-machine workflow and save machine-readable release evidence.
- [x] Declare production-reference readiness only after every preceding task passes.

Evidence: the documented locked-environment sequence, all-file pre-commit hooks, 156-test non-integration suite, Compose configuration for default and streaming profiles, final application image build, fresh-state Compose integration, both Kustomize renders, and diff checks passed on 2026-07-22. Machine-readable results are saved in `release/evidence.json`; the repository is qualified as a production reference, not as a substitute for environment-specific load, security, and disaster-recovery acceptance.
