# Realtime Fraud Detection Pipeline Wiki (A to Z)

## 1) What this project is
This repository is an end-to-end, local-first fraud detection system built around:
- Offline training on PaySim data.
- Stream-style inference (parse -> feature extraction -> ensemble scoring -> routing).
- Analyst feedback ingestion.
- Online model updates (`SGDClassifier.partial_fit`).
- Operational visibility with Prometheus and Grafana.

The project is intentionally structured so you can:
- Run everything locally.
- Validate each stage independently.
- Move from prototype behavior to more realistic streaming behavior without rewriting from scratch.

## 2) Why this project exists
Fraud detection systems usually fail at one or more of these:
- Training-serving mismatch (features differ between offline training and online scoring).
- No feedback loop (analyst labels never reach model updates).
- Limited observability (hard to explain why alerts do or do not fire).
- Difficult reproducibility (hard for new contributors to run or verify the pipeline).

This repository addresses those by:
- Enforcing a feature contract (`FEATURES_V1`) used by both training and serving.
- Exposing feedback APIs and online updating services.
- Emitting Prometheus metrics and provisioning Grafana dashboards.
- Shipping CLI/scripted workflows + tests for nearly every module.

## 3) End-to-end mental model
If you remember one flow, remember this:

1. Data prep creates `data/processed/paysim_features.parquet`.
2. Models are trained and stored in `models/*.joblib`.
3. Incoming transaction events are parsed and validated.
4. Events are converted to serving features.
5. Ensemble model scores each event and routes it to `anomalies` or `metrics`.
6. Analysts provide labels through API.
7. Online updater consumes labels and does batched `partial_fit`.
8. Metrics are scraped by Prometheus and visualized in Grafana.

## 4) High-level architecture (runtime)
- Data layer:
  - CSV ingestion and validation.
  - Feature engineering and EDA.
- Model layer:
  - Isolation Forest.
  - AutoEncoder-like MLP reconstructor.
  - SGD classifier.
- Serving layer:
  - Event parser.
  - Feature extractor.
  - Ensemble scoring and routing.
  - Feedback API.
  - Online updater service.
- Observability layer:
  - `/metrics` endpoints.
  - Prometheus scrape config.
  - Grafana pre-provisioned dashboard.

## 5) Repository map (what each folder does)
- `src/`: all production Python modules.
- `tests/`: unit tests (core contract of behavior).
- `scripts/`: task runners and start scripts (PowerShell + CMD + Python entry wrappers).
- `infra/`: Docker Compose and monitoring provisioning.
- `schemas/`: JSON schemas for event and DLQ payloads.
- `docs/`: supporting docs and generated analysis artifacts.
- `assets/`: screenshots for repository documentation.
- `data/`: local data staging (`raw`, `processed`, and sample fixtures).
- `models/`: trained model artifacts and training metrics.
- `reports/`: generated benchmark/readiness outputs.

## 6) Detailed source guide (file by file)

### 6.1 `src/common`
- `src/common/feature_contract.py`
  - Defines `FEATURES_V1` (canonical feature order).
  - Defines leakage fields to remove in serving (`LEAKAGE_FIELDS`).
  - This is the core guardrail against train/serve skew.
- `src/common/metrics_stub.py`
  - In-memory Prometheus-style metrics registry.
  - Supports counters and gauges.
  - Renders Prometheus text exposition format via `render_prometheus()`.

### 6.2 `src/data`
- `src/data/download_paysim.py`
  - Downloads PaySim through `kagglehub`.
  - Selects CSV (largest by default if multiple files).
  - Copies selected CSV into `data/raw/`.
- `src/data/prepare_paysim.py`
  - Validates schema and domain assumptions:
  - Required columns present.
  - Allowed transaction types.
  - Fraud type constraints (`isFraud` mostly on transfer/cash-out).
  - Basic step-range sanity checks.
  - Outputs structured validation result.
- `src/data/feature_engineering.py`
  - Converts raw PaySim columns to serving/training features.
  - Adds synthetic timestamp from step.
  - Computes ratio, balance-difference, one-hot type flags, hour-of-day, transaction velocity.
  - Writes parquet artifact used by model trainers.
- `src/data/eda_paysim.py`
  - Produces summary statistics:
  - Class imbalance.
  - Fraud-by-type table.
  - Amount quantiles.
  - Step range and invalid fraud type checks.

### 6.3 `src/models`
- `src/models/train_if.py`
  - Trains Isolation Forest using `FEATURES_V1`.
  - Prefers normal-only rows if `isFraud` exists.
  - Saves model and JSON metrics.
- `src/models/train_ae.py`
  - Trains MLP regressor as a reconstruction model (autoencoder-like behavior).
  - Scales features with `MinMaxScaler`.
  - Computes validation reconstruction error and `threshold_p99`.
  - Saves payload (`model`, `scaler`, `threshold_p99`) + JSON metrics.
- `src/models/train_sgd.py`
  - Trains supervised `SGDClassifier` with `log_loss`.
  - Tracks PR-AUC, precision, recall on validation split.
  - Saves model payload + JSON metrics.
- `src/models/mlflow_logging.py`
  - Optional MLflow logging adapter.
  - Coerces params/metrics types before logging.
  - Logs artifacts if files exist.

### 6.4 `src/streaming`
- `src/streaming/event_parser.py`
  - Parses JSON payloads (or dict/bytes), validates against schema requirements.
  - Rejects bad types/values/constraints.
  - Strips leakage fields before downstream processing.
  - Returns either `ParseResult.event` (valid) or DLQ payload with error metadata.
- `src/streaming/feature_extractor.py`
  - Builds serving-time feature vector from event fields.
  - Enforces required fields and computes feature math.
- `src/streaming/ensemble_scoring.py`
  - Loads IF/AE/SGD artifacts into a single `EnsembleModels` object.
  - Converts event features to ordered vector via `FEATURES_V1`.
  - Computes:
  - IF score (sigmoid of negative decision function).
  - AE score (normalized reconstruction error).
  - SGD probability.
  - Weighted ensemble score.
  - Routes to anomaly/normal topics via threshold.
- `src/streaming/pipeline_skeleton.py`
  - Main streaming flow:
  - parse -> enrich features -> optional score -> route.
  - Handles DLQ fallback on failures.
  - Emits stream metrics (`stream_events_*`, latency metrics).
- `src/streaming/adaptive_threshold.py`
  - Adaptive threshold helper using quantiles with floor/ceiling/min-samples guards.
- `src/streaming/parser_pipeline.py`
  - Batch helper around parser layer for local tests/integration style runs.
- `src/streaming/kafka_bootstrap.py`
  - Ensures required Kafka topics exist.
  - Handles race condition for already-created topics.
- `src/streaming/flink_job_wrapper.py`
  - Flink integration scaffold:
  - env-driven config loading.
  - model path validation.
  - local wrapper execution path for routing logic.

### 6.5 `src/api`
- `src/api/main.py`
  - FastAPI app for analyst feedback.
  - Endpoints:
  - `GET /health`
  - `GET /metrics`
  - `POST /feedback`
  - Tracks request/publish/error counters.
- `src/api/feedback_publisher.py`
  - Publisher abstraction.
  - Kafka implementation with conservative startup behavior (fixed `api_version`) to reduce eager broker failures.
- `src/api/app_entry.py`
  - Uvicorn app factory wiring Kafka bootstrap/topic configuration via environment variables.

### 6.6 `src/online`
- `src/online/online_sgd_updater.py`
  - Stateful online learning component.
  - Buffers feedback rows until batch criteria.
  - Applies `partial_fit` updates.
  - Persists updated model + metadata.
  - Emits update signals (in-memory publisher by default in tests).
  - Tracks updater metrics.
- `src/online/updater_service.py`
  - Two modes in one module:
  - One-shot updater execution (`run_updater_service_once` via CLI).
  - Always-on online service app (`app_factory`) exposing:
  - `GET /health`
  - `GET /metrics`
  - `POST /feedback`
  - `POST /feedback/sample`
  - `POST /stream/event`
  - `POST /stream/sample`
  - `POST /flush`
  - Loads trained stream models when available; falls back to demo models if not.

### 6.7 `src/demo`
- `src/demo/demo_flow.py`
  - End-to-end local demo:
  - synthetic events -> stream scoring -> feedback API call -> online update.
  - Returns compact summary metrics.
- `src/demo/readiness_check.py`
  - Readiness validator combining:
  - demo flow success checks.
  - benchmark latency/quality checks.
  - pass/fail output in JSON report.
- `src/demo/local_runner.py`
  - command utilities used by local script wrappers.

### 6.8 `src/evaluation`
- `src/evaluation/benchmark_report.py`
  - Reproducible benchmark and quality report generator.
  - Generates synthetic labeled traffic for evaluation.
  - Supports:
  - demo model mode.
  - trained artifact mode (`--use-trained-models`).
  - Produces latency quantiles and precision/recall under fixed alert budget.

### 6.9 `src/observability`
- `src/observability/healthcheck.py`
  - Programmatic health checks against API `/health` and `/metrics`.
  - Used by task scripts for quick operational verification.

## 7) Supporting files outside `src`

### 7.1 scripts
- `scripts/tasks.ps1`
  - Main command dispatcher.
  - Includes init/test/demo/service/benchmark/readiness helpers.
- `scripts/start_feedback_api.ps1`
  - Starts feedback API on port 8000.
- `scripts/start_online_service.ps1`
  - Starts always-on online service on port 8001.
- `scripts/start_online_updater.ps1`
  - Runs one-shot updater pass (optional JSONL input).
- `scripts/run_local_demo.ps1`
  - Executes local demo flow; optional compose bring-up.
- `scripts/healthcheck_api.ps1`
  - Executes health check module.
- `scripts/*.cmd`
  - Windows-friendly wrappers for `.ps1` scripts.
- `scripts/benchmark_report.py`, `scripts/demo_readiness.py`, `scripts/demo_run.py`
  - Python launcher wrappers for report/demo commands.

### 7.2 infra
- `infra/docker-compose.yml`
  - Spins up local services:
  - Zookeeper, Kafka, Redis, Postgres, Prometheus, Grafana, Flink JobManager/TaskManager.
- `infra/prometheus/prometheus.yml`
  - Scrape jobs:
  - `prometheus` (self)
  - `feedback_api` (`host.docker.internal:8000`)
  - `online_updater` (`host.docker.internal:8001`)
- `infra/grafana/provisioning/datasources/prometheus.yml`
  - Provisioned Prometheus datasource (`uid: prometheus`).
- `infra/grafana/dashboards/realtime_fraud_overview.json`
  - Dashboard panels for feedback, stream, latency, updater, and error/skip metrics.

### 7.3 schemas
- `schemas/event_v1.json`
  - Canonical input event schema for parser validation.
- `schemas/dlq_v1.json`
  - DLQ payload schema.

### 7.4 tests
The test suite is broad and purpose-driven:
- Contract and schema enforcement tests.
- Feature and parser correctness tests.
- Model training artifact tests.
- Streaming routing/scoring tests.
- API and updater behavior tests.
- Dashboard and observability tests.
- Benchmark/readiness tests.

`tests/unit/*` acts as executable documentation for expected behavior.

## 8) Workflow guide for a newcomer

### 8.1 First run
1. Create a separate conda environment.
2. Install dependencies from `requirements.txt`.
3. Start Docker stack.
4. Download and validate PaySim.
5. Build engineered features parquet.
6. Train IF/AE/SGD.
7. Start API service and online service.
8. Seed online metrics and open Prometheus/Grafana.
9. Run benchmark and readiness reports.

### 8.2 Daily development loop
1. Pull latest code.
2. Run tests.
3. Run the specific task you are modifying.
4. Validate metrics/targets and local behavior.
5. Add/adjust unit tests for your change.

### 8.3 Debugging order
1. Service health (`/health`).
2. Metrics endpoint non-empty (`/metrics`).
3. Prometheus targets `UP`.
4. Raw queries in Prometheus.
5. Grafana panel query correctness and datasource mapping.

## 9) How the pipeline works (detailed)

### 9.1 Offline pipeline
1. `download_paysim.py` fetches raw CSV.
2. `prepare_paysim.py` validates schema and fraud constraints.
3. `feature_engineering.py` computes training/serving-aligned features.
4. Model trainers fit and persist model artifacts.
5. Optional MLflow logger records runs.

### 9.2 Online scoring pipeline
1. Raw event enters parser.
2. Parser validates and sanitizes (leakage removal).
3. Feature extractor computes `FEATURES_V1` keys.
4. Ensemble scorer computes IF + AE + SGD + weighted score.
5. Router sends high-score events to anomaly channel; otherwise normal metrics channel.
6. Metrics are updated for throughput, routing, and latency.

### 9.3 Feedback -> model update loop
1. Analyst sends feedback to API.
2. Feedback event is published and counted.
3. Online updater accepts/filters feedback.
4. Updater flushes at batch threshold or forced flush.
5. Updated model and metadata are persisted.
6. Update counters/signals are emitted.

## 10) Data contracts and invariants
- `FEATURES_V1` order is the serving/training contract.
- Leakage fields must not enter scoring path.
- Event parser enforces required fields and type/value constraints.
- Updater only accepts known labels and full feature vectors.
- Trained-model benchmark mode fails fast if model artifacts are missing.

## 11) Observability model

### 11.1 Metrics emitted
- Feedback API counters:
  - `feedback_requests_total`
  - `feedback_published_total`
  - `feedback_publish_errors_total`
- Stream processing counters/gauges:
  - `stream_events_in_total`
  - `stream_events_anomaly_total`
  - `stream_events_normal_total`
  - `stream_events_dlq_total`
  - `stream_process_latency_ms_total`
  - `stream_last_process_latency_ms`
- Online updater metrics:
  - `online_feedback_received_total`
  - `online_feedback_accepted_total`
  - `online_feedback_skipped_total`
  - `online_updates_total`
  - `online_updater_buffer_size`
  - `online_last_update_batch_size`
  - `online_update_count`

### 11.2 Why graphs can look flat
- Counters only change when events are generated.
- `rate(...)` panels need multiple scrapes over a time window.
- If `online_updater` target is down, online metrics stop moving.

## 12) Benchmarks and readiness reports
- `benchmark_report.py`:
  - Synthetic event generation with configurable fraud ratio and alert budget.
  - Reports p50/p95/max latency and precision/recall under alert budget.
- `readiness_check.py`:
  - Asserts practical readiness gates:
  - anomaly path exercised.
  - feedback path exercised.
  - online update path exercised.
  - latency SLO check.
  - quality fields present.

## 13) Challenges encountered and mitigations

### Challenge: API startup brittle when Kafka unavailable
- Symptom: boot-time broker errors (`NoBrokersAvailable`).
- Mitigation:
  - Kafka producer configured to avoid eager broker probing at startup.
  - Health checks can still run before Kafka is up.

### Challenge: Prometheus target up but dashboards still empty
- Symptom: no panel values despite running services.
- Root causes:
  - one-shot traffic not enough for `rate()` windows.
  - service not continuously running.
  - datasource UID mismatch.
- Mitigation:
  - introduced always-on online service on `:8001`.
  - added `seed-online-metrics` task.
  - fixed Grafana datasource UID provisioning (`uid: prometheus`).

### Challenge: online update flush failures with single-class feedback
- Symptom: `partial_fit` errors when update batch contains only one class.
- Mitigation:
  - sample feedback path includes both label classes in one forced update.
  - updater tests enforce guarded behavior and edge cases.

### Challenge: flat observability curves during demos
- Symptom: static charts interpreted as broken system.
- Mitigation:
  - documented need for repeated traffic and shorter time windows.
  - added explicit metric seeding commands and troubleshooting flow.

### Challenge: train/serve skew risk
- Symptom: model behaves differently online vs offline.
- Mitigation:
  - strict `FEATURES_V1` contract reused across training and scoring.
  - parser/feature/scoring tests verify feature presence/order expectations.

## 14) Current limits (important for newcomers)
- Flink integration is scaffold-level, not full production stream job.
- Some services are local-dev-oriented (single-node assumptions).
- Metrics store is in-memory for app processes (no long-term TSDB beyond Prometheus retention).
- Online updater currently supports basic batched update flow, not advanced drift-aware policies.
- Some evaluation/demo components intentionally use synthetic/demo models for speed.

## 15) Extension roadmap ideas
- Add persistent feature store/state store for stream features.
- Implement production-grade message consumers/producers around Kafka topics.
- Add drift detection and model promotion gates.
- Add CI artifacts for benchmark/readiness outputs on every PR.
- Improve model-serving consistency with named-feature DataFrame inference path.

## 16) Practical checklist before saying "it works"
- All required model artifacts exist in `models/`.
- API (`:8000`) and online service (`:8001`) health endpoints return OK.
- Prometheus targets `feedback_api` and `online_updater` are UP.
- Prometheus queries for stream + updater metrics return non-empty.
- Grafana dashboard panels display non-flat recent movement after seeding/traffic.
- `pytest -q` passes.
- `benchmark_report.json` and `demo_readiness_report.json` are generated.

## 17) Command cheat sheet
- Run tests:
  - `scripts/tasks.ps1 -Task test`
- Start API:
  - `scripts/tasks.ps1 -Task start-api`
- Start online service:
  - `scripts/tasks.ps1 -Task start-online-service`
- Seed metrics:
  - `scripts/tasks.ps1 -Task seed-online-metrics`
- Benchmark:
  - `scripts/tasks.ps1 -Task benchmark-report`
  - `scripts/tasks.ps1 -Task benchmark-report-trained`
- Readiness:
  - `scripts/tasks.ps1 -Task demo-readiness`
  - `scripts/tasks.ps1 -Task demo-readiness-trained`

## 18) Final note for beginners
Treat this repository as a guided reference implementation:
- You can run it quickly.
- You can inspect each layer independently.
- You can extend one module at a time with tests.

The fastest way to learn it is:
1. Run `local-demo`.
2. Inspect generated metrics in Prometheus.
3. Open Grafana and connect panels to the corresponding query.
4. Read one module + one related test file at a time.
