# Architecture

## Development topology

The locked Python 3.11 environment runs data preparation, training, evaluation, API, and reliability tests. Each `MetricsRegistry` owns an isolated real Prometheus collector registry, making tests deterministic without replacing production metric semantics. Model and data artifacts remain outside Git.

## Compose topology

The Compose stack separates the public feedback API from the internal backplane. The API writes feedback and an outbox row in one Postgres transaction. The relay publishes with Kafka idempotence and acknowledgements. The updater disables auto-commit and commits only after a candidate is durably promoted or rejected. MLflow/Postgres owns immutable registry versions and aliases. Prometheus scrapes service endpoints and Grafana provisions the checked-in dashboard.

Kafka has internal and host listeners, topic auto-creation is disabled, and `topic-init` creates the contract topics. Named volumes preserve Kafka, Postgres, MLflow, Flink checkpoints, Prometheus, and Grafana state. The PyFlink job is enabled with the `streaming` profile after trained IF, AE, and SGD artifacts exist.

## Kubernetes topology

The Kustomize base deploys the API, relay, updater, and a Flink Operator `FlinkDeployment`. It does not deploy production Kafka, Postgres, MLflow/object storage, ingress, or a secrets manager. Those dependencies are supplied through ConfigMaps, external secrets, and an external RWX/object-backed model cache. Checkpoints and savepoints use external object storage.

The API and relay may scale horizontally. The updater remains a single writer. Flink parallelism is managed by the operator and uses savepoint upgrades. Default-deny network policy, restricted pod security, resources, probes, disruption budgets, and autoscaling form the deployment baseline.

## Model lifecycle

Offline training uses chronological train/promotion/test partitions. Calibration and threshold selection use only promotion validation data. Immutable bundles record the feature contract, dataset hash, Git revision, thresholds, metrics, and checksum. MLflow assigns `candidate` and `champion` aliases.

Online feedback stages a separate SGD candidate. It is evaluated before local atomic promotion and MLflow alias movement. Only then is a checksum-bearing model-update event published. Flink accepts only promoted filenames under its model directory with the exact feature order and SHA-256 checksum.
