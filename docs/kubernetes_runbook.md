# Kubernetes Deployment and Rollback

The Kustomize base deploys only stateless application workloads, a small model cache PVC, and a `FlinkDeployment`. Production Kafka, Postgres, object storage, MLflow, and secret management remain external dependencies. Install the Flink Kubernetes Operator and its CRDs before applying this repository.

## Development render

```bash
kubectl kustomize deploy/kubernetes/overlays/development
kubectl apply -k deploy/kubernetes/overlays/development
```

Replace every `replace-me` value first. In production, omit the development `secretGenerator`; create `fraud-secrets` through the platform secret manager. Replace the checkpoint/savepoint bucket, external endpoints, image registry, tags, and resource values. Bind `fraud-model-cache` to an external RWX or object-backed CSI storage class; it is a cache and not the authoritative registry.

## Rollout checks

```bash
kubectl -n realtime-fraud rollout status deployment/feedback-api
kubectl -n realtime-fraud rollout status deployment/outbox-relay
kubectl -n realtime-fraud get flinkdeployment fraud-streaming
kubectl -n realtime-fraud get hpa,pdb,networkpolicy
```

The updater intentionally uses `Recreate` and one replica because its local candidate file is single-writer state. The relay also stays single-replica until database row leasing or `SKIP LOCKED` claiming is implemented. Kafka offsets and MLflow aliases remain the durable coordination points.

## Rollback

Roll back application Deployments with `kubectl rollout undo`. Roll back a model by moving the MLflow `champion` alias to the last verified version; do not replace files inside running pods. For Flink code changes, set `spec.job.state: suspended`, confirm a savepoint, deploy the prior image digest, and restore with `upgradeMode: savepoint`. Never delete checkpoint or savepoint objects during incident recovery.
