#!/usr/bin/env bash
# Build, verify, and start the demo stack on the deployment host.
#
# The preflight model check is the important part: the Flink image pins a
# different scikit-learn than the one the artifacts were trained with, and a
# version mismatch surfaces as a job that dies on startup rather than as a
# build failure. Catch it here, before an evaluator is given the URL.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.demo.yml --profile streaming)
REQUIRED_MODELS=(
  models/isolation_forest_v1.joblib
  models/autoencoder_v1.joblib
  models/sgd_classifier_v1.joblib
)

fail() { echo "ERROR: $*" >&2; exit 1; }

[[ -f infra/.env ]] || fail "infra/.env is missing. Run scripts/deploy/make-secrets.sh first."
[[ -f infra/secrets/feedback_api_key.txt ]] || fail "infra/secrets/feedback_api_key.txt is missing. Run scripts/deploy/make-secrets.sh."
[[ -f infra/caddy/auth.conf ]] || fail "infra/caddy/auth.conf is missing. Run scripts/deploy/make-secrets.sh."

for model in "${REQUIRED_MODELS[@]}"; do
  [[ -f "$model" ]] || fail "$model is missing. Upload the trained artifacts before starting."
done

echo "==> Building application and Flink images"
"${COMPOSE[@]}" build

echo "==> Preflight: loading model artifacts inside the Flink image"
docker run --rm \
  -v "$REPO_ROOT/models:/opt/fraud/models:ro" \
  --entrypoint python3 \
  realtime-fraud/flink:local -c '
import sys, warnings, joblib, sklearn
warnings.simplefilter("error", UserWarning)
print("flink image scikit-learn:", sklearn.__version__)
failed = []
for name in ("isolation_forest_v1", "autoencoder_v1", "sgd_classifier_v1"):
    path = f"/opt/fraud/models/{name}.joblib"
    try:
        joblib.load(path)
        print("  OK     ", name)
    except Exception as exc:
        failed.append(name)
        print("  FAILED ", name, type(exc).__name__, exc)
if failed:
    print("\nThe Flink image cannot load:", ", ".join(failed))
    print("Align infra/flink/requirements.txt with the training environment, or retrain.")
    sys.exit(1)
' || fail "Model preflight failed. The streaming job would crash on startup."

echo "==> Starting the stack"
# --wait is not used here: topic-init and flink-job are one-shot services that
# exit 0 by design, and --wait treats an exited container as a failed start.
"${COMPOSE[@]}" up -d

echo "==> Waiting for the public entry point"
deadline=$(( SECONDS + 300 ))
until curl -fsS -o /dev/null http://localhost/ 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    "${COMPOSE[@]}" ps
    fail "Timed out waiting for the proxy. Inspect logs with: ${COMPOSE[*]} logs"
  fi
  sleep 5
done

echo "==> Waiting for Grafana"
deadline=$(( SECONDS + 180 ))
until curl -fsS -o /dev/null http://localhost/grafana/api/health 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    fail "Grafana did not become ready. Check: ${COMPOSE[*]} logs grafana"
  fi
  sleep 5
done

echo "==> Service status"
"${COMPOSE[@]}" ps

# The streaming job is submitted by a one-shot container. A non-zero exit means
# the job never reached the cluster, which leaves the dashboard empty.
if "${COMPOSE[@]}" ps --all --format '{{.Service}} {{.State}} {{.ExitCode}}' \
    | grep -qE '^flink-job exited [^0]'; then
  echo
  echo "WARNING: flink-job exited non-zero. The scoring job is not running."
  echo "Inspect with: ${COMPOSE[*]} logs flink-job"
fi

BASE_URL="$(grep -E '^DEMO_BASE_URL=' infra/.env | cut -d= -f2-)"
cat <<EOF

Demo is up.

  Landing page   $BASE_URL/
  Grafana        $BASE_URL/grafana/
  Feedback API   $BASE_URL/api/docs
  Prometheus     $BASE_URL/prometheus/graph
  MLflow         $BASE_URL/mlflow/
  Flink REST     $BASE_URL/flink/overview

Full Flink console over an SSH tunnel:
  ssh -i <key.pem> -L 8081:localhost:8081 ubuntu@<public-ip>   then open http://localhost:8081

Stop the stack without losing data:  scripts/deploy/stop-demo.sh
EOF
