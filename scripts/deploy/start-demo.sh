#!/usr/bin/env bash
# Build, verify, and start the demo stack on the deployment host.
#
# Safe to re-run. Intended to be the single command after an instance start:
# it refreshes the public address, preflights the models, and brings the stack up.
#
# The preflight model check is the important part. Model artifacts are joblib
# pickles that cross from training into this image, and they do not survive a
# numpy major-version boundary. A mismatch surfaces as a job that dies on
# startup rather than as a build failure, so catch it here, before an evaluator
# is given the URL. See docs/aws_demo_deployment.md for the version contract.
#
# Set SKIP_URL_REFRESH=1 to leave DEMO_BASE_URL untouched, for example when the
# instance has an Elastic IP or is reached through a domain name.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Only flink-job carries the streaming profile, so the base array brings up
# everything else without submitting a job. Submission is guarded separately,
# because each recreation of that one-shot container submits another copy and
# duplicates starve each other of task slots.
COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.demo.yml)
COMPOSE_JOB=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.demo.yml --profile streaming)
FLINK_JOB_NAME="realtime-fraud-scoring"
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

# A stopped instance is assigned a new public IP on start. DEMO_BASE_URL feeds
# Grafana's root_url and Prometheus's external-url, and a stale value breaks
# Grafana's login redirect. Refresh it before Compose reads the file, and leave
# every credential in place so anything already given to an evaluator still works.
if [[ "${SKIP_URL_REFRESH:-0}" == "1" ]]; then
  echo "==> Skipping public address refresh (SKIP_URL_REFRESH=1)"
else
  echo "==> Checking the public address"
  # stderr is suppressed because the only expected failure here is "not on EC2",
  # and its message tells the reader to pass --base-url, which is not the advice
  # that applies in this context.
  if ! scripts/deploy/make-secrets.sh --base-url-only 2>/dev/null; then
    echo "    No EC2 public IP detected. Leaving DEMO_BASE_URL unchanged."
  fi
fi

echo "==> Building application and Flink images"
"${COMPOSE[@]}" build

echo "==> Preflight: loading model artifacts inside the Flink image"
# MSYS_NO_PATHCONV stops Git Bash on Windows from rewriting the -v argument,
# which otherwise mounts an empty directory and makes this look like a model
# problem. It is an unused variable on Linux.
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "$REPO_ROOT/models:/opt/fraud/models:ro" \
  --entrypoint python3 \
  realtime-fraud/flink:local -c '
import sys, warnings, joblib, numpy, sklearn
# InconsistentVersionWarning means the scores may be silently wrong, which for a
# fraud demo is worse than a crash. Treat it as a failure, not a note.
warnings.simplefilter("error", UserWarning)
print("  flink image: numpy", numpy.__version__, "| scikit-learn", sklearn.__version__)
missing, unloadable = [], []
for name in ("isolation_forest_v1", "autoencoder_v1", "sgd_classifier_v1"):
    path = f"/opt/fraud/models/{name}.joblib"
    try:
        joblib.load(path)
        print("  OK     ", name)
    except FileNotFoundError:
        missing.append(name)
        print("  MISSING", name)
    except Exception as exc:
        unloadable.append(name)
        print("  FAILED ", name, type(exc).__name__, exc)
# The caller already verified these exist on the host, so absence inside the
# container means the bind mount did not take, not that training is needed.
if missing:
    print("\nNot visible inside the container:", ", ".join(missing))
    print("They exist on the host, so the bind mount of ./models failed.")
    print("Check Docker file sharing for this path rather than retraining.")
if unloadable:
    print("\nThe Flink image cannot load:", ", ".join(unloadable))
    print("Artifacts must be trained with the same numpy and scikit-learn this image")
    print("installs. Retrain with: uv run python -m src.models.train_{if,ae,sgd}")
    print("See docs/aws_demo_deployment.md, section \"The numpy ceiling\".")
sys.exit(1 if (missing or unloadable) else 0)
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

# Submit the scoring job only when one is not already on the cluster. Re-running
# this script otherwise stacks duplicate jobs, and with a fixed number of task
# slots the extras sit in RESTARTING forever.
echo "==> Checking for an existing scoring job"
running_jobs="$(curl -fsS --max-time 10 http://localhost:8081/jobs/overview 2>/dev/null \
  | grep -o "\"name\":\"$FLINK_JOB_NAME\",\"state\":\"RUNNING\"" | wc -l || echo 0)"

if [[ "${running_jobs:-0}" -gt 0 ]]; then
  echo "    '$FLINK_JOB_NAME' is already RUNNING. Not submitting another."
else
  echo "==> Submitting the scoring job"
  "${COMPOSE_JOB[@]}" up -d --force-recreate flink-job
  sleep 20
  # A non-zero exit means the job never reached the cluster, which leaves the
  # dashboard empty while every other service looks healthy.
  if "${COMPOSE_JOB[@]}" ps --all --format '{{.Service}} {{.State}} {{.ExitCode}}' \
      | grep -qE '^flink-job exited [^0]'; then
    echo
    echo "WARNING: flink-job exited non-zero. The scoring job is not running."
    echo "Inspect with: ${COMPOSE_JOB[*]} logs flink-job"
  fi
fi

echo "==> Service status"
"${COMPOSE_JOB[@]}" ps

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
