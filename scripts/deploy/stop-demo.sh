#!/usr/bin/env bash
# Stop the demo stack. Named volumes are preserved so Kafka offsets, Postgres
# feedback, MLflow versions, Flink checkpoints, and Grafana state survive.
#
# Pass --destroy to remove volumes as well. That is not reversible.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

COMPOSE=(docker compose -f infra/docker-compose.yml -f infra/docker-compose.demo.yml --profile streaming)

if [[ "${1:-}" == "--destroy" ]]; then
  read -r -p "Remove all demo volumes and durable state? Type 'destroy' to confirm: " reply
  [[ "$reply" == "destroy" ]] || { echo "Aborted."; exit 1; }
  "${COMPOSE[@]}" down --volumes
  echo "Stack and volumes removed."
else
  "${COMPOSE[@]}" stop
  echo "Stack stopped. Volumes preserved. Restart with scripts/deploy/start-demo.sh."
  echo "Remember to stop the EC2 instance too, or it keeps billing."
fi
