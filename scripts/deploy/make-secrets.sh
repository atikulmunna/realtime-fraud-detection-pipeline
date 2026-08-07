#!/usr/bin/env bash
# Generate or refresh the per-deployment configuration the demo overlay expects.
#
#   infra/.env                           Compose variables, including DEMO_BASE_URL
#   infra/secrets/feedback_api_key.txt   API key for the feedback service
#   infra/caddy/auth.conf                bcrypt basic_auth block for the proxy
#
# Modes:
#   (default)         Generate everything. Refuses to overwrite an existing setup.
#   --force           Regenerate credentials, reusing the existing DB password.
#   --base-url-only   Rewrite DEMO_BASE_URL and leave all credentials alone.
#
# --base-url-only exists because a stopped EC2 instance gets a new public IP on
# every start. The address has to be refreshed, but rotating the Grafana password
# and API key at the same time would invalidate credentials already handed to an
# evaluator.
#
# --force reuses POSTGRES_PASSWORD on purpose. Postgres applies that value only
# when it initialises an empty data directory, so once the stack has run, the
# postgres_data volume still holds the old password and rotating it breaks the
# API and MLflow. Pass --rotate-db-password to override, and recreate the volume.
#
# Usage:
#   scripts/deploy/make-secrets.sh
#   scripts/deploy/make-secrets.sh --force
#   scripts/deploy/make-secrets.sh --base-url-only
#   scripts/deploy/make-secrets.sh --base-url-only --base-url http://203.0.113.10

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FORCE=0
BASE_URL_ONLY=0
ROTATE_DB_PASSWORD=0
BASE_URL=""

usage() {
  sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --base-url-only) BASE_URL_ONLY=1; shift ;;
    --rotate-db-password) ROTATE_DB_PASSWORD=1; shift ;;
    --base-url) BASE_URL="${2:-}"; [[ -n "$BASE_URL" ]] || { echo "--base-url needs a value" >&2; exit 2; }; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown argument: $1" >&2; usage 2 ;;
  esac
done

if [[ $FORCE -eq 1 && $BASE_URL_ONLY -eq 1 ]]; then
  echo "--force and --base-url-only are mutually exclusive." >&2
  exit 2
fi

ENV_FILE="$REPO_ROOT/infra/.env"
API_KEY_FILE="$REPO_ROOT/infra/secrets/feedback_api_key.txt"
AUTH_FILE="$REPO_ROOT/infra/caddy/auth.conf"

random_secret() {
  # 32 URL-safe characters. Stripping +/= keeps the value safe to paste into a
  # shell, a .env line, and an HTTP header without quoting surprises.
  openssl rand -base64 48 | tr -d '\n+/=' | cut -c1-32
}

detect_base_url() {
  # IMDSv2 requires a token; fall back cleanly when running outside EC2.
  local token public_ip
  token="$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" --connect-timeout 2 2>/dev/null || true)"
  if [[ -n "$token" ]]; then
    public_ip="$(curl -fsS -H "X-aws-ec2-metadata-token: $token" \
      "http://169.254.169.254/latest/meta-data/public-ipv4" --connect-timeout 2 2>/dev/null || true)"
    if [[ -n "$public_ip" ]]; then
      echo "http://$public_ip"
      return
    fi
  fi
  echo ""
}

resolve_base_url() {
  if [[ -z "$BASE_URL" ]]; then
    BASE_URL="$(detect_base_url)"
  fi
  if [[ -z "$BASE_URL" ]]; then
    echo "Could not determine the public address. Pass --base-url http://<host>." >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# Address-only refresh
# ---------------------------------------------------------------------------

if [[ $BASE_URL_ONLY -eq 1 ]]; then
  [[ -f "$ENV_FILE" ]] || { echo "infra/.env does not exist. Run without --base-url-only first." >&2; exit 1; }
  resolve_base_url

  current="$(grep -E '^DEMO_BASE_URL=' "$ENV_FILE" | cut -d= -f2- || true)"
  if [[ "$current" == "$BASE_URL" ]]; then
    echo "DEMO_BASE_URL is already $BASE_URL. Nothing to do."
    exit 0
  fi

  # '|' as the delimiter because the replacement is a URL containing '/'.
  if grep -qE '^DEMO_BASE_URL=' "$ENV_FILE"; then
    sed -i "s|^DEMO_BASE_URL=.*|DEMO_BASE_URL=$BASE_URL|" "$ENV_FILE"
  else
    echo "DEMO_BASE_URL=$BASE_URL" >> "$ENV_FILE"
  fi
  chmod 600 "$ENV_FILE"

  echo "DEMO_BASE_URL updated."
  echo "  was  ${current:-<unset>}"
  echo "  now  $BASE_URL"
  echo
  echo "Credentials are unchanged. Restart the affected services to pick it up:"
  echo "  scripts/deploy/start-demo.sh"
  exit 0
fi

# ---------------------------------------------------------------------------
# Full generation
# ---------------------------------------------------------------------------

if [[ $FORCE -eq 0 && -f "$ENV_FILE" ]]; then
  cat >&2 <<'EOF'
infra/.env already exists.

  To refresh only the public address after a restart, keeping every credential:
    scripts/deploy/make-secrets.sh --base-url-only

  To rotate every credential and start over:
    scripts/deploy/make-secrets.sh --force
EOF
  exit 1
fi

resolve_base_url

# Postgres writes POSTGRES_PASSWORD only when it initialises an empty data
# directory. Once the stack has run, postgres_data still holds the original, so
# rotating this value here would leave the API and MLflow failing with an opaque
# "password authentication failed for user fraud". Carry the existing one forward.
POSTGRES_PASSWORD=""
if [[ -f "$ENV_FILE" && $ROTATE_DB_PASSWORD -eq 0 ]]; then
  POSTGRES_PASSWORD="$(grep -E '^POSTGRES_PASSWORD=' "$ENV_FILE" | cut -d= -f2- || true)"
  if [[ -n "$POSTGRES_PASSWORD" ]]; then
    echo "Reusing the existing POSTGRES_PASSWORD; the database volume is initialised with it."
    echo "Pass --rotate-db-password to change it, then recreate the postgres_data volume."
  fi
fi
if [[ -z "$POSTGRES_PASSWORD" ]]; then
  POSTGRES_PASSWORD="$(random_secret)"
  if [[ $ROTATE_DB_PASSWORD -eq 1 ]]; then
    echo "WARNING: rotating POSTGRES_PASSWORD. An existing postgres_data volume will"
    echo "         reject it. Recreate it first, which discards stored feedback:"
    echo "           docker compose -f infra/docker-compose.yml -f infra/docker-compose.demo.yml down"
    echo "           docker volume rm realtime-fraud_postgres_data"
  fi
fi

GRAFANA_ADMIN_PASSWORD="$(random_secret)"
FEEDBACK_API_KEY="$(random_secret)"
DEMO_USER="evaluator"
DEMO_PASSWORD="$(random_secret)"

mkdir -p "$REPO_ROOT/infra/secrets" "$REPO_ROOT/infra/caddy"

# caddy hash-password runs in the same image the proxy uses, so no local Caddy is needed.
DEMO_PASSWORD_HASH="$(docker run --rm caddy:2.8-alpine \
  caddy hash-password --plaintext "$DEMO_PASSWORD")"

cat > "$ENV_FILE" <<EOF
# Generated by scripts/deploy/make-secrets.sh. Never commit this file.
APP_ENV=production
DEMO_BASE_URL=$BASE_URL
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=$GRAFANA_ADMIN_PASSWORD
DEMO_TRAFFIC_RATE=4
DEMO_FRAUD_RATIO=0.04
DEMO_USERS=1000000
EOF

printf '%s' "$FEEDBACK_API_KEY" > "$API_KEY_FILE"

cat > "$AUTH_FILE" <<EOF
basic_auth {
	$DEMO_USER $DEMO_PASSWORD_HASH
}
EOF

chmod 600 "$ENV_FILE" "$AUTH_FILE"

# The API key is the one secret consumed by a container that drops privileges:
# the app image runs as uid 999 (fraud), while this file is owned by the host
# user. Compose bind-mounts file-based secrets with the host's ownership and
# mode verbatim, and outside swarm it ignores the uid/gid/mode fields, so 0600
# makes the file unreadable inside the container and the API dies at startup
# with PermissionError on /run/secrets/feedback_api_key.
#
# Docker Desktop on Windows does not preserve Linux ownership on bind mounts, so
# this only manifests on a real Linux host. 0644 is acceptable for a
# single-tenant, time-boxed demo instance; a production deployment should take
# the key from a secret manager instead. See docs/aws_demo_deployment.md.
chmod 644 "$API_KEY_FILE"

cat <<EOF

Credentials generated. Record these now; the plaintext is not stored anywhere.

  Demo URL          $BASE_URL
  Grafana           admin / $GRAFANA_ADMIN_PASSWORD
  Proxy basic auth  $DEMO_USER / $DEMO_PASSWORD
  Feedback API key  $FEEDBACK_API_KEY

Prometheus, MLflow, and the Flink REST path sit behind the proxy basic auth.
Grafana keeps its own separate login.

After a stop and start the public IP changes. Refresh it without rotating any
of the above:
  scripts/deploy/make-secrets.sh --base-url-only
EOF
