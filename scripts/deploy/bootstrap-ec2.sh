#!/usr/bin/env bash
# EC2 user-data script for the demo host. Targets Ubuntu 24.04 LTS.
#
# Installs Docker, provisions swap, clones the repository, and pre-pulls images.
# It deliberately stops short of starting the stack: the Flink job requires the
# trained model artifacts, which are gitignored and must be uploaded separately.
#
# Paste into the EC2 "User data" field at launch, or run manually:
#   sudo bash scripts/deploy/bootstrap-ec2.sh

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/atikulmunna/realtime-fraud-detection-pipeline.git}"
TARGET_USER="${TARGET_USER:-ubuntu}"
TARGET_DIR="/home/$TARGET_USER/realtime-fraud-detection-pipeline"

log() { echo "[bootstrap] $*"; }

log "Installing base packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y ca-certificates curl git openssl

log "Installing Docker Engine and the Compose plugin"
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
cat > /etc/apt/sources.list.d/docker.list <<EOF
deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable
EOF
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
usermod -aG docker "$TARGET_USER"
systemctl enable --now docker

# The stack budgets about 7 GB of an 8 GB host. Swap absorbs JVM startup spikes
# rather than letting the kernel OOM-kill Kafka or a Flink TaskManager mid-demo.
if [[ ! -f /swapfile ]]; then
  log "Creating 2 GB swap"
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi

# Docker's default json-file logs are unbounded and will fill a 30 GB root volume
# during a long-running demo.
log "Capping container log size"
cat > /etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": { "max-size": "10m", "max-file": "3" }
}
EOF
systemctl restart docker

if [[ ! -d "$TARGET_DIR" ]]; then
  log "Cloning $REPO_URL"
  sudo -u "$TARGET_USER" git clone "$REPO_URL" "$TARGET_DIR"
fi

log "Pre-pulling third-party images"
cd "$TARGET_DIR"
for image in confluentinc/cp-zookeeper:7.6.1 confluentinc/cp-kafka:7.6.1 postgres:16 \
             prom/prometheus:v2.54.1 grafana/grafana:11.1.5 caddy:2.8-alpine \
             flink:1.19.1-scala_2.12-java17 python:3.11-slim; do
  docker pull "$image" || log "WARNING: failed to pull $image"
done

chown -R "$TARGET_USER:$TARGET_USER" "$TARGET_DIR"

cat <<EOF

[bootstrap] Host is ready. Remaining steps, from your workstation and then this host:

  1. Upload the trained artifacts (they are gitignored, so the clone has none):
       scp -i <key.pem> models/isolation_forest_v1.joblib \\
                        models/autoencoder_v1.joblib \\
                        models/sgd_classifier_v1.joblib \\
                        $TARGET_USER@<public-ip>:$TARGET_DIR/models/

  2. On this host, generate credentials and start the stack:
       cd $TARGET_DIR
       scripts/deploy/make-secrets.sh
       scripts/deploy/start-demo.sh

EOF
