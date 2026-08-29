#!/usr/bin/env bash
# Arm an automatic power-off so a forgotten demo cannot bill indefinitely.
#
# A stopped instance costs only its EBS volume, so the expensive failure mode is
# leaving the box running after an evaluation. This schedules an OS-level
# shutdown, which for an EBS-backed instance stops it and ends compute billing.
#
# SAFETY: the instance's shutdown behaviour must be "stop", which is the EC2
# default. If it is set to "terminate", this DESTROYS the instance and its
# volume. Verify before relying on this:
#
#   aws ec2 describe-instance-attribute --instance-id <id> \
#     --attribute instanceInitiatedShutdownBehavior
#
# The timer is re-armed on every start-demo.sh run, so an active demo keeps
# extending it rather than accumulating overlapping shutdowns.
#
# Usage:
#   scripts/deploy/arm-autostop.sh [HOURS]    # default 4
#   scripts/deploy/arm-autostop.sh --cancel   # run indefinitely
#   scripts/deploy/arm-autostop.sh --status

set -euo pipefail

SHUTDOWN_FLAG=/run/systemd/shutdown/scheduled

show_status() {
  if [[ -f "$SHUTDOWN_FLAG" ]]; then
    local usec
    usec="$(grep -oP '(?<=^USEC=)\d+' "$SHUTDOWN_FLAG" 2>/dev/null || true)"
    if [[ -n "$usec" ]]; then
      echo "Auto-stop armed for: $(date -d "@$((usec / 1000000))" '+%Y-%m-%d %H:%M:%S %Z')"
      return
    fi
    echo "Auto-stop is armed."
    return
  fi
  echo "No auto-stop scheduled. This instance will run until stopped manually."
}

case "${1:-}" in
  --status)
    show_status
    exit 0
    ;;
  --cancel)
    sudo shutdown -c 2>/dev/null || true
    echo "Auto-stop cancelled. Remember to stop the instance yourself:"
    echo "  aws ec2 stop-instances --instance-ids <id>"
    exit 0
    ;;
esac

HOURS="${1:-4}"
if ! [[ "$HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$(echo "$HOURS <= 0" | bc -l 2>/dev/null || echo 1)" == "1" ]]; then
  echo "ERROR: hours must be a positive number, got '$HOURS'." >&2
  exit 2
fi

MINUTES="$(printf '%.0f' "$(echo "$HOURS * 60" | bc -l)")"
[[ "$MINUTES" -ge 1 ]] || MINUTES=1

# Clear any existing schedule first, so re-running extends rather than conflicts.
sudo shutdown -c 2>/dev/null || true
sudo shutdown -h "+$MINUTES" "Demo auto-stop: this instance powers off in $HOURS hour(s). Cancel with: sudo shutdown -c" >/dev/null 2>&1

echo "Auto-stop armed: powering off in $HOURS hour(s) (~$MINUTES minutes)."
show_status
cat <<EOF

  Extend      scripts/deploy/arm-autostop.sh 8
  Cancel      scripts/deploy/arm-autostop.sh --cancel
  Check       scripts/deploy/arm-autostop.sh --status

Powering off stops the instance and ends compute billing. The EBS volume
(about \$2.40/month) remains so the demo can be restarted.
EOF
