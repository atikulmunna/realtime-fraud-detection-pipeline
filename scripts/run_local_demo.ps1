param(
  [switch]$WithCompose,
  [string]$CondaEnv = "realtime-fraud"
)

$ErrorActionPreference = "Stop"

if ($WithCompose) {
  docker compose -f infra/docker-compose.yml up -d
}

$env:PYTHONPATH = (Resolve-Path ".").Path
conda run -n $CondaEnv python scripts/demo_run.py
