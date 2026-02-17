param(
  [string]$CondaEnv = "realtime-fraud",
  [string]$BaseUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".").Path

conda run -n $CondaEnv python -m src.observability.healthcheck --base-url $BaseUrl
