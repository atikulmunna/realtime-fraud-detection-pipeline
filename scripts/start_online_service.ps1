param(
  [string]$CondaEnv = "realtime-fraud",
  [string]$BindHost = "0.0.0.0",
  [int]$Port = 8001
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".").Path

conda run -n $CondaEnv uvicorn src.online.updater_service:app_factory --factory --host $BindHost --port $Port

