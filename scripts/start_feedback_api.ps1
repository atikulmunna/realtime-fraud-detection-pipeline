param(
  [string]$CondaEnv = "realtime-fraud",
  [string]$BindHost = "0.0.0.0",
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".").Path

conda run -n $CondaEnv uvicorn src.api.app_entry:app_factory --factory --host $BindHost --port $Port
