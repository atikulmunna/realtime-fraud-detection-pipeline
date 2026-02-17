param(
  [string]$CondaEnv = "realtime-fraud",
  [string]$ModelPath = "models/sgd_classifier_v1.joblib",
  [int]$BatchSize = 500,
  [string]$FeedbackFile = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".").Path

$args = @(
  "-m", "src.online.updater_service",
  "--model-path", $ModelPath,
  "--batch-size", "$BatchSize"
)

if ($FeedbackFile -ne "") {
  $args += @("--feedback-file", $FeedbackFile)
}

conda run -n $CondaEnv python @args
