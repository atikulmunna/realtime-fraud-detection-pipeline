param(
  [string]$CondaEnv = "realtime-fraud",
  [string]$ModelPath = "models/sgd_classifier_v1.joblib",
  [int]$BatchSize = 500,
  [string]$BootstrapServers = "localhost:9092",
  [string]$Topic = "feedback",
  [string]$GroupId = "fraud-online-updater"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".").Path

$args = @(
  "-m", "src.online.feedback_consumer_service",
  "--model-path", $ModelPath,
  "--batch-size", "$BatchSize",
  "--bootstrap-servers", $BootstrapServers,
  "--topic", $Topic,
  "--group-id", $GroupId
)

conda run -n $CondaEnv python @args

