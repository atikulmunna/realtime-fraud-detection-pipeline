param(
  [string]$CondaEnv = "realtime-fraud",
  [string]$ModelPath = "models/sgd_classifier_v1.joblib",
  [int]$BatchSize = 500,
  [string]$BootstrapServers = "localhost:9092",
  [string]$Topic = "feedback",
  [string]$GroupId = "fraud-online-updater",
  [string]$PromotionHoldout = "",
  [double]$MinPrecision = 0.0,
  [double]$MinRecall = 0.0,
  [double]$MinPrAuc = 0.0
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path ".").Path

$args = @(
  "-m", "src.online.feedback_consumer_service",
  "--model-path", $ModelPath,
  "--batch-size", "$BatchSize",
  "--bootstrap-servers", $BootstrapServers,
  "--topic", $Topic,
  "--group-id", $GroupId,
  "--min-precision", "$MinPrecision",
  "--min-recall", "$MinRecall",
  "--min-pr-auc", "$MinPrAuc"
)

if ($PromotionHoldout -ne "") {
  $args += @("--promotion-holdout", $PromotionHoldout)
}

conda run -n $CondaEnv python @args
