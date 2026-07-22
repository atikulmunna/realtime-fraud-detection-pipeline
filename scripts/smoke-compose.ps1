param(
  [int]$TimeoutSeconds = 180
)

$ErrorActionPreference = 'Stop'
$composeFile = 'infra/docker-compose.yml'
$apiKeyPath = 'infra/secrets/feedback_api_key.txt.example'

uv run python -m src.demo.seed_local_model --output models/sgd_classifier_v1.joblib
docker compose -f $composeFile up -d --build

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
do {
  try {
    $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2
    if ($health.status -eq 'ok') { break }
  } catch {}
  Start-Sleep -Seconds 2
} while ((Get-Date) -lt $deadline)

if ((Get-Date) -ge $deadline) {
  docker compose -f $composeFile ps
  throw 'Feedback API did not become healthy before the timeout.'
}

$apiKey = (Get-Content -LiteralPath $apiKeyPath -Raw).Trim()
$runId = [guid]::NewGuid().ToString('N')
$features = @{
  amount = 900.0
  amount_ratio = 0.9
  balance_diff_orig = 900.0
  is_transfer = 1.0
  is_cashout = 0.0
  hour_of_day = 8.0
  txn_velocity_1h = 1.0
}
$body = @{
  feedback_id = "compose-smoke-feedback-$runId"
  anomaly_id = "compose-smoke-anomaly-$runId"
  label = 'true_positive'
  analyst_id = 'compose-smoke'
  features = $features
} | ConvertTo-Json -Depth 4

$response = Invoke-RestMethod `
  -Method Post `
  -Uri 'http://127.0.0.1:8000/feedback' `
  -Headers @{'X-API-Key' = $apiKey; 'Idempotency-Key' = "compose-smoke-request-$runId"} `
  -ContentType 'application/json' `
  -Body $body

if ($response.status -ne 'accepted') { throw 'Feedback was not accepted.' }

do {
  try {
    $relayMetrics = Invoke-WebRequest -Uri 'http://127.0.0.1:8003/metrics' -UseBasicParsing -TimeoutSec 2
    if ($relayMetrics.Content -match 'outbox_published_total 1') { break }
  } catch {}
  Start-Sleep -Seconds 2
} while ((Get-Date) -lt $deadline)

if ((Get-Date) -ge $deadline) { throw 'Durable feedback was not relayed before the timeout.' }

do {
  try {
    $updaterMetrics = Invoke-WebRequest -Uri 'http://127.0.0.1:8002/metrics' -UseBasicParsing -TimeoutSec 2
    if ($updaterMetrics.Content -match 'online_updates_total [1-9]') { break }
  } catch {}
  Start-Sleep -Seconds 2
} while ((Get-Date) -lt $deadline)

if ((Get-Date) -ge $deadline) { throw 'The updater did not register and promote a candidate before the timeout.' }

Write-Output "Compose smoke passed: feedback_id=$($response.feedback_id)"
