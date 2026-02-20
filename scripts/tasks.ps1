param(
  [Parameter(Mandatory=$true)]
  [ValidateSet('init','test','smoke','download-data','train-all','local-demo','local-demo-compose','start-api','start-online-service','start-feedback-consumer','seed-online-metrics','start-updater','start-updater-sample','healthcheck-api','benchmark-report','benchmark-report-trained','demo-readiness','demo-readiness-trained')]
  [string]$Task
)

switch ($Task) {
  'init' {
    if (-not (Test-Path .venv)) { python -m venv .venv }
    .\.venv\Scripts\python -m pip install -r requirements.txt
  }
  'test' {
    pytest -q
  }
  'smoke' {
    pytest -q tests/unit/test_feature_contract.py tests/unit/test_event_schema.py
  }
  'download-data' {
    .\.venv\Scripts\python -m src.data.download_paysim --out data/raw
  }
  'train-all' {
    $csv = Get-ChildItem -Path "data/raw" -Filter "*.csv" -File -ErrorAction SilentlyContinue |
      Sort-Object Length -Descending |
      Select-Object -First 1
    if (-not $csv) {
      throw "No CSV found under data/raw. Run 'scripts/tasks.ps1 -Task download-data' first."
    }

    python -m src.data.prepare_paysim --input $csv.FullName --json
    python -m src.data.feature_engineering --input $csv.FullName --output data/processed/paysim_features.parquet
    python -m src.models.train_if --input data/processed/paysim_features.parquet
    python -m src.models.train_ae --input data/processed/paysim_features.parquet
    python -m src.models.train_sgd --input data/processed/paysim_features.parquet
  }
  'local-demo' {
    .\scripts\run_local_demo.ps1
  }
  'local-demo-compose' {
    .\scripts\run_local_demo.ps1 -WithCompose
  }
  'start-api' {
    .\scripts\start_feedback_api.ps1
  }
  'start-online-service' {
    .\scripts\start_online_service.ps1
  }
  'start-feedback-consumer' {
    .\scripts\start_feedback_consumer.ps1
  }
  'seed-online-metrics' {
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/stream/sample" -ErrorAction Stop | Out-Null
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8001/feedback/sample" -ErrorAction Stop | Out-Null
    Write-Output "Seeded online service metrics."
  }
  'start-updater' {
    .\scripts\start_online_updater.ps1
  }
  'start-updater-sample' {
    .\scripts\start_online_updater.ps1 -FeedbackFile data/samples/feedback.sample.jsonl -BatchSize 2
  }
  'healthcheck-api' {
    .\scripts\healthcheck_api.ps1
  }
  'benchmark-report' {
    python -m src.evaluation.benchmark_report --output reports/benchmark_report.json
  }
  'benchmark-report-trained' {
    python -m src.evaluation.benchmark_report --use-trained-models --output reports/benchmark_report.json
  }
  'demo-readiness' {
    python -m src.demo.readiness_check --output reports/demo_readiness_report.json
  }
  'demo-readiness-trained' {
    python -m src.demo.readiness_check --use-trained-models --output reports/demo_readiness_report.json
  }
}
