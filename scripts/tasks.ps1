param(
  [Parameter(Mandatory=$true)]
  [ValidateSet('init','test','smoke','download-data','local-demo','local-demo-compose','start-api','start-updater','start-updater-sample','healthcheck-api')]
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
  'local-demo' {
    .\scripts\run_local_demo.ps1
  }
  'local-demo-compose' {
    .\scripts\run_local_demo.ps1 -WithCompose
  }
  'start-api' {
    .\scripts\start_feedback_api.ps1
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
}
