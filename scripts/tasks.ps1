param(
  [Parameter(Mandatory=$true)]
  [ValidateSet('init','test','smoke','download-data')]
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
}
