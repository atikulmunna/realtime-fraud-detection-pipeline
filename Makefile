.PHONY: init test lint download-data

init:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

test:
	pytest -q

lint:
	python -m py_compile src/common/feature_contract.py

download-data:
	python -m src.data.download_paysim --out data/raw
