.PHONY: test lint format clean run-step1 run-step2 bench docs help audit clean-cache

PYTHON ?= python
CONFIG_STEP1 = config/config_step1_matchingpennies.py
CONFIG_STEP2 = config/config_step2_source_matchingpennies.py
DERIV = data/derivatives/mne-bids-pipeline

help:
	@echo "Targets: test, lint, format, docs, run-step1, run-step2, bench, clean, audit, clean-cache"

test:
	pytest tests/ -v

lint:
	ruff check .
	ruff format --check .

format:
	ruff check . --fix
	ruff format .

run-step1:
	$(PYTHON) -m cli.main step1 --config $(CONFIG_STEP1)

run-step2:
	$(PYTHON) -m cli.main step2 --config $(CONFIG_STEP2)

bench:
	$(PYTHON) -m cli.main bench --output reports/BENCH_MATRIX_RESULTS.json

docs:
	bash scripts/build_docs.sh

clean:
	rm -rf $(DERIV)/_cache __pycache__ .pytest_cache .ruff_cache

audit:
	bash scripts/repo_sanity.sh

clean-cache:
	rm -rf data/derivatives/_cache .joblib_cache
