#!/usr/bin/env bash
# Coverage report completo per Tesi_2.0.
# Richiede pytest-cov: pip install pytest-cov
set -e

REPORT_DIR="reports"
mkdir -p "$REPORT_DIR"

pytest tests/ \
  --cov=parcellation \
  --cov=connectivity \
  --cov=features \
  --cov=ml_training \
  --cov=source_reconstruction \
  --cov=common \
  --cov=analysis \
  --cov=pipeline_mne_bids \
  --cov=dashboard \
  --cov-report=term \
  --cov-report=html:"$REPORT_DIR/coverage_html" \
  --cov-report=json:"$REPORT_DIR/coverage.json" \
  "$@"
