#!/usr/bin/env bash
# Genera API documentation HTML via pdoc.
# Richiede: pip install pdoc
# Output:  docs/api/  (HTML navigabile, index.html come entry point)
set -e

PYTHONPATH="${PYTHONPATH:-.}" pdoc -o docs/api/ \
    parcellation \
    connectivity \
    features \
    ml_training \
    source_reconstruction \
    common \
    analysis \
    pipeline_mne_bids \
    dashboard

echo "Docs generated in docs/api/  (entry: docs/api/index.html)"
