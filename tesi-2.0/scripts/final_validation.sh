#!/usr/bin/env bash
# Sprint DR-FINAL-pipeline-validation: smoke check finale pipeline Tesi_2.0.
# Esegue: ruff + pytest integration + pytest perf regression + pytest core
# Output: exit 0 = PASS, exit 1 = FAIL
set -euo pipefail

REPO="/home/seraxel/Scrivania/Tesi_2.0"
cd "$REPO"

REPORT="${REPO}/reports/FINAL_VALIDATION.md"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
ERRORS=0

echo "# Final Pipeline Validation" > "$REPORT"
echo "" >> "$REPORT"
echo "**Timestamp**: ${TIMESTAMP}" >> "$REPORT"
echo "**Script**: scripts/final_validation.sh" >> "$REPORT"
echo "" >> "$REPORT"

run_check() {
    local label="$1"; shift
    echo -n "  [$label] "
    if "$@" > /tmp/fv_out.txt 2>&1; then
        echo "PASS"
        echo "| \`${label}\` | PASS | — |" >> "$REPORT"
    else
        echo "FAIL"
        echo "| \`${label}\` | **FAIL** | \`$(tail -3 /tmp/fv_out.txt | tr '\n' ' ')\` |" >> "$REPORT"
        ((ERRORS++)) || true
    fi
}

echo "## Check results" >> "$REPORT"
echo "" >> "$REPORT"
echo "| Check | Stato | Note |" >> "$REPORT"
echo "|-------|-------|------|" >> "$REPORT"

echo "=== Final Validation === ${TIMESTAMP}"
echo ""
echo "--- Ruff ---"
run_check "ruff/common" python3 -m ruff check common/
run_check "ruff/connectivity" python3 -m ruff check connectivity/
run_check "ruff/features" python3 -m ruff check features/
run_check "ruff/ml_training" python3 -m ruff check ml_training/
run_check "ruff/pipeline" python3 -m ruff check pipeline_mne_bids/
run_check "ruff/analysis" python3 -m ruff check analysis/

echo ""
echo "--- Pytest integration ---"
run_check "pytest/integration" python3 -m pytest tests/test_integration_final.py -q --tb=no

echo ""
echo "--- Pytest perf regression ---"
run_check "pytest/perf-regression" python3 -m pytest tests/test_perf_regression.py -q --tb=no

echo ""
echo "--- Pytest core (no heavy) ---"
run_check "pytest/core" python3 -m pytest tests/ -q --tb=no \
  --ignore=tests/test_e2e_smoke_minimal.py \
  --ignore=tests/test_rs_epochs.py \
  --ignore=tests/test_neuromaps_helper.py \
  --ignore=tests/test_dispatcher_base.py

echo "" >> "$REPORT"
if [ "$ERRORS" -eq 0 ]; then
    VERDICT="**PASS**"
    echo "" && echo "=== VERDICT: PASS ==="
else
    VERDICT="**FAIL (${ERRORS} checks failed)**"
    echo "" && echo "=== VERDICT: FAIL (${ERRORS} errors) ==="
fi

echo "## Verdict" >> "$REPORT"
echo "" >> "$REPORT"
echo "${VERDICT}" >> "$REPORT"
echo "" >> "$REPORT"
echo "Errors: ${ERRORS}" >> "$REPORT"

cat "$REPORT"
exit "$ERRORS"
