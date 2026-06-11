#!/bin/bash
# Data directory audit — inventory, orphan detection, cleanup recommendations
# NO automatic deletion — report only

set -o pipefail

ROOT="/home/seraxel/Scrivania/Tesi_2.0"

{
    echo "# Data Directory Audit Report"
    echo ""
    echo "**Timestamp**: $(date -Iseconds)"
    echo "**Constraint**: disk critical 22GB liberi (22 GB required for scientific pipeline)"
    echo ""
    echo "## System Disk Status"
    echo ""
    echo '```'
    df -h "$ROOT"
    echo '```'
    echo ""
    echo "**Free available**: $(df -h "$ROOT" | tail -1 | awk '{print $4}')"
    echo ""

    # Data directory inventory
    echo "## Data Directory Inventory"
    echo ""
    echo "Total size:"
    echo '```'
    du -sh "$ROOT/data" 2>/dev/null
    echo '```'
    echo ""
    echo "Subdirectory breakdown:"
    echo ""
    echo '```'
    du -sh "$ROOT/data"/* 2>/dev/null | sort -h
    echo '```'
    echo ""

    # Large files
    echo "## Large Files (>100MB)"
    echo ""
    echo '```'
    find "$ROOT/data" -type f -size +100M -exec ls -lh {} + 2>/dev/null | awk '{print $5, $9}' | sort -h
    echo '```'
    echo ""

    # Mock data (S-29 generated, potentially large and redundant)
    echo "## Mock Data Inventory"
    echo ""
    if [ -d "$ROOT/data/mock_eceo" ]; then
        echo "**mock_eceo** (S-29 generated):"
        echo '```'
        du -sh "$ROOT/data/mock_eceo"
        echo '```'
        echo ""
    else
        echo "**mock_eceo**: not found"
        echo ""
    fi

    if [ -d "$ROOT/data/mock_validation_test" ]; then
        echo "**mock_validation_test** (S-52 test):"
        echo '```'
        du -sh "$ROOT/data/mock_validation_test"
        echo '```'
        echo ""
    else
        echo "**mock_validation_test**: not found"
        echo ""
    fi

    # Derivatives (STEP 1 output — protected)
    echo "## Derivatives Directory (Protected)"
    echo ""
    echo "**Status**: Contains STEP 1 output (read-only, no delete)"
    echo ""
    if [ -d "$ROOT/data/derivatives" ]; then
        echo '```'
        du -sh "$ROOT/data/derivatives"/*/ 2>/dev/null | sort -h
        echo '```'
    else
        echo "**derivatives**: not found"
    fi
    echo ""

    # EEG matchingpennies (main dataset — likely large)
    echo "## EEG MatchingPennies Dataset"
    echo ""
    if [ -d "$ROOT/data/eeg_matchingpennies" ]; then
        echo '```'
        du -sh "$ROOT/data/eeg_matchingpennies"
        echo '```'
        echo ""
        echo "Subdirectories:"
        echo '```'
        du -sh "$ROOT/data/eeg_matchingpennies"/* 2>/dev/null | sort -h | head -20
        echo '```'
    else
        echo "**eeg_matchingpennies**: not found (submodule?)"
    fi
    echo ""

    # Cleanup recommendations
    echo "## Cleanup Recommendations (NO auto-delete)"
    echo ""
    echo "**PROTECTED (do NOT delete)**:"
    echo "- \`data/raw/\` (read-only symlink, source data)"
    echo "- \`data/derivatives/mne-bids-pipeline/\` (STEP 1 output)"
    echo ""
    echo "**CANDIDATES for cleanup (operator decision)**:"
    echo ""

    # Check mock_eceo
    if [ -d "$ROOT/data/mock_eceo" ]; then
        MOCK_SIZE=$(du -sh "$ROOT/data/mock_eceo" 2>/dev/null | awk '{print $1}')
        echo "1. **mock_eceo/** ($MOCK_SIZE)"
        echo "   - Purpose: S-29 generated mock BIDS dataset (testing only)"
        echo "   - Recoverable: $MOCK_SIZE"
        echo "   - Keep if: re-running S-29 or S-52 tests"
        echo "   - Remove if: tests complete, need disk"
        echo ""
    fi

    # Check test directories
    if [ -d "$ROOT/data/mock_validation_test" ]; then
        echo "2. **mock_validation_test/** (test scratch)"
        echo "   - Purpose: S-52 validation test working dir"
        echo "   - Note: should be auto-cleaned per S-52 spec"
        echo "   - Action: manual cleanup if persists"
        echo ""
    fi

    # Check eeg_matchingpennies size (if submodule, likely large)
    if [ -d "$ROOT/data/eeg_matchingpennies" ]; then
        EEG_SIZE=$(du -sh "$ROOT/data/eeg_matchingpennies" 2>/dev/null | awk '{print $1}')
        echo "3. **eeg_matchingpennies/** ($EEG_SIZE)"
        echo "   - Purpose: main dataset (STEP 1-7 pipeline input)"
        echo "   - Status: submodule (external git-annex or similar)"
        echo "   - Action: check if sparse checkout can reduce size"
        echo "   - WARNING: required for scientific pipeline execution"
        echo ""
    fi

    echo "## Disk Space Estimation (post-cleanup scenarios)"
    echo ""
    echo "**Current state**: $(df -h "$ROOT" | tail -1 | awk '{print $4}') free"
    echo ""
    echo "**Scenario A** (remove mock_eceo only):"
    if [ -d "$ROOT/data/mock_eceo" ]; then
        MOCK=$(du -sb "$ROOT/data/mock_eceo" 2>/dev/null | awk '{print int($1/1024/1024/1024) "GB"}')
        echo "- Estimated recovery: $MOCK"
    fi
    echo ""
    echo "**Scenario B** (remove mock_eceo + mock_validation_test):"
    echo "- Estimated recovery: see Scenario A + minor test artifacts"
    echo ""
    echo "**Scenario C** (optimize eeg_matchingpennies via sparse checkout):"
    echo "- Estimated recovery: varies (contact git-annex maintainer)"
    echo "- Risk: may break submodule integrity"
    echo ""

    echo "## Recommendations"
    echo ""
    echo "**Priority order** (for $22 GB goal):"
    echo "1. Remove mock_eceo if tests complete → immediate recovery"
    echo "2. Check git-annex optimization for eeg_matchingpennies"
    echo "3. Verify no duplicate .fif/.vhdr/.eeg files"
    echo ""
    echo "**Next steps**: Operator reviews this report, authorizes cleanup via separate task."
    echo ""

} > "${ROOT}/reports/DATA_AUDIT.md" 2>&1

echo "✓ Data audit complete → ${ROOT}/reports/DATA_AUDIT.md"
