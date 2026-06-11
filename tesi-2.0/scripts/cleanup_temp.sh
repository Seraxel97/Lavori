#!/bin/bash
# Cleanup orphan __pycache__ and temporary files

set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${REPO_ROOT}/reports/CLEANUP_LOG.md"
mkdir -p "${REPO_ROOT}/reports"

{
    echo "# Cleanup Log"
    echo ""
    echo "**Timestamp**: $(date -Iseconds)"
    echo "**Repo**: ${REPO_ROOT}"
    echo ""
    echo "## Actions Taken"
    echo ""
} > "$LOG_FILE"

# Function to log and cleanup
cleanup_pattern() {
    local pattern="$1"
    local description="$2"

    echo "Cleaning: $description"
    {
        echo "### $description"
        echo ""
        echo '```'
    } >> "$LOG_FILE"

    count=0
    while IFS= read -r -d '' item; do
        rm -rf "$item"
        echo "$item"
        ((count++))
    done < <(find "$REPO_ROOT" -name "$pattern" -print0 2>/dev/null)

    {
        echo '```'
        echo ""
        echo "**Count**: $count removed"
        echo ""
    } >> "$LOG_FILE"

    echo "  ✓ Removed $count items"
}

# Cleanup patterns
cleanup_pattern "__pycache__" "__pycache__ directories"
cleanup_pattern "*.pyc" "Compiled Python files (.pyc)"
cleanup_pattern ".DS_Store" "macOS metadata files"
cleanup_pattern "*.egg-info" "egg-info directories"

# Cleanup specific files
echo "Cleaning: Coverage and cache files"
{
    echo "### Coverage and cache files"
    echo ""
    echo '```'
} >> "$LOG_FILE"

count=0
for pattern in ".coverage" ".pytest_cache" ".ruff_cache" ".mypy_cache"; do
    while IFS= read -r -d '' item; do
        rm -rf "$item"
        echo "$item"
        ((count++))
    done < <(find "$REPO_ROOT" -name "$pattern" -print0 2>/dev/null)
done

{
    echo '```'
    echo ""
    echo "**Count**: $count removed"
    echo ""
} >> "$LOG_FILE"

echo "  ✓ Removed $count cache/coverage files"

# Cleanup root-level temp files (but not .planning, .git, data/)
echo "Cleaning: Root-level temp files"
{
    echo "### Root-level temp files"
    echo ""
    echo '```'
} >> "$LOG_FILE"

count=0
for tmpfile in "$REPO_ROOT"/.tmp* "$REPO_ROOT"/*.tmp "$REPO_ROOT"/*.log; do
    if [ -e "$tmpfile" ]; then
        rm -rf "$tmpfile"
        echo "$tmpfile"
        ((count++))
    fi
done

{
    echo '```'
    echo ""
    echo "**Count**: $count removed"
    echo ""
    echo "## Summary"
    echo ""
    echo "✓ Cleanup completed successfully"
    echo ""
    echo "**Preserved**:"
    echo "- .planning/ (orchestration state)"
    echo "- .git/ (version control)"
    echo "- data/ (datasets)"
    echo "- .env files (secrets, if any)"
} >> "$LOG_FILE"

echo "  ✓ Removed $count temp files"

echo ""
echo "✓ Cleanup complete → $LOG_FILE"
