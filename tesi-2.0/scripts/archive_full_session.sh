#!/usr/bin/env bash
# Sprint S-64: archivia MSG_*, SYNC_*, REVIEW_* da .planning/ in audits/<date>/full_session/
# Usa mv (non delete). Idempotente: salta file già archiviati.
set -euo pipefail

PLANNING="/home/seraxel/Scrivania/Tesi_2.0/.planning"
DATE="${1:-$(date +%Y-%m-%d)}"
DEST="${PLANNING}/audits/${DATE}/full_session"

mkdir -p "$DEST"
echo "Archiving to: $DEST"
echo ""

archived=0
skipped=0

for pattern in "MSG_*.md" "SYNC_*.md" "REVIEW_*.md"; do
    for f in "${PLANNING}"/${pattern}; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        if [ -f "${DEST}/${fname}" ]; then
            echo "  skip (exists): $fname"
            ((skipped++)) || true
        else
            mv "$f" "${DEST}/"
            echo "  archived: $fname"
            ((archived++)) || true
        fi
    done
done

echo ""
echo "Done. archived=${archived} skipped=${skipped}"
echo "Archive dir: ${DEST}"
echo "File count: $(ls -1 ${DEST} | wc -l)"
