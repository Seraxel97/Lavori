#!/usr/bin/env bash
# Sposta SYNC_<id>.md di sprint done in .planning/audits/<date>/sync_archive/
set +e
QUEUE="/home/seraxel/Scrivania/Tesi_2.0/.planning/DISPATCH_QUEUE_tesi.jsonl"
ARCHIVE_DIR="/home/seraxel/Scrivania/Tesi_2.0/.planning/audits/$(date +%Y-%m-%d)/sync_archive"
mkdir -p "$ARCHIVE_DIR"
while IFS= read -r line; do
    [ -z "$line" ] && continue
    id=$(echo "$line" | python3 -c "import sys, json; e=json.load(sys.stdin); print(e['id']) if e.get('status')=='done' else ''" 2>/dev/null)
    [ -z "$id" ] && continue
    sync_file=".planning/SYNC_${id}.md"
    if [ -f "$sync_file" ]; then
        mv "$sync_file" "$ARCHIVE_DIR/" && echo "  archived: $sync_file"
    fi
done < "$QUEUE"
