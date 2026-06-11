#!/usr/bin/env bash
# Verifica che tutti tmux session ts-named siano in /home/seraxel/Scrivania/Tesi_2.0/
EXPECTED="/home/seraxel/Scrivania/Tesi_2.0"
FAILED=0
for s in $(tmux ls 2>/dev/null | grep -oE "^[a-z0-9-]+-ts" || true); do
    cwd=$(tmux display-message -t "$s" -p '#{pane_current_path}' 2>/dev/null)
    if [[ "$cwd" == "$EXPECTED"* ]]; then
        echo "  OK  $s : $cwd"
    else
        echo "  FAIL $s : $cwd (expected: $EXPECTED)"
        FAILED=1
    fi
done
exit $FAILED
