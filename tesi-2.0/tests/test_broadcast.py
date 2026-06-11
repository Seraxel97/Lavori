"""Tests DR-VAULT-broadcast-log: common/broadcast.py emit + concorrenza."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from common.broadcast import emit, read_log


def test_emit_and_read_back(tmp_path: Path) -> None:
    """emit → file JSONL → read_log parsa correttamente ogni campo."""
    log = tmp_path / "broadcast.jsonl"

    import common.broadcast as bmod
    original = bmod.LOG_PATH
    bmod.LOG_PATH = log
    try:
        emit("sprint_claimed", sprint_id="S-XX", owner="sonnet1-ts")
        emit("sprint_done", sprint_id="S-XX", verdict="PASS")
        emit("worker_idle", worker="sonnet1-ts", age_min=12)
    finally:
        bmod.LOG_PATH = original

    events = read_log(log)
    assert len(events) == 3, f"Attese 3 righe, trovate {len(events)}"

    assert events[0]["event"] == "sprint_claimed"
    assert events[0]["sprint_id"] == "S-XX"
    assert events[0]["owner"] == "sonnet1-ts"
    assert "ts" in events[0]

    assert events[1]["event"] == "sprint_done"
    assert events[1]["verdict"] == "PASS"

    assert events[2]["event"] == "worker_idle"
    assert events[2]["age_min"] == 12

    # Ogni riga deve essere JSON valido indipendentemente
    for line in log.read_text().splitlines():
        parsed = json.loads(line)
        assert "ts" in parsed and "event" in parsed


def test_emit_100_concurrent_no_corruption(tmp_path: Path) -> None:
    """100 emit concorrenti → 100 righe, nessuna corruzione JSONL."""
    log = tmp_path / "concurrent_broadcast.jsonl"

    import common.broadcast as bmod
    original = bmod.LOG_PATH
    bmod.LOG_PATH = log

    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            emit("test_event", idx=idx, worker=f"t-{idx:03d}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        bmod.LOG_PATH = original

    assert not errors, f"Thread errors: {errors}"

    lines = [ln for ln in log.read_text().splitlines() if ln.strip()]
    assert len(lines) == 100, f"Attese 100 righe, trovate {len(lines)}"

    # Ogni riga deve essere JSON parsabile e avere event + idx
    indices_seen: set[int] = set()
    for line in lines:
        event = json.loads(line)   # raise se corrotto
        assert event["event"] == "test_event"
        assert "idx" in event
        indices_seen.add(event["idx"])

    assert indices_seen == set(range(100)), "Indici mancanti o duplicati"
