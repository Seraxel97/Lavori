"""Tests per queue_lib.py — race condition prevention via fcntl.flock."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from common.queue_lib import append_sprint, queue_lock, update_status


def test_append_concurrent(tmp_path: Path) -> None:
    """5 thread append simultaneo → 5 righe, ognuna parsabile."""
    queue = tmp_path / "queue.jsonl"
    queue.touch()

    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            append_sprint(queue, {"id": f"T-{idx:02d}", "status": "queued"})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    lines = [ln for ln in queue.read_text().splitlines() if ln.strip()]
    assert len(lines) == 5, f"Expected 5 lines, got {len(lines)}"
    ids = {json.loads(ln)["id"] for ln in lines}
    assert ids == {f"T-{i:02d}" for i in range(5)}


def test_update_status_atomic(tmp_path: Path) -> None:
    """append + update_status → entry status aggiornato correttamente."""
    queue = tmp_path / "queue.jsonl"
    append_sprint(queue, {"id": "S-TEST", "status": "queued", "owner": None})

    found = update_status(queue, "S-TEST", status="done", owner="sonnet2-ts")

    assert found
    lines = [ln for ln in queue.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["status"] == "done"
    assert entry["owner"] == "sonnet2-ts"
    assert entry["id"] == "S-TEST"


def test_lock_blocks_concurrent(tmp_path: Path) -> None:
    """Secondo lock si blocca finché il primo non rilascia."""
    queue = tmp_path / "queue.jsonl"
    queue.touch()

    order: list[str] = []
    released = threading.Event()

    def holder() -> None:
        with queue_lock(queue, "a"):
            order.append("lock_acquired")
            time.sleep(0.15)
            order.append("before_release")
        released.set()

    def waiter() -> None:
        released.wait(timeout=0.05)
        with queue_lock(queue, "a"):
            order.append("lock_acquired_by_waiter")

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)
    t1.start()
    time.sleep(0.02)
    t2.start()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)

    assert "before_release" in order
    assert "lock_acquired_by_waiter" in order
    before_idx = order.index("before_release")
    waiter_idx = order.index("lock_acquired_by_waiter")
    assert waiter_idx > before_idx, f"Waiter acquired lock before holder released: {order}"
