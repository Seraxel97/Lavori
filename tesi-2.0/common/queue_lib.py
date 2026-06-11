"""Queue file lock helpers — fcntl-based lock per evitare append concurrent corruption."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import IO


@contextmanager
def queue_lock(queue_path: Path, mode: str = "a") -> Generator[IO]:
    """Context manager che apre il file con flock esclusivo."""
    f = open(queue_path, mode)  # noqa: SIM115
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield f
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def append_sprint(queue_path: Path, entry: dict) -> None:
    """Append safely una riga jsonl con lock."""
    with queue_lock(queue_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    from common.broadcast import emit

    emit("sprint_added", sprint_id=entry["id"], owner=entry.get("owner"))


def update_status(queue_path: Path, sprint_id: str, **updates) -> bool:
    """Aggiorna in-place lo status di uno sprint (read-modify-write atomico via lock)."""
    with queue_lock(queue_path, "r+") as f:
        lines = f.read().splitlines()
        f.seek(0)
        f.truncate()
        found = False
        for ln in lines:
            if not ln.strip():
                continue
            e = json.loads(ln)
            if e["id"] == sprint_id:
                e.update(updates)
                found = True
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    if found:
        from common.broadcast import emit

        if updates.get("status") == "claimed":
            emit("sprint_claimed", sprint_id=sprint_id, owner=updates.get("owner"))
        if updates.get("status") == "done":
            emit("sprint_done", sprint_id=sprint_id, verdict=updates.get("verdict"))
    return found
