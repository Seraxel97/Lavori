"""Tests per hb_lib.py — atomic heartbeat write via tempfile+os.replace."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from common.hb_lib import is_stale, read_all_heartbeats, read_heartbeat, write_heartbeat


def test_write_heartbeat_creates_valid_json(tmp_path: Path) -> None:
    """write_heartbeat produce un JSON valido leggibile da read_heartbeat."""
    hb = tmp_path / "worker.json"
    data = {"worker": "sonnet2-ts", "status": "working", "task": "S-99"}
    write_heartbeat(hb, data)
    assert hb.exists()
    loaded = json.loads(hb.read_text())
    assert loaded == data


def test_write_heartbeat_atomic_overwrite(tmp_path: Path) -> None:
    """Sovrascrittura atomica: il file finale contiene solo l'ultimo dato."""
    hb = tmp_path / "worker.json"
    write_heartbeat(hb, {"status": "v1"})
    write_heartbeat(hb, {"status": "v2"})
    assert read_heartbeat(hb) == {"status": "v2"}


def test_read_heartbeat_missing_returns_none(tmp_path: Path) -> None:
    """read_heartbeat ritorna None se il file non esiste."""
    hb = tmp_path / "nonexistent.json"
    assert read_heartbeat(hb) is None


def test_read_heartbeat_corrupt_returns_none(tmp_path: Path) -> None:
    """read_heartbeat ritorna None se il file è JSON malformato."""
    hb = tmp_path / "bad.json"
    hb.write_text("{corrupted json{{")
    assert read_heartbeat(hb) is None


def test_write_heartbeat_concurrent(tmp_path: Path) -> None:
    """5 thread scrivono concorrentemente: file finale è JSON valido."""
    hb = tmp_path / "concurrent.json"
    errors: list[Exception] = []

    def worker(idx: int) -> None:
        try:
            write_heartbeat(hb, {"worker": f"t-{idx}", "status": "working"})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    # Qualunque sia il vincitore della race, il file deve essere JSON valido
    result = read_heartbeat(hb)
    assert result is not None
    assert "worker" in result
    assert result["status"] == "working"


def test_write_heartbeat_creates_parent(tmp_path: Path) -> None:
    """write_heartbeat crea la directory padre se non esiste."""
    hb = tmp_path / "subdir" / "nested" / "worker.json"
    write_heartbeat(hb, {"status": "ok"})
    assert hb.exists()


# ── read_all_heartbeats ───────────────────────────────────────────────────────


def test_read_all_heartbeats_three_files(tmp_path: Path) -> None:
    """read_all_heartbeats legge 3 HB fake e aggiunge _source_file."""
    workers = [
        {"worker": "sonnet1-ts", "status": "working", "sprint_id": "S-01"},
        {"worker": "sonnet2-ts", "status": "done",    "sprint_id": "S-02"},
        {"worker": "haiku1-ts",  "status": "idle",    "sprint_id": None},
    ]
    for w in workers:
        write_heartbeat(tmp_path / f"{w['worker']}.json", w)

    results = read_all_heartbeats(tmp_path)
    assert len(results) == 3

    names = {r["_source_file"] for r in results}
    assert names == {"sonnet1-ts.json", "sonnet2-ts.json", "haiku1-ts.json"}

    for r in results:
        assert "worker" in r
        assert "status" in r


def test_read_all_heartbeats_empty_dir(tmp_path: Path) -> None:
    """read_all_heartbeats su dir vuota ritorna lista vuota."""
    assert read_all_heartbeats(tmp_path) == []


def test_read_all_heartbeats_missing_dir(tmp_path: Path) -> None:
    """read_all_heartbeats su dir inesistente ritorna lista vuota."""
    assert read_all_heartbeats(tmp_path / "ghost") == []


def test_read_all_heartbeats_skips_corrupt(tmp_path: Path) -> None:
    """read_all_heartbeats salta file JSON corrotti senza errori."""
    (tmp_path / "good.json").write_text('{"worker": "ok"}')
    (tmp_path / "bad.json").write_text("{bad json{{")
    results = read_all_heartbeats(tmp_path)
    assert len(results) == 1
    assert results[0]["worker"] == "ok"


def test_read_all_heartbeats_ignores_tmp(tmp_path: Path) -> None:
    """read_all_heartbeats ignora file .tmp (scrittura in corso)."""
    (tmp_path / "worker.json").write_text('{"status": "ok"}')
    (tmp_path / "worker.tmp").write_text('{"status": "partial"}')
    results = read_all_heartbeats(tmp_path)
    assert len(results) == 1


# ── is_stale ──────────────────────────────────────────────────────────────────


def test_is_stale_recent_ts(tmp_path: Path) -> None:
    """HB con ts recente (now) non è stale."""
    hb = {"ts": time.time(), "status": "working"}
    assert not is_stale(hb, threshold_min=10)


def test_is_stale_old_ts(tmp_path: Path) -> None:
    """HB con ts di 20 minuti fa è stale con threshold=10."""
    old_ts = time.time() - 20 * 60
    hb = {"ts": old_ts, "status": "working"}
    assert is_stale(hb, threshold_min=10)


def test_is_stale_threshold_variable(tmp_path: Path) -> None:
    """Threshold variabile: stesso HB è stale con threshold=1 ma non con threshold=60."""
    ts_5min_ago = time.time() - 5 * 60
    hb = {"ts": ts_5min_ago, "status": "working"}
    assert is_stale(hb, threshold_min=1)       # 5 min > 1 min → stale
    assert not is_stale(hb, threshold_min=60)  # 5 min < 60 min → fresh


def test_is_stale_iso_timestamp(tmp_path: Path) -> None:
    """is_stale riconosce campo timestamp in formato ISO-8601."""
    from datetime import UTC, datetime, timedelta
    recent = (datetime.now(UTC) - timedelta(minutes=2)).isoformat()
    old = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    assert not is_stale({"timestamp": recent}, threshold_min=10)
    assert is_stale({"timestamp": old}, threshold_min=10)


def test_is_stale_ts_takes_priority_over_timestamp(tmp_path: Path) -> None:
    """Campo ts ha priorità su timestamp."""
    from datetime import UTC, datetime, timedelta
    # ts recente, timestamp vecchio → non stale (ts vince)
    recent_ts = time.time()
    old_iso = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    hb = {"ts": recent_ts, "timestamp": old_iso}
    assert not is_stale(hb, threshold_min=10)


def test_is_stale_no_timestamp_returns_true(tmp_path: Path) -> None:
    """HB senza campo temporale è considerato stale (fail-safe)."""
    hb = {"worker": "ghost", "status": "working"}
    assert is_stale(hb, threshold_min=10)
