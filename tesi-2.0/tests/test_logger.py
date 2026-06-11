"""Test common/logger.py — JsonLogger structured logging."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from common.logger import JsonLogger

# ── Test 1: emit eventi → file JSONL creato e parseable ──────────────────────


def test_logger_creates_file(tmp_path: Path) -> None:
    """Logger crea il file JSONL al primo emit."""
    logger = JsonLogger(run_id="test_run_001", log_dir=tmp_path)
    assert not logger.path.exists()

    logger.log("step_start", step="preprocessing")
    assert logger.path.exists()


def test_logger_jsonl_parseable(tmp_path: Path) -> None:
    """Ogni riga del file è JSON valido con i campi attesi."""
    logger = JsonLogger(run_id="test_run_002", log_dir=tmp_path)
    logger.log("step_start", step="preprocessing")
    logger.log("step_done", step="preprocessing", duration_s=3.2)

    lines = logger.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    for line in lines:
        entry = json.loads(line)
        assert "ts" in entry
        assert "iso" in entry
        assert "event" in entry
        assert entry["run_id"] == "test_run_002"
        assert isinstance(entry["ts"], float)


def test_logger_event_fields(tmp_path: Path) -> None:
    """I campi extra vengono serializzati correttamente."""
    logger = JsonLogger(run_id="test_run_003", log_dir=tmp_path)
    logger.log("metric_computed", atlas="aparc", metric="wpli", ba=0.612)

    events = logger.read_events()
    assert len(events) == 1
    e = events[0]
    assert e["event"] == "metric_computed"
    assert e["atlas"] == "aparc"
    assert e["metric"] == "wpli"
    assert abs(e["ba"] - 0.612) < 1e-9


def test_logger_multiple_runs_separate_files(tmp_path: Path) -> None:
    """Run ID diversi producono file separati."""
    l1 = JsonLogger(run_id="run_A", log_dir=tmp_path)
    l2 = JsonLogger(run_id="run_B", log_dir=tmp_path)
    l1.log("event_a")
    l2.log("event_b")

    assert l1.path != l2.path
    assert l1.path.exists()
    assert l2.path.exists()
    assert json.loads(l1.path.read_text().strip())["event"] == "event_a"
    assert json.loads(l2.path.read_text().strip())["event"] == "event_b"


def test_logger_read_events_empty(tmp_path: Path) -> None:
    """read_events ritorna lista vuota se il file non esiste ancora."""
    logger = JsonLogger(run_id="ghost_run", log_dir=tmp_path)
    assert logger.read_events() == []


def test_logger_creates_log_dir(tmp_path: Path) -> None:
    """Il log_dir viene creato automaticamente se non esiste."""
    nested = tmp_path / "deep" / "nested" / "logs"
    logger = JsonLogger(run_id="test_nested", log_dir=nested)
    logger.log("init")
    assert nested.exists()
    assert logger.path.exists()


# ── Test 2: 100 emit concurrent — thread-safe ─────────────────────────────────


def test_logger_thread_safe_100_emits(tmp_path: Path) -> None:
    """100 thread concorrenti emettono senza corruzione del file JSONL."""
    logger = JsonLogger(run_id="concurrent_run", log_dir=tmp_path)
    n_threads = 100
    errors: list[Exception] = []

    def emit(i: int) -> None:
        try:
            logger.log("thread_event", thread_id=i, value=i * 1.5)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=emit, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errori in thread: {errors}"

    events = logger.read_events()
    assert len(events) == n_threads, (
        f"Attesi {n_threads} eventi, trovati {len(events)} "
        f"— possibile corruzione JSONL"
    )

    thread_ids = {e["thread_id"] for e in events}
    assert thread_ids == set(range(n_threads)), "Thread ID mancanti — scritture perse"

    for e in events:
        assert e["run_id"] == "concurrent_run"
        assert e["event"] == "thread_event"


def test_logger_thread_safe_ordering(tmp_path: Path) -> None:
    """Gli eventi nel file sono singole righe JSON complete (no interleaving)."""
    logger = JsonLogger(run_id="order_run", log_dir=tmp_path)
    n = 50

    def emit_many(start: int) -> None:
        for i in range(start, start + 10):
            logger.log("item", idx=i)

    threads = [threading.Thread(target=emit_many, args=(i * 10,)) for i in range(n // 10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    raw_lines = logger.path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == n

    for line in raw_lines:
        parsed = json.loads(line)
        assert "idx" in parsed
