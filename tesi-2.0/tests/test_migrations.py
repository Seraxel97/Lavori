"""Test schema migrations — apply/rollback/idempotency."""

import json
from pathlib import Path

import pytest

from common.state_lib import (
    _load_migration_modules,
    apply_migration,
    load_schema_version,
    migrate_on_load,
    rollback_migration,
)

# ── Fixture: JSONL temporaneo con record senza severity ──────────────────────

_SAMPLE_RECORDS = [
    {"id": "S-01", "sprint": "test sprint 1", "owner": None, "status": "queued",
     "deps": [], "scope": "test/s01", "branch": "test/s01", "created": "2026-05-01T00:00:00Z"},
    {"id": "S-02", "sprint": "test sprint 2", "owner": "sonnet1-ts", "status": "done",
     "deps": ["S-01"], "scope": "test/s02", "branch": "test/s02",
     "created": "2026-05-01T00:01:00Z", "severity": "HIGH"},
    {"id": "DR-01", "sprint": "deep review 1", "owner": None, "status": "queued",
     "deps": [], "scope": "fix/dr01", "branch": "fix/dr01",
     "created": "2026-05-01T00:02:00Z"},
]


@pytest.fixture
def tmp_queue(tmp_path: Path) -> Path:
    """JSONL temporaneo con record di test."""
    q = tmp_path / "TEST_QUEUE.jsonl"
    q.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in _SAMPLE_RECORDS),
        encoding="utf-8",
    )
    return q


@pytest.fixture(autouse=True)
def load_migrations():
    """Assicura che le migrazioni siano caricate prima di ogni test."""
    _load_migration_modules()


# ── Test 1: apply 001 e 002 ───────────────────────────────────────────────────

def test_apply_001_and_002(tmp_queue: Path):
    r001 = apply_migration("001", tmp_queue)
    assert r001["status"] == "applied"
    assert r001["records_affected"] == len(_SAMPLE_RECORDS)

    r002 = apply_migration("002", tmp_queue)
    assert r002["status"] == "applied"

    records = [json.loads(line) for line in tmp_queue.read_text().splitlines() if line.strip()]
    for rec in records:
        assert "severity" in rec, f"severity mancante in {rec['id']}"

    assert records[0]["severity"] == "LOW"
    assert records[1]["severity"] == "HIGH", "severity gia' presente deve restare intatta"
    assert records[2]["severity"] == "LOW"

    assert load_schema_version(tmp_queue) == "002"


# ── Test 2: rollback 002 ──────────────────────────────────────────────────────

def test_rollback_002(tmp_queue: Path):
    apply_migration("001", tmp_queue)
    apply_migration("002", tmp_queue)

    rb = rollback_migration("002", tmp_queue)
    assert rb["status"] == "rolled_back"

    records = [json.loads(line) for line in tmp_queue.read_text().splitlines() if line.strip()]
    for rec in records:
        assert "severity" not in rec, f"severity non rimossa da {rec['id']}"

    assert load_schema_version(tmp_queue) == "001"

    rb_again = rollback_migration("002", tmp_queue)
    assert rb_again["status"] == "skipped"


# ── Test 3: migrate_on_load idempotente ───────────────────────────────────────

def test_migrate_on_load_idempotent(tmp_queue: Path):
    results_first = migrate_on_load(tmp_queue, target_version="002")
    applied_first = [r for r in results_first if r["status"] == "applied"]
    assert len(applied_first) == 2

    results_second = migrate_on_load(tmp_queue, target_version="002")
    for r in results_second:
        assert r["status"] == "skipped", f"atteso skipped, got {r}"

    records = [json.loads(line) for line in tmp_queue.read_text().splitlines() if line.strip()]
    for rec in records:
        assert "severity" in rec

    assert load_schema_version(tmp_queue) == "002"
