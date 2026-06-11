"""Tests for broadcast emit hooks in queue_lib."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from common.broadcast import read_log
from common.queue_lib import append_sprint, update_status


class TestBroadcastHooks:
    """Test emit hooks in queue operations."""

    def test_append_emits_added(self) -> None:
        """Test: append_sprint emits 'sprint_added' event."""
        with TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.jsonl"
            log_path = Path(tmpdir) / "broadcast_log.jsonl"

            # Mock emit to write to temp log
            import common.broadcast as bc

            original_log_path = bc.LOG_PATH
            bc.LOG_PATH = log_path

            try:
                entry = {"id": "S-10", "owner": "haiku1-ts", "task": "test"}
                append_sprint(queue_path, entry)

                events = read_log(log_path)
                assert len(events) >= 1, "No events emitted"
                assert any(e["event"] == "sprint_added" for e in events)
                event = [e for e in events if e["event"] == "sprint_added"][0]
                assert event["sprint_id"] == "S-10"
                assert event["owner"] == "haiku1-ts"
            finally:
                bc.LOG_PATH = original_log_path

    def test_claim_emits_claimed(self) -> None:
        """Test: update_status with claimed emits 'sprint_claimed' event."""
        with TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.jsonl"
            log_path = Path(tmpdir) / "broadcast_log.jsonl"

            # Setup: create initial entry
            entry = {"id": "S-20", "status": "pending", "owner": None}
            with open(queue_path, "w") as f:
                f.write(json.dumps(entry) + "\n")

            # Mock emit to write to temp log
            import common.broadcast as bc

            original_log_path = bc.LOG_PATH
            bc.LOG_PATH = log_path

            try:
                update_status(
                    queue_path, "S-20", status="claimed", owner="sonnet1-ts"
                )

                events = read_log(log_path)
                assert any(e["event"] == "sprint_claimed" for e in events)
                event = [e for e in events if e["event"] == "sprint_claimed"][0]
                assert event["sprint_id"] == "S-20"
                assert event["owner"] == "sonnet1-ts"
            finally:
                bc.LOG_PATH = original_log_path

    def test_done_emits_done(self) -> None:
        """Test: update_status with done emits 'sprint_done' event."""
        with TemporaryDirectory() as tmpdir:
            queue_path = Path(tmpdir) / "queue.jsonl"
            log_path = Path(tmpdir) / "broadcast_log.jsonl"

            # Setup: create initial entry
            entry = {"id": "S-30", "status": "claimed", "verdict": None}
            with open(queue_path, "w") as f:
                f.write(json.dumps(entry) + "\n")

            # Mock emit to write to temp log
            import common.broadcast as bc

            original_log_path = bc.LOG_PATH
            bc.LOG_PATH = log_path

            try:
                update_status(queue_path, "S-30", status="done", verdict="PASS")

                events = read_log(log_path)
                assert any(e["event"] == "sprint_done" for e in events)
                event = [e for e in events if e["event"] == "sprint_done"][0]
                assert event["sprint_id"] == "S-30"
                assert event["verdict"] == "PASS"
            finally:
                bc.LOG_PATH = original_log_path
