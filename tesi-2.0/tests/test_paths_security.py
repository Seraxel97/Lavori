"""Tests for path security validation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from common.paths import _validate_env_path


class TestPathsSecurity:
    """Test path validation for security."""

    def test_relative_path_rejected(self) -> None:
        """Test: relative path in env var raises ValueError."""
        with pytest.raises(ValueError, match="must be absolute path"):
            _validate_env_path("TEST_PATH", Path("./relative/path"))

    def test_nonexistent_path_rejected(self) -> None:
        """Test: nonexistent path with must_exist=True raises FileNotFoundError."""
        nonexistent = Path("/nonexistent/path/12345/should/not/exist")
        with pytest.raises(FileNotFoundError, match="not found"):
            _validate_env_path("TEST_PATH", nonexistent, must_exist=True)

    def test_traversal_resolved(self) -> None:
        """Test: path traversal (~/../../etc/passwd) is resolved to canonical absolute."""
        with TemporaryDirectory() as tmpdir:
            # Create a test directory structure
            test_dir = Path(tmpdir) / "test"
            test_dir.mkdir()

            # Simulate traversal attempt: use a path that resolves to tmpdir
            traversal_path = test_dir / ".." / "test"
            resolved = _validate_env_path(
                "TEST_PATH", traversal_path, must_exist=False
            )

            # Should resolve to canonical absolute path
            assert resolved.is_absolute(), "Should be absolute"
            assert ".." not in str(resolved), "Should not contain .."
            assert resolved == resolved.resolve(), "Should be fully resolved"
