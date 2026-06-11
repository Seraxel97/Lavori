"""Tests for progress bar wrapper."""

from __future__ import annotations

import pytest

from common import progress as progress_module


class TestProgress:
    """Test progress bar wrapper with/without tqdm."""

    def test_progress_passthrough(self) -> None:
        """Test: progress() returns iterable when tqdm unavailable.

        Simulates missing tqdm by temporarily disabling _HAS_TQDM.
        """
        original_has_tqdm = progress_module._HAS_TQDM
        try:
            progress_module._HAS_TQDM = False
            data = [1, 2, 3, 4, 5]
            result = progress_module.progress(data, desc="test")

            # Should return original iterable (not wrapped)
            assert list(result) == data, "Passthrough should preserve list"
        finally:
            progress_module._HAS_TQDM = original_has_tqdm

    def test_progress_with_tqdm(self) -> None:
        """Test: progress() wraps with tqdm when available."""
        if not progress_module._HAS_TQDM:
            pytest.skip("tqdm not installed")

        data = [1, 2, 3, 4, 5]
        result = progress_module.progress(data, total=5, desc="test")

        # tqdm wraps but is still iterable
        result_list = list(result)
        assert result_list == data, "tqdm-wrapped iteration should preserve data"
