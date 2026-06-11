"""Tests dashboard/plot_top_features.py — PNG headless via matplotlib Agg."""

from __future__ import annotations

import numpy as np
import pytest

from dashboard.plot_top_features import plot_top_edges, plot_top_roi


def test_plot_top_roi_creates_png(tmp_path: pytest.TempPathFactory) -> None:
    """plot_top_roi crea un file PNG non vuoto."""
    scores = {f"roi-{i:02d}": float(np.random.default_rng(i).standard_normal()) * 0.05
              for i in range(15)}
    out = tmp_path / "test_roi.png"
    result = plot_top_roi(scores, "aparc", n_top=10, output_path=out)
    assert result == out
    assert out.exists(), "PNG non creato"
    assert out.stat().st_size > 1000, f"PNG troppo piccolo: {out.stat().st_size} bytes"


def test_plot_top_edges_creates_png(tmp_path: pytest.TempPathFactory) -> None:
    """plot_top_edges crea un file PNG non vuoto."""
    n = 10
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((n, n)) * 0.05
    mat = (mat + mat.T) / 2
    np.fill_diagonal(mat, 0.0)
    names = [f"roi-{i:02d}" for i in range(n)]
    out = tmp_path / "test_edges.png"
    result = plot_top_edges(mat, names, n_top=5, output_path=out)
    assert result == out
    assert out.exists(), "PNG non creato"
    assert out.stat().st_size > 1000, f"PNG troppo piccolo: {out.stat().st_size} bytes"
