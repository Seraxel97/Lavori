"""Tests S-16 — compare_runs + fdr_helper."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from analysis.compare_runs import load_bench, marginal_by_axis, top_n
from analysis.fdr_helper import apply_fdr

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_bench_json(tmp_path, runs: list[dict]) -> str:
    """Crea un BENCH_MATRIX_RESULTS.json temporaneo."""
    data = {"meta": {"dataset": "test"}, "runs": runs}
    p = tmp_path / "bench.json"
    p.write_text(json.dumps(data))
    return str(p)


def _synthetic_runs(n: int = 20) -> list[dict]:
    rng = np.random.default_rng(0)
    metrics = ["coh", "wpli", "plv"]
    atlases = ["aparc", "schaefer100"]
    algos = ["LR", "SVM"]
    bands = ["alpha", "beta"]
    rows = []
    for i in range(n):
        rows.append({
            "metric": metrics[i % len(metrics)],
            "atlas": atlases[i % len(atlases)],
            "algorithm": algos[i % len(algos)],
            "band": bands[i % len(bands)],
            "ba": float(rng.uniform(0.45, 0.75)),
        })
    return rows


# ── test 1: load_bench + top_n ────────────────────────────────────────────────


def test_load_and_top_n(tmp_path):
    """load_bench produce DataFrame corretto, top_n ritorna N righe ordinate desc."""
    runs = _synthetic_runs(20)
    path = _make_bench_json(tmp_path, runs)

    df = load_bench(path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    assert "ba" in df.columns

    top = top_n(df, 5)
    assert len(top) == 5
    # Verifica ordinamento decrescente
    bas = top["ba"].tolist()
    assert bas == sorted(bas, reverse=True), "top_n non ordinato desc"


def test_load_bench_placeholder(tmp_path):
    """load_bench su placeholder (runs=[]) ritorna DataFrame vuoto con colonne."""
    path = _make_bench_json(tmp_path, [])
    df = load_bench(path)
    assert isinstance(df, pd.DataFrame)
    assert df.empty
    assert "ba" in df.columns


# ── test 2: marginal_by_axis consistency ──────────────────────────────────────


def test_marginal_consistency(tmp_path):
    """marginal_by_axis: n_runs totali == len(df), ba_mean coerente."""
    runs = _synthetic_runs(20)
    path = _make_bench_json(tmp_path, runs)
    df = load_bench(path)

    for axis in ("metric", "atlas", "algorithm", "band"):
        agg = marginal_by_axis(df, axis)
        assert "ba_mean" in agg.columns
        assert "n_runs" in agg.columns
        assert agg["n_runs"].sum() == len(df), f"{axis}: n_runs sum mismatch"
        # ba_mean per ogni gruppo deve essere la media reale
        for _, row in agg.iterrows():
            expected_mean = df.loc[df[axis] == row[axis], "ba"].mean()
            assert abs(row["ba_mean"] - expected_mean) < 1e-9, (
                f"{axis}={row[axis]}: ba_mean {row['ba_mean']:.6f} != {expected_mean:.6f}"
            )


def test_marginal_invalid_axis(tmp_path):
    """marginal_by_axis solleva ValueError per asse non valido."""
    df = pd.DataFrame({"ba": [0.6]})
    with pytest.raises(ValueError, match="axis deve essere in"):
        marginal_by_axis(df, "invalid_axis")


# ── test 3: fdr_helper edge cases ────────────────────────────────────────────


def test_fdr_basic():
    """apply_fdr: p tutti < alpha → tutti reject; p tutti == 1 → nessun reject."""
    p_sig = np.full(10, 0.001)
    p_corr, reject = apply_fdr(p_sig, alpha=0.05)
    assert len(p_corr) == 10
    assert reject.all(), "Tutti significativi attesi reject=True"

    p_ns = np.ones(10)
    p_corr_ns, reject_ns = apply_fdr(p_ns, alpha=0.05)
    assert not reject_ns.any(), "p=1 non deve essere rigettato"


def test_fdr_empty():
    """apply_fdr su array vuoto ritorna tuple di array vuoti."""
    p_corr, reject = apply_fdr([])
    assert len(p_corr) == 0
    assert len(reject) == 0


def test_fdr_invalid():
    """apply_fdr solleva ValueError per p fuori [0, 1]."""
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        apply_fdr([0.5, -0.1, 1.0])
