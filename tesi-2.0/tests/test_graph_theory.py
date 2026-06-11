"""Test step 1.2: graph_theory metrics su matrice di connettività."""

from __future__ import annotations

import numpy as np
import pytest

from features.graph_theory import _threshold_proportional, compute_graph_metrics

RNG = np.random.default_rng(42)
N = 15  # piccola matrice per test veloci


@pytest.fixture(scope="module")
def W_sym():
    W = RNG.uniform(0, 1, (N, N))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    return W


@pytest.fixture(scope="module")
def metrics(W_sym):
    return compute_graph_metrics(W_sym, threshold=0.20)


def test_output_keys(metrics):
    expected = {
        "mean_degree", "mean_strength", "clustering_coeff",
        "path_length", "global_efficiency", "local_efficiency",
        "modularity_q", "small_worldness",
    }
    assert set(metrics.keys()) == expected


def test_no_nan_core(metrics):
    for key in ("mean_degree", "mean_strength", "clustering_coeff",
                "global_efficiency", "local_efficiency", "modularity_q"):
        assert not np.isnan(metrics[key]), f"{key} is NaN"


def test_clustering_range(metrics):
    assert 0.0 <= metrics["clustering_coeff"] <= 1.0


def test_global_efficiency_range(metrics):
    assert 0.0 <= metrics["global_efficiency"] <= 1.0


def test_modularity_range(metrics):
    assert -0.5 <= metrics["modularity_q"] <= 1.0


def test_mean_degree_positive(metrics):
    assert metrics["mean_degree"] >= 0.0


def test_threshold_reduces_edges():
    W = RNG.uniform(0, 1, (10, 10))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    W_thr = _threshold_proportional(W, 0.20)
    n_before = np.sum(W > 0) // 2
    n_after = np.sum(W_thr > 0) // 2
    assert n_after <= n_before


def test_invalid_input():
    with pytest.raises(ValueError):
        compute_graph_metrics(np.ones((3, 4)))
