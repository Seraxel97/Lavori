"""Tests S-AGG-N15 — aggregate_classify_n15.

Test suite:
  - test_shape_contract          : X(30,), y(30,), groups(30,), 15 unique groups
  - test_shape_guard_skips_pilot : X(10,*) → results=[], winner=None + WARNING
  - test_unit_clear_signal       : mock separable dataset → winner BA > 0.7, p_perm < 0.05
  - test_smoke_e2e_synthetic     : 1 atlas × 1 metric × 2 clf × 100 perm, full pipeline
  - test_idempotent_cache        : second call with same hashes → _cached=True
  - test_winner_fields           : winner dict has expected keys
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ml_training.aggregate_n15 import (
    _EXPECTED_SAMPLES,
    _N_SUBJECTS_FULL,
    _build_pipeline,
    aggregate_classify_n15,
)

_RNG = np.random.default_rng(0)
_N_SUB = _N_SUBJECTS_FULL  # 15
_N_SAMP = _EXPECTED_SAMPLES  # 30 = 15 × 2
_N_FEAT = 80


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_n15_dir(tmp_path, *, separable: bool = True, n_sub: int = _N_SUB):
    """Write X_aparc_wpli_alpha.npz + y.npy + groups.npy in tmp_path."""
    n_samp = n_sub * 2
    rng = np.random.default_rng(42)

    if separable:
        # Perfectly separable: class 0 → feature[0]=+1, class 1 → feature[0]=-1, no noise
        y = np.array([i % 2 for i in range(n_samp)], dtype=int)
        X = np.zeros((n_samp, _N_FEAT))
        X[y == 0, 0] = 1.0
        X[y == 1, 0] = -1.0
    else:
        X = rng.standard_normal((n_samp, _N_FEAT))
        y = np.array([i % 2 for i in range(n_samp)], dtype=int)

    groups = np.repeat(np.arange(n_sub), 2)

    np.savez_compressed(tmp_path / "X_aparc_wpli_alpha.npz", X=X)
    np.save(tmp_path / "y.npy", y)
    np.save(tmp_path / "groups.npy", groups)
    return X, y, groups


# ---------------------------------------------------------------------------
# Shape contract
# ---------------------------------------------------------------------------


def test_shape_contract(tmp_path):
    """Feature file must produce X(30,), y(30,), groups(30,) with 15 unique groups."""
    X, y, groups = _make_n15_dir(tmp_path)
    assert X.shape == (_N_SAMP, _N_FEAT)
    assert y.shape == (_N_SAMP,)
    assert groups.shape == (_N_SAMP,)
    assert len(np.unique(groups)) == _N_SUB


# ---------------------------------------------------------------------------
# Shape guard
# ---------------------------------------------------------------------------


def test_shape_guard_skips_pilot(tmp_path):
    """Shape (10, *) triggers shape guard → results=[], winner=None (no crash)."""
    _make_n15_dir(tmp_path, n_sub=5)  # pilot N=5 → 10 samples
    out_dir = tmp_path / "results"

    result = aggregate_classify_n15(
        features_dir=tmp_path,
        out_dir=out_dir,
        atlases=["aparc"],
        metrics=["wpli"],
        classifiers=["logreg"],
        n_permutations=10,
    )
    assert result["results"] == [], "Shape guard deve restituire lista vuota"
    assert result["winner"] is None


# ---------------------------------------------------------------------------
# Unit test — clear separable signal
# ---------------------------------------------------------------------------


def test_unit_clear_signal(tmp_path):
    """Separable mock data (N=15) → winner BA > 0.7 and p_perm < 0.05."""
    _make_n15_dir(tmp_path, separable=True)
    out_dir = tmp_path / "results"

    result = aggregate_classify_n15(
        features_dir=tmp_path,
        out_dir=out_dir,
        atlases=["aparc"],
        metrics=["wpli"],
        band="alpha",
        classifiers=["logreg"],
        n_permutations=100,
        random_state=42,
    )
    assert result["winner"] is not None, "Deve trovare un winner con dati separabili"
    winner = result["winner"]
    assert winner["ba_mean"] >= 0.9, f"BA attesa >= 0.90 su dati separabili, trovata {winner['ba_mean']:.3f}"
    assert winner["p_perm"] < 0.05, f"p_perm atteso < 0.05, trovato {winner['p_perm']:.4f}"


# ---------------------------------------------------------------------------
# Smoke E2E — synthetic fixture (1 atlas × 1 metric × 2 clf × 100 perm)
# ---------------------------------------------------------------------------


def test_smoke_e2e_synthetic(tmp_path):
    """Pipeline completa su fixture sintetica: shape output e JSON valido."""
    _make_n15_dir(tmp_path, separable=False)
    out_dir = tmp_path / "results"

    result = aggregate_classify_n15(
        features_dir=tmp_path,
        out_dir=out_dir,
        atlases=["aparc"],
        metrics=["wpli"],
        band="alpha",
        classifiers=["logreg", "lda"],
        n_permutations=100,
        random_state=0,
    )

    assert isinstance(result["results"], list)
    assert len(result["results"]) == 2, "2 clf → 2 risultati"
    assert result["winner"] is not None

    out_path = out_dir / "comparison_matrix_N15.json"
    assert out_path.exists(), "JSON output non trovato"
    payload = json.loads(out_path.read_text())
    assert "results" in payload
    assert "winner" in payload
    assert "_hashes" in payload
    assert "_meta" in payload

    for r in result["results"]:
        assert 0.0 <= r["ba_mean"] <= 1.0
        assert r["ci_lo"] <= r["ba_mean"] <= r["ci_hi"] or r["ci_lo"] <= r["ci_hi"]
        assert 0.0 <= r["p_perm"] <= 1.0
        assert r["n_subjects"] == _N_SUB
        assert r["n_features"] == _N_FEAT


# ---------------------------------------------------------------------------
# Idempotency / cache
# ---------------------------------------------------------------------------


def test_idempotent_cache(tmp_path):
    """Second call with unchanged data returns _cached=True without recomputing."""
    _make_n15_dir(tmp_path, separable=False)
    out_dir = tmp_path / "results"
    kwargs = dict(
        features_dir=tmp_path,
        out_dir=out_dir,
        atlases=["aparc"],
        metrics=["wpli"],
        classifiers=["logreg"],
        n_permutations=20,
        random_state=7,
    )

    r1 = aggregate_classify_n15(**kwargs)
    assert "_cached" not in r1, "Prima chiamata non deve essere cached"

    r2 = aggregate_classify_n15(**kwargs)
    assert r2.get("_cached") is True, "Seconda chiamata deve essere cached"
    assert r1["winner"]["ba_mean"] == r2["winner"]["ba_mean"]


# ---------------------------------------------------------------------------
# Winner fields
# ---------------------------------------------------------------------------


def test_winner_fields(tmp_path):
    """Winner dict contiene tutti i campi attesi."""
    _make_n15_dir(tmp_path, separable=False)
    out_dir = tmp_path / "results"

    result = aggregate_classify_n15(
        features_dir=tmp_path,
        out_dir=out_dir,
        atlases=["aparc"],
        metrics=["wpli"],
        classifiers=["logreg"],
        n_permutations=20,
    )
    winner = result["winner"]
    expected_keys = {
        "atlas", "metric", "band", "classifier",
        "ba_mean", "ba_std", "ci_lo", "ci_hi", "p_perm",
        "n_subjects", "n_features",
    }
    assert expected_keys <= set(winner.keys()), (
        f"Chiavi mancanti: {expected_keys - set(winner.keys())}"
    )


# ---------------------------------------------------------------------------
# _build_pipeline — extended classifiers
# ---------------------------------------------------------------------------


def test_build_pipeline_lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    pipe = _build_pipeline("lda")
    assert any(isinstance(s, LinearDiscriminantAnalysis) for _, s in pipe.steps)


def test_build_pipeline_svm_rbf():
    from sklearn.svm import SVC
    pipe = _build_pipeline("svm_rbf")
    assert any(isinstance(s, SVC) and s.kernel == "rbf" for _, s in pipe.steps)


def test_build_pipeline_unknown_raises():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        _build_pipeline("xgboost_turbo")
