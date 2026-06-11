"""S-ML-EXTEND — Tests for extended 6-classifier aggregate_classify_n15.

Test suite:
  - test_six_classifiers_smoke      : 6 clf on mock N=15 → 6 entries, all BA finite
  - test_six_classifiers_all_pass   : all 6 clf produce valid p_perm ∈ [0, 1]
  - test_multi_band_support         : bands=["alpha","beta"] → 2× results vs single band
  - test_no_regression_original_3   : logreg/svm_rbf/lda results unchanged at seed=42
  - test_trigger_guard_x_files      : helper to check X-file count threshold logic
"""

from __future__ import annotations

import numpy as np

from ml_training.aggregate_n15 import (
    _N_SUBJECTS_FULL,
    _build_pipeline,
    aggregate_classify_n15,
)

_N_SUB = _N_SUBJECTS_FULL  # 15
_N_SAMP = _N_SUB * 2       # 30
_N_FEAT = 60


def _make_features(tmp_path, *, band: str = "alpha", separable: bool = False):
    """Write synthetic X_aparc_coh_{band}.npz + y.npy + groups.npy."""
    rng = np.random.default_rng(42)
    y = np.array([i % 2 for i in range(_N_SAMP)], dtype=int)
    if separable:
        X = np.zeros((_N_SAMP, _N_FEAT))
        X[y == 0, 0] = 1.0
        X[y == 1, 0] = -1.0
    else:
        X = rng.standard_normal((_N_SAMP, _N_FEAT))
    groups = np.repeat(np.arange(_N_SUB), 2)
    np.savez_compressed(tmp_path / f"X_aparc_coh_{band}.npz", X=X)
    np.save(tmp_path / "y.npy", y)
    np.save(tmp_path / "groups.npy", groups)


# ---------------------------------------------------------------------------
# Smoke — 6 classifiers
# ---------------------------------------------------------------------------

def test_six_classifiers_smoke(tmp_path):
    """6 classifier on mock N=15 → exactly 6 entries, all ba_mean finite."""
    _make_features(tmp_path)
    out = tmp_path / "results"

    result = aggregate_classify_n15(
        features_dir=tmp_path,
        out_dir=out,
        atlases=["aparc"],
        metrics=["coh"],
        band="alpha",
        classifiers=["logreg", "svm_rbf", "lda", "rf", "mlp", "gb"],
        n_permutations=20,
        random_state=0,
    )

    assert result["winner"] is not None
    assert len(result["results"]) == 6, f"Attesi 6 risultati, trovati {len(result['results'])}"
    for r in result["results"]:
        assert np.isfinite(r["ba_mean"]), f"ba_mean non finito per {r['classifier']}"
        assert r["classifier"] in ["logreg", "svm_rbf", "lda", "rf", "mlp", "gb"]


def test_six_classifiers_all_pass(tmp_path):
    """All 6 classifiers produce p_perm ∈ [0, 1] and CI with ci_lo ≤ ci_hi."""
    _make_features(tmp_path)
    out = tmp_path / "results"

    result = aggregate_classify_n15(
        features_dir=tmp_path,
        out_dir=out,
        atlases=["aparc"],
        metrics=["coh"],
        classifiers=["logreg", "svm_rbf", "lda", "rf", "mlp", "gb"],
        n_permutations=20,
        random_state=1,
    )

    for r in result["results"]:
        assert 0.0 <= r["p_perm"] <= 1.0, f"{r['classifier']}: p_perm={r['p_perm']}"
        assert r["ci_lo"] <= r["ci_hi"], f"{r['classifier']}: CI inverted"


# ---------------------------------------------------------------------------
# Multi-band support
# ---------------------------------------------------------------------------

def test_multi_band_support(tmp_path):
    """bands=['alpha','beta'] → 2× results compared to single band."""
    _make_features(tmp_path, band="alpha")
    _make_features(tmp_path, band="beta")
    out = tmp_path / "results"

    r_alpha = aggregate_classify_n15(
        features_dir=tmp_path, out_dir=out,
        atlases=["aparc"], metrics=["coh"],
        band="alpha",
        classifiers=["logreg"],
        n_permutations=10,
    )
    n_single = len(r_alpha["results"])

    r_both = aggregate_classify_n15(
        features_dir=tmp_path, out_dir=out,
        atlases=["aparc"], metrics=["coh"],
        bands=["alpha", "beta"],
        classifiers=["logreg"],
        n_permutations=10,
        random_state=42,
    )
    n_multi = len(r_both["results"])

    assert n_multi == 2 * n_single, (
        f"multi-band devrebbe produrre {2*n_single} risultati, trovati {n_multi}"
    )
    bands_in_results = {r["band"] for r in r_both["results"]}
    assert bands_in_results == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# No-regression: original 3 classifiers unchanged at seed=42
# ---------------------------------------------------------------------------

def test_no_regression_original_3(tmp_path):
    """logreg/svm_rbf/lda: ba_mean matches between 3-clf and 6-clf runs (same seed)."""
    _make_features(tmp_path)
    out3 = tmp_path / "out3"
    out6 = tmp_path / "out6"

    r3 = aggregate_classify_n15(
        features_dir=tmp_path, out_dir=out3,
        atlases=["aparc"], metrics=["coh"],
        classifiers=["logreg", "svm_rbf", "lda"],
        n_permutations=20, random_state=42,
    )
    r6 = aggregate_classify_n15(
        features_dir=tmp_path, out_dir=out6,
        atlases=["aparc"], metrics=["coh"],
        classifiers=["logreg", "svm_rbf", "lda", "rf", "mlp", "gb"],
        n_permutations=20, random_state=42,
    )

    ba3 = {r["classifier"]: r["ba_mean"] for r in r3["results"]}
    ba6 = {r["classifier"]: r["ba_mean"] for r in r6["results"]}

    for clf in ["logreg", "svm_rbf", "lda"]:
        assert ba3[clf] == ba6[clf], (
            f"{clf}: ba_mean diverge tra 3-clf ({ba3[clf]}) e 6-clf ({ba6[clf]})"
        )


# ---------------------------------------------------------------------------
# _build_pipeline — rf, mlp, gb
# ---------------------------------------------------------------------------

def test_build_pipeline_rf():
    from sklearn.ensemble import RandomForestClassifier
    pipe = _build_pipeline("rf")
    assert any(isinstance(s, RandomForestClassifier) for _, s in pipe.steps)


def test_build_pipeline_mlp():
    from sklearn.neural_network import MLPClassifier
    pipe = _build_pipeline("mlp")
    assert any(isinstance(s, MLPClassifier) for _, s in pipe.steps)


def test_build_pipeline_gb():
    from sklearn.ensemble import GradientBoostingClassifier
    pipe = _build_pipeline("gb")
    assert any(isinstance(s, GradientBoostingClassifier) for _, s in pipe.steps)
