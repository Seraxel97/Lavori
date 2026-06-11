"""Test step 1.5: ML età — sanity checks + leakage check."""

from __future__ import annotations

import numpy as np
import pytest

from ml_training.ml_age import load_data, run_cv


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_data_shape(data):
    X, y_age, groups = data
    assert X.shape[0] == len(y_age) == len(groups) == 200  # 100 soggetti × 2 cond


def test_y_age_range(data):
    _, y_age, _ = data
    assert y_age.min() >= 18.0
    assert y_age.max() <= 100.0


def test_groups_aligned(data):
    X, y_age, groups = data
    # Stesso soggetto deve avere stessa età nelle due righe (EO e EC)
    for g in np.unique(groups):
        mask = groups == g
        ages = y_age[mask]
        assert np.all(ages == ages[0]), f"Soggetto {g} ha età diverse: {ages}"


def test_cv_keys():
    X, y_age, groups = load_data()
    res = run_cv(X, y_age, groups, "dummy", outer_k=5)
    for key in ("mae", "r2", "brain_age_gap", "mae_ci95", "r2_ci95"):
        assert key in res


def test_dummy_r2_leq_zero():
    """Dummy deve avere R²≈0 (non migliora su media)."""
    X, y_age, groups = load_data()
    res = run_cv(X, y_age, groups, "dummy", outer_k=5)
    # DummyRegressor predice la mean del train fold → R² ≈ 0 sul test
    assert res["r2"] <= 0.1, f"Dummy R²={res['r2']:.3f} insolitamente alto"


def test_shuffle_leakage():
    """Shuffle y_age → R² deve essere basso (no data leakage)."""
    X, y_age, groups = load_data()
    rng = np.random.default_rng(99)
    y_shuf = rng.permutation(y_age)
    res = run_cv(X, y_shuf, groups, "elasticnet", outer_k=5)
    assert res["r2"] < 0.3, f"R² su shuffle={res['r2']:.3f} — possibile leakage!"
