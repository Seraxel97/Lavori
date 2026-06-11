"""Test step 1.4: ML sesso — leakage check + sanity strutturale."""

from __future__ import annotations

import numpy as np
import pytest

from ml_training.ml_sex import load_data, run_cv


@pytest.fixture(scope="module")
def data():
    return load_data()


def test_data_shape(data):
    X, y_sex, groups = data
    assert X.shape[0] == len(y_sex) == len(groups)
    assert X.shape[0] == 200  # 100 soggetti × 2 condizioni


def test_y_sex_binary(data):
    _, y_sex, _ = data
    assert set(y_sex.tolist()).issubset({0, 1})


def test_groups_aligned(data):
    X, y_sex, groups = data
    # Stesso soggetto deve avere stesso sesso nelle due righe
    for g in np.unique(groups):
        mask = groups == g
        assert (
            len(np.unique(y_sex[mask])) == 1
        ), f"Soggetto {g} ha sessi diversi!"


def test_shuffle_leakage():
    """Shuffle y_sex → BA deve essere vicino a 0.5 (no leakage)."""
    X, y_sex, groups = load_data()
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y_sex)
    res = run_cv(X, y_shuf, groups, "logreg", outer_k=5)
    assert res["ba_mean"] < 0.65, f"BA shuffle={res['ba_mean']:.3f} — possibile leakage!"


def test_cv_result_keys():
    X, y_sex, groups = load_data()
    res = run_cv(X, y_sex, groups, "logreg", outer_k=5)
    assert "ba_mean" in res
    assert "ba_folds" in res
    assert len(res["ba_folds"]) == 5
