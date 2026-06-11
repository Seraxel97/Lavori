"""Test ml_runner_no_leakage — GroupKFold assertion enforcement.

Verifica:
  - _assert_no_leakage() lancia RuntimeError su leakage intenzionale
  - run_classification() completa senza leakage su dati reali ridotti
  - run_regression() completa senza leakage su dati reali ridotti
  - n_subjects nei risultati corrisponde a unique(groups)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.components.ml_runner import (  # noqa: E402
    _assert_no_leakage,
    run_classification,
    run_regression,
)


# ---------------------------------------------------------------------------
# Test _assert_no_leakage diretto
# ---------------------------------------------------------------------------
def test_assert_no_leakage_ok():
    groups = np.array([0, 0, 1, 1, 2, 2])
    train_idx = np.array([0, 1, 2, 3])  # soggetti 0 e 1
    test_idx = np.array([4, 5])          # soggetto 2
    _assert_no_leakage(groups, train_idx, test_idx)  # non deve lanciare


def test_assert_no_leakage_raises_on_overlap():
    groups = np.array([0, 0, 1, 1, 2, 2])
    train_idx = np.array([0, 1, 2, 3])  # soggetti 0 e 1
    test_idx = np.array([0, 4])          # soggetto 0 anche in test → leakage
    with pytest.raises(RuntimeError, match="Subject leakage"):
        _assert_no_leakage(groups, train_idx, test_idx)


# ---------------------------------------------------------------------------
# Test run_classification con dati sintetici
# ---------------------------------------------------------------------------
@pytest.fixture()
def synthetic_clf_data():
    """Dati sintetici N=20 soggetti (2 campioni each), classificazione binaria."""
    rng = np.random.default_rng(42)
    n_sub = 20
    n_feat = 10
    groups = np.repeat(np.arange(n_sub), 2)  # 40 campioni
    y = np.array([i % 2 for i in range(n_sub * 2)])  # alternato 0/1
    X = rng.standard_normal((n_sub * 2, n_feat))
    return X, y, groups


@pytest.fixture()
def synthetic_reg_data():
    """Dati sintetici N=15 soggetti (2 campioni each), regressione continua."""
    rng = np.random.default_rng(42)
    n_sub = 15
    n_feat = 8
    groups = np.repeat(np.arange(n_sub), 2)
    y = rng.uniform(20, 80, size=n_sub * 2)
    X = rng.standard_normal((n_sub * 2, n_feat))
    return X, y, groups


def test_run_classification_no_leakage(synthetic_clf_data):
    X, y, groups = synthetic_clf_data
    result = run_classification(X, y, groups, clf_name="logreg", n_perm=0, n_boot=50)
    assert "ba" in result
    assert 0.0 <= result["ba"] <= 1.0
    assert result["n_subjects"] == len(np.unique(groups))
    assert len(result["ba_per_fold"]) == len(np.unique(groups))


def test_run_classification_ci_ordered(synthetic_clf_data):
    X, y, groups = synthetic_clf_data
    result = run_classification(X, y, groups, clf_name="logreg", n_perm=0, n_boot=100)
    assert result["ci_lo"] <= result["ba"] <= result["ci_hi"], (
        f"CI non contiene BA: [{result['ci_lo']:.3f}, {result['ci_hi']:.3f}] vs {result['ba']:.3f}"
    )


def test_run_classification_with_perm(synthetic_clf_data):
    X, y, groups = synthetic_clf_data
    result = run_classification(X, y, groups, clf_name="logreg", n_perm=10, n_boot=50)
    assert result["p_perm"] is not None
    assert 0.0 < result["p_perm"] <= 1.0
    assert len(result["null_distribution"]) == 10


def test_run_regression_no_leakage(synthetic_reg_data):
    X, y, groups = synthetic_reg_data
    result = run_regression(X, y, groups, reg_name="ridge", n_perm=0, n_boot=50)
    assert "mae" in result
    assert result["mae"] >= 0.0
    assert result["n_subjects"] == len(np.unique(groups))


def test_run_regression_ci_ordered(synthetic_reg_data):
    X, y, groups = synthetic_reg_data
    result = run_regression(X, y, groups, reg_name="ridge", n_perm=0, n_boot=100)
    assert result["ci_mae_lo"] <= result["mae"] <= result["ci_mae_hi"], (
        f"CI non contiene MAE: [{result['ci_mae_lo']:.2f}, {result['ci_mae_hi']:.2f}] "
        f"vs {result['mae']:.2f}"
    )


def test_run_classification_groupkfold_n_folds(synthetic_clf_data):
    """Il numero di fold deve essere uguale al numero di soggetti unici."""
    X, y, groups = synthetic_clf_data
    n_unique = len(np.unique(groups))
    result = run_classification(X, y, groups, clf_name="logreg", n_perm=0, n_boot=20)
    assert len(result["ba_per_fold"]) == n_unique, (
        f"Atteso {n_unique} fold, trovato {len(result['ba_per_fold'])}"
    )
