"""Test STEP 6 ML dispatcher — 5 algoritmi + GroupKFold + permutation."""

from __future__ import annotations

import numpy as np
import pytest

from ml_training.ml_dispatcher import Algorithm, CVResult, run_all_algorithms, run_cv
from ml_training.permutation import PermutationResult, fdr_correction, permutation_test

# ── dati sintetici ────────────────────────────────────────────────────────────
_RNG = np.random.default_rng(2)
_N_SAMPLES, _N_FEATURES = 60, 20

# Classi bilanciate
_Y = np.array([0] * 30 + [1] * 30)
# Segnale + rumore: primo feature discriminativo
_X = _RNG.standard_normal((_N_SAMPLES, _N_FEATURES))
_X[:30, 0] += 2.0  # rende il problema leggermente classificabile

# Groups: 6 soggetti × 10 epoch
_GROUPS = np.repeat(np.arange(6), 10)

_ALL_ALGOS: list[Algorithm] = ["logreg", "svm", "mlp", "rf", "gb"]


# ── run_cv ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("algo", _ALL_ALGOS)
def test_run_cv_returns_cvresult(algo: str) -> None:
    """run_cv ritorna CVResult con campi validi."""
    result = run_cv(_X, _Y, groups=_GROUPS, algorithm=algo, n_splits=3)  # type: ignore[arg-type]
    assert isinstance(result, CVResult)
    assert result.algorithm == algo
    assert len(result.ba_per_fold) == 3
    assert 0.0 <= result.ba_mean <= 1.0
    assert result.ba_std >= 0.0
    assert result.n_features == _N_FEATURES
    assert result.n_samples == _N_SAMPLES


@pytest.mark.parametrize("algo", _ALL_ALGOS)
def test_run_cv_ba_above_chance(algo: str) -> None:
    """BA su dati con segnale deve essere > 50% (chance level)."""
    result = run_cv(_X, _Y, groups=_GROUPS, algorithm=algo, n_splits=3)  # type: ignore[arg-type]
    # Con segnale robusto (feature 0 +2 sigma), tutti gli algo devono battere chance
    assert result.ba_mean > 0.5, f"{algo}: BA {result.ba_mean:.3f} <= 0.5 (chance)"


def test_run_cv_confusion_matrix_shape() -> None:
    """Confusion matrix aggregata deve avere shape (2, 2) per problema binario."""
    result = run_cv(_X, _Y, groups=_GROUPS, algorithm="logreg", n_splits=3)
    assert result.confusion.shape == (2, 2)


def test_run_cv_stratified_no_groups() -> None:
    """Senza groups usa StratifiedKFold."""
    result = run_cv(_X, _Y, groups=None, algorithm="logreg", n_splits=3)
    assert isinstance(result, CVResult)
    assert len(result.ba_per_fold) == 3


def test_run_cv_data_leakage_check() -> None:
    """Verifica che il scaler non sia fittato su test set (no data leakage).

    Proxy: BA con scaler corretto deve essere ≥ BA con dati randomizzati.
    """
    result = run_cv(_X, _Y, groups=_GROUPS, algorithm="logreg", n_splits=3)
    bad_result = run_cv(_RNG.permutation(_X), _Y, groups=_GROUPS, algorithm="logreg", n_splits=3)
    # Con dati corretti deve battere dati permutati (almeno in media)
    assert result.ba_mean >= bad_result.ba_mean - 0.15  # margine 15% per noise


# ── run_all_algorithms ────────────────────────────────────────────────────────


def test_run_all_algorithms_returns_all() -> None:
    """run_all_algorithms ritorna dict con tutti gli algoritmi."""
    results = run_all_algorithms(_X, _Y, groups=_GROUPS, n_splits=3)
    assert set(results.keys()) == {"logreg", "logreg_nested", "svm", "mlp", "rf", "gb"}


def test_run_all_algorithms_subset() -> None:
    """Subset di algoritmi."""
    results = run_all_algorithms(_X, _Y, groups=_GROUPS, algorithms=["logreg", "rf"], n_splits=3)
    assert set(results.keys()) == {"logreg", "rf"}


# ── permutation_test (piccolo n per velocità) ──────────────────────────────────


def test_permutation_test_structure() -> None:
    """permutation_test ritorna PermutationResult con struttura corretta."""
    result = permutation_test(
        _X, _Y, groups=_GROUPS, algorithm="logreg", n_permutations=20, n_splits=3
    )
    assert isinstance(result, PermutationResult)
    assert len(result.null_distribution) == 20
    assert 0.0 <= result.p_value <= 1.0
    assert result.observed_ba > 0.0


def test_permutation_test_null_is_near_chance() -> None:
    """La distribuzione null (label permutate) deve essere vicina a 0.5."""
    result = permutation_test(
        _X, _Y, groups=_GROUPS, algorithm="logreg", n_permutations=30, n_splits=3
    )
    # La media null non dovrebbe essere troppo lontana da 0.5
    assert abs(result.null_distribution.mean() - 0.5) < 0.2


# ── fdr_correction ────────────────────────────────────────────────────────────


def test_fdr_correction_all_significant() -> None:
    """P-value molto piccoli → tutti rifiutati."""
    p = np.array([0.001, 0.002, 0.003, 0.004])
    reject, p_corr = fdr_correction(p, alpha=0.05)
    assert reject.all()
    assert len(p_corr) == len(p)


def test_fdr_correction_none_significant() -> None:
    """P-value grandi → nessuno rifiutato."""
    p = np.array([0.8, 0.9, 0.7, 0.85])
    reject, p_corr = fdr_correction(p, alpha=0.05)
    assert not reject.any()


def test_fdr_correction_mixed() -> None:
    """Mix di p-value: solo i piccoli vengono rifiutati."""
    p = np.array([0.001, 0.5, 0.002, 0.8])
    reject, _ = fdr_correction(p, alpha=0.05)
    assert reject[0]  # p=0.001 → reject
    assert reject[2]  # p=0.002 → reject
    assert not reject[1]  # p=0.5 → no reject
    assert not reject[3]  # p=0.8 → no reject


def test_fdr_correction_output_shape() -> None:
    """Output shape uguale a input."""
    p = np.random.default_rng(5).uniform(0, 1, 100)
    reject, p_corr = fdr_correction(p)
    assert reject.shape == p.shape
    assert p_corr.shape == p.shape


# ── logreg_nested (FIX-02) ────────────────────────────────────────────────────


def test_nested_cv_picks_best_C() -> None:
    """logreg_nested: run_cv deve restituire best_C_per_fold non vuoto."""
    rng = np.random.default_rng(42)
    n_sub = 9
    n_ep = 4
    n_feat = 10
    n = n_sub * n_ep

    X = rng.standard_normal((n, n_feat))
    y = np.array([i % 2 for i in range(n_sub)] * n_ep)
    groups = np.repeat(np.arange(n_sub), n_ep)

    result = run_cv(X, y, groups=groups, algorithm="logreg_nested", n_splits=n_sub)  # type: ignore[arg-type]

    assert len(result.best_C_per_fold) == n_sub, (
        f"best_C_per_fold atteso len={n_sub}, trovato {len(result.best_C_per_fold)}"
    )
    assert all(c in [0.01, 0.1, 1.0, 10.0, 100.0] for c in result.best_C_per_fold), (
        f"C non in grid: {result.best_C_per_fold}"
    )
    assert 0.0 <= result.ba_mean <= 1.0
