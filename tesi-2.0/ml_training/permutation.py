"""
STEP 6b — Permutation testing per balanced accuracy.

Approccio: permuta le label y ``n_permutations`` volte, esegue CV ad ogni permutazione,
confronta la distribuzione null con la BA osservata → p-value empirico.

FDR correction (Benjamini-Hochberg) disponibile per test multipli (multi-config).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_training.ml_dispatcher import Algorithm, run_cv


@dataclass
class PermutationResult:
    """Risultato di un permutation test.

    Attributes
    ----------
    observed_ba:
        Balanced accuracy osservata (media folds).
    null_distribution:
        Array (n_permutations,) di BA null (permutate).
    p_value:
        p-value empirico = P(null >= observed_ba).
    n_permutations:
        Numero di permutazioni eseguite.
    """

    observed_ba: float
    null_distribution: np.ndarray
    p_value: float
    n_permutations: int


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    algorithm: Algorithm = "logreg",
    *,
    n_permutations: int = 1000,
    n_splits: int = 5,
    random_state: int = 42,
) -> PermutationResult:
    """Esegue permutation test sulla balanced accuracy.

    Parameters
    ----------
    X:
        Feature matrix (n_samples, n_features).
    y:
        Label array (n_samples,).
    groups:
        Group IDs per GroupKFold.
    algorithm:
        Classificatore da usare.
    n_permutations:
        Numero di permutazioni (default: 1000).
    n_splits:
        Fold per CV ad ogni permutazione.
    random_state:
        Seed per riproducibilità.

    Returns
    -------
    PermutationResult con distribuzione null e p-value.
    """
    rng = np.random.default_rng(random_state)

    # BA osservata
    obs_result = run_cv(X, y, groups=groups, algorithm=algorithm, n_splits=n_splits)
    observed_ba = obs_result.ba_mean

    # Distribuzione null
    null_bas = np.zeros(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        perm_result = run_cv(
            X, y_perm, groups=groups, algorithm=algorithm, n_splits=n_splits
        )
        null_bas[i] = perm_result.ba_mean

    p_value = float(np.mean(null_bas >= observed_ba))

    return PermutationResult(
        observed_ba=observed_ba,
        null_distribution=null_bas,
        p_value=p_value,
        n_permutations=n_permutations,
    )


def fdr_correction(
    p_values: np.ndarray | list[float],
    *,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Correzione FDR Benjamini-Hochberg per test multipli.

    Parameters
    ----------
    p_values:
        Array di p-value grezzi.
    alpha:
        Livello di significatività FDR (default: 0.05).

    Returns
    -------
    reject : np.ndarray bool
        True dove l'ipotesi nulla viene rifiutata dopo FDR.
    p_corrected : np.ndarray
        P-value corretti (adjusted).
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    rank = np.arange(1, n + 1)

    # BH threshold: p_i <= alpha * i/n (ordinate)
    threshold = alpha * rank / n
    p_sorted = p_values[order]

    # Trova il massimo k per cui p_k <= threshold_k
    cummax = np.maximum.accumulate((p_sorted <= threshold)[::-1])[::-1]

    reject = np.zeros(n, dtype=bool)
    reject[order] = cummax

    # P-value corretti: p_adj_i = min(p_i * n/rank_i, 1) nel senso BH
    p_corrected = np.zeros(n)
    p_corrected[order] = np.minimum(1.0, p_sorted * n / rank)
    # Forza monotonia (BH standard): step-up
    p_corrected[order] = np.minimum.accumulate(p_corrected[order][::-1])[::-1]

    return reject, p_corrected
