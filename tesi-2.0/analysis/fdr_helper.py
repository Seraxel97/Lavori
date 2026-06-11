"""FDR correction helper per batch di p-value (Benjamini-Hochberg default).

Wrapper su `statsmodels.stats.multitest.multipletests`.
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests


def apply_fdr(
    p_values: np.ndarray | list[float],
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> tuple[np.ndarray, np.ndarray]:
    """Applica correzione FDR a un batch di p-value.

    Parameters
    ----------
    p_values:
        Array 1D di p-value in [0, 1]. Array vuoto ritorna tuple di array vuoti.
    alpha:
        Soglia di significativita' (default 0.05).
    method:
        Metodo di correzione passato a multipletests
        ('fdr_bh', 'fdr_by', 'bonferroni', ecc.).

    Returns
    -------
    p_corrected : np.ndarray, shape (n,)
        P-value corretti (adjusted).
    reject : np.ndarray[bool], shape (n,)
        Maschera True dove l'ipotesi nulla viene rigettata.

    Raises
    ------
    ValueError
        Se p_values contiene valori fuori [0, 1].
    """
    p_arr = np.asarray(p_values, dtype=float)
    if p_arr.ndim != 1:
        raise ValueError(f"p_values deve essere 1D, got shape {p_arr.shape}")
    if p_arr.size == 0:
        return np.array([], dtype=float), np.array([], dtype=bool)
    if np.any((p_arr < 0) | (p_arr > 1)):
        raise ValueError("p_values devono essere in [0, 1]")

    reject, p_corrected, _, _ = multipletests(p_arr, alpha=alpha, method=method)
    return p_corrected.astype(float), reject.astype(bool)
