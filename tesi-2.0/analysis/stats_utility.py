"""Statistical analysis utilities — bootstrap CI, Cohen's d, power analysis.

Funzioni:
    bootstrap_ci(values, statistic, n_boot, alpha) -> (lo, hi)
    cohen_d(a, b)                                 -> float
    statistical_power(effect_size, n, alpha)       -> float
    format_pvalue(p_value, n_perm)                -> str
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from statsmodels.stats.power import TTestIndPower


def bootstrap_ci(
    values: np.ndarray | list[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    alpha: float = 0.05,
    *,
    random_state: int = 42,
) -> tuple[float, float]:
    """Calcola un intervallo di confidenza bootstrap (BCa-style percentile).

    Parameters
    ----------
    values:
        Array 1D di valori campionati (es. balanced accuracy per fold).
    statistic:
        Funzione scalare applicata a ciascun bootstrap sample (default: np.mean).
    n_boot:
        Numero di campioni bootstrap (default: 1000).
    alpha:
        Livello di significatività; l'IC è (1 - alpha) * 100% (default: 0.05 → 95%).
    random_state:
        Seed per riproducibilità.

    Returns
    -------
    (lo, hi) : tuple[float, float]
        Estremi inferiore e superiore dell'intervallo di confidenza.

    Notes
    -----
    Usa il metodo percentile standard (non BCa corretto): semplice e robusto
    per n_boot >= 1000. Per statistiche con forte asimmetria, considerare BCa
    (disponibile in `scipy.stats.bootstrap`).
    """
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_state)
    boot_stats = np.array(
        [statistic(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    )
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


def bootstrap_ci_bca(
    values: np.ndarray | list[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    *,
    random_state: int = 42,
) -> tuple[float, float]:
    """Intervallo di confidenza bootstrap BCa (Bias-Corrected accelerated).

    Implementa Efron & Tibshirani (1993) con jackknife per stima accelerazione.

    Parameters
    ----------
    values:
        Array 1D di valori campionati.
    statistic:
        Funzione scalare (default: np.mean).
    n_boot:
        Numero campioni bootstrap (default: 2000).
    alpha:
        Livello significatività (default: 0.05 → CI 95%).
    random_state:
        Seed riproducibilità.

    Returns
    -------
    (lo, hi) : tuple[float, float]

    Notes
    -----
    BCa corregge bias e asimmetria della distribuzione bootstrap.
    Superiore al metodo percentile per statistiche asimmetriche (N<50).
    Ref: Efron & Tibshirani (1993); DiCiccio & Romano (1988).
    """
    from scipy.stats import norm as _norm

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(random_state)

    # Bootstrap distribution
    boot_stats = np.array([statistic(rng.choice(arr, size=n, replace=True)) for _ in range(n_boot)])
    theta_hat = statistic(arr)

    # Bias correction z0
    prop_less = np.mean(boot_stats < theta_hat)
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = float(_norm.ppf(prop_less))

    # Acceleration via jackknife
    jack_stats = np.array([statistic(np.delete(arr, i)) for i in range(n)])
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = float(num / den) if abs(den) > 1e-15 else 0.0

    # Adjusted percentiles
    z_lo = _norm.ppf(alpha / 2)
    z_hi = _norm.ppf(1 - alpha / 2)

    def _adj_pct(z_alpha: float) -> float:
        num_inner = z0 + z_alpha
        denom_inner = 1.0 - a * (z0 + z_alpha)
        if abs(denom_inner) < 1e-15:
            return z_alpha
        return float(_norm.cdf(z0 + num_inner / denom_inner))

    pct_lo = _adj_pct(z_lo) * 100
    pct_hi = _adj_pct(z_hi) * 100
    pct_lo = np.clip(pct_lo, 0.01, 99.99)
    pct_hi = np.clip(pct_hi, 0.01, 99.99)

    lo = float(np.percentile(boot_stats, pct_lo))
    hi = float(np.percentile(boot_stats, pct_hi))
    return lo, hi


def cohen_d(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Calcola Cohen's d per due campioni indipendenti.

    Usa la deviazione standard pooled non corretta (standard per confronti
    tra due gruppi di dimensione comparabile):

        d = (mean_a - mean_b) / s_pooled
        s_pooled = sqrt((var_a + var_b) / 2)

    Parameters
    ----------
    a:
        Primo campione (es. BA gruppo 1).
    b:
        Secondo campione (es. BA gruppo 2).

    Returns
    -------
    float
        Cohen's d (positivo se mean_a > mean_b). Interpretazione convenzionale:
        |d| < 0.2 trascurabile, 0.2–0.5 piccolo, 0.5–0.8 medio, > 0.8 grande.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    mean_diff = float(np.mean(a_arr) - np.mean(b_arr))
    s_pooled = float(np.sqrt((np.var(a_arr, ddof=0) + np.var(b_arr, ddof=0)) / 2))
    if s_pooled == 0.0:
        return 0.0 if mean_diff == 0.0 else float("inf")
    return mean_diff / s_pooled


def hedges_g(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    """Calcola Hedges' g — versione bias-corrected di Cohen's d.

    Raccomandato per N < 50 (campioni piccoli) dove Cohen's d tende a sovrastimare
    l'effect size. La correzione è:

        g = d × (1 − 3 / (4(n1 + n2) − 9))

    dove d è Cohen's d con pooled SD non corretta.

    Parameters
    ----------
    a:
        Primo campione.
    b:
        Secondo campione.

    Returns
    -------
    float
        Hedges' g (|g| < |d| per N piccolo; converge a d per N grande).

    Notes
    -----
    Per N1+N2 >= 10 la correzione è calcolabile. Per campioni molto piccoli
    (N1+N2 < 10) la correzione può divergere; in quel caso restituisce d.

    References
    ----------
    Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect
    size and related estimators. Journal of Educational Statistics, 6(2), 107–128.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    n1, n2 = len(a_arr), len(b_arr)
    d = cohen_d(a_arr, b_arr)
    denom = 4 * (n1 + n2) - 9
    if denom <= 0:
        return d  # campioni troppo piccoli, fallback a d
    correction = 1.0 - 3.0 / denom
    return d * correction


def format_pvalue(p_value: float, n_perm: int) -> str:
    """Formatta un p-value empirico da permutation test in modo onesto.

    Quando nessuna permutazione supera il valore osservato (p_value == 0.0),
    restituisce ``"p < 1/n_perm"`` (es. ``"p < 0.001"`` per n_perm=1000) invece
    di ``"p=0.0"`` che sarebbe fuorviante (risoluzione finita del test).

    Parameters
    ----------
    p_value:
        P-value empirico calcolato come ``#{null >= obs} / (n_perm + 1)`` o
        ``#{null >= obs} / n_perm``. Deve essere in [0, 1].
    n_perm:
        Numero di permutazioni eseguite. Determina la risoluzione minima
        rappresentabile.

    Returns
    -------
    str
        Stringa formattata:
        - ``"p < 0.001"`` (o il valore ``1/n_perm`` appropriato) se p == 0.0
        - ``"p = 0.XXX"`` altrimenti (3 cifre decimali)

    Examples
    --------
    >>> format_pvalue(0.0, 1000)
    'p < 0.001'
    >>> format_pvalue(0.0, 500)
    'p < 0.002'
    >>> format_pvalue(0.042, 1000)
    'p = 0.042'
    >>> format_pvalue(0.001, 1000)
    'p = 0.001'
    """
    if p_value == 0.0:
        threshold = 1.0 / max(n_perm, 1)
        return f"p < {threshold:.3f}"
    return f"p = {p_value:.3f}"


def statistical_power(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
    *,
    alternative: str = "two-sided",
) -> float:
    """Calcola la potenza statistica per un t-test indipendente a due campioni.

    Usa `statsmodels.stats.power.TTestIndPower` con rapporto di campioni k=1
    (gruppi bilanciati). Restituisce la probabilità di rifiutare correttamente
    H0 dato l'effect size e il numero di soggetti per gruppo.

    Parameters
    ----------
    effect_size:
        Cohen's d (dimensione dell'effetto atteso).
    n:
        Numero di osservazioni per gruppo (campioni bilanciati).
    alpha:
        Livello di significatività (default: 0.05).
    alternative:
        Tipo di test: 'two-sided', 'larger', 'smaller' (default: 'two-sided').

    Returns
    -------
    float
        Potenza statistica in [0, 1]. Convenzionalmente, power >= 0.80 è
        considerato adeguato per disegni sperimentali.

    Examples
    --------
    >>> statistical_power(0.5, 50)  # effetto medio, 50 soggetti/gruppo
    0.697...
    >>> statistical_power(0.8, 50)  # effetto grande
    0.961...
    """
    analysis = TTestIndPower()
    power = analysis.solve_power(
        effect_size=effect_size,
        nobs1=n,
        alpha=alpha,
        power=None,
        ratio=1.0,
        alternative=alternative,
    )
    p = float(power)
    # statsmodels può restituire NaN per combinazioni effect_size/n molto grandi
    # dove la potenza teorica è ≈1.0 (overflow numerico interno)
    if np.isnan(p):
        return 1.0
    return max(0.0, min(1.0, p))
