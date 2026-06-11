"""Aperiodic (1/f) features via specparam (FOOOF).

Features per segnale:
  aperiodic_exp   — esponente χ (proxy E/I ratio)
  aperiodic_offset — offset b in log-log
  delta_power     — band 1-4 Hz (trapz su PSD)
  theta_power     — band 4-8 Hz
  alpha_power     — band 8-13 Hz
  beta_power      — band 13-30 Hz
  gamma_power     — band 30-45 Hz
  spectral_entropy — Shannon entropy su PSD normalizzato

NB: aperiodic_exp è inferenza proxy per E/I — dichiararlo in report.
"""

from __future__ import annotations

import warnings

import numpy as np
import specparam
from scipy.integrate import trapezoid
from scipy.stats import entropy as shannon_entropy

_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

FEATURE_NAMES = [
    "aperiodic_exp",
    "aperiodic_offset",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "spectral_entropy",
]


def _band_power(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if mask.sum() < 2:
        return np.nan
    return float(trapezoid(psd[mask], freqs[mask]))


def compute_aperiodic_features(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: tuple[float, float] = (1.0, 45.0),
) -> dict[str, float]:
    """Calcola feature aperiodiche su un singolo PSD.

    Parameters
    ----------
    freqs : shape (n_freqs,)
    psd   : shape (n_freqs,), valori di potenza (non in dB)
    freq_range : (f_min, f_max) per il fit FOOOF

    Returns
    -------
    dict con le 8 feature di FEATURE_NAMES
    """
    sm = specparam.SpectralModel(
        peak_width_limits=[1.0, 8.0],
        max_n_peaks=5,
        min_peak_height=0.05,
        verbose=False,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm.fit(freqs, psd, freq_range=list(freq_range))
        if sm.results.has_model:
            ap = sm.results.get_params("aperiodic")
            aperiodic_offset = float(ap[0])
            aperiodic_exp = float(ap[1])
        else:
            aperiodic_offset = np.nan
            aperiodic_exp = np.nan
    except Exception:
        aperiodic_offset = np.nan
        aperiodic_exp = np.nan

    band_powers = {
        name: _band_power(freqs, psd, lo, hi) for name, (lo, hi) in _BANDS.items()
    }

    psd_norm = psd / (psd.sum() + 1e-12)
    sp_entropy = float(shannon_entropy(psd_norm + 1e-12))

    return {
        "aperiodic_exp": aperiodic_exp,
        "aperiodic_offset": aperiodic_offset,
        "delta_power": band_powers["delta"],
        "theta_power": band_powers["theta"],
        "alpha_power": band_powers["alpha"],
        "beta_power": band_powers["beta"],
        "gamma_power": band_powers["gamma"],
        "spectral_entropy": sp_entropy,
    }


def compute_aperiodic_batch(
    freqs: np.ndarray,
    psd_matrix: np.ndarray,
    freq_range: tuple[float, float] = (1.0, 45.0),
) -> tuple[np.ndarray, list[str]]:
    """Processa un batch di PSD.

    Parameters
    ----------
    freqs      : shape (n_freqs,)
    psd_matrix : shape (n_signals, n_freqs)

    Returns
    -------
    X     : np.ndarray shape (n_signals, n_features)
    names : list[str] — FEATURE_NAMES
    """
    rows = []
    for psd in psd_matrix:
        feat = compute_aperiodic_features(freqs, psd, freq_range=freq_range)
        rows.append([feat[k] for k in FEATURE_NAMES])
    return np.array(rows, dtype=float), FEATURE_NAMES
