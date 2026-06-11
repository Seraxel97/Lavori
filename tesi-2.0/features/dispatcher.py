"""
STEP 5 — Feature dispatcher: univariate EEG + FC-flatten → matrice X per ML.

Due sorgenti di feature:
  1. **Univariate** (mne-features): per ogni epoch, per ogni ROI label.
     Funzioni di default: variance, mean, std, kurtosis, hjorth_mobility,
     hjorth_complexity, rms, pow_freq_bands (5 bande EEG standard).
     Output: (n_epochs, n_labels × n_uni_per_label).

  2. **FC-flatten**: upper triangle di ogni matrice FC per ogni banda.
     Output: (n_fc_features,) broadcast su tutti gli epoch.
     n_fc_features = n_bands × n_labels × (n_labels - 1) / 2.

Output finale X: (n_epochs, n_uni_features + n_fc_features).

Ordine colonne: [uni_features... | fc_features...]

Normalizzazione: NESSUNA normalizzazione qui. Le feature raw vengono passate
direttamente al classificatore. La normalizzazione (StandardScaler) è applicata
SOLO dentro la sklearn Pipeline in ml_training/ml_dispatcher.py, fittata
esclusivamente sul train fold per prevenire data leakage.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from mne_features import univariate as mf_uni

# ── Funzioni univariate di default ───────────────────────────────────────────

_DEFAULT_FREQ_BANDS = np.array([1.0, 4.0, 8.0, 13.0, 30.0, 45.0])

_DEFAULT_UNI_FUNCS: list[tuple[str, Callable]] = [
    ("variance", mf_uni.compute_variance),
    ("mean", mf_uni.compute_mean),
    ("std", mf_uni.compute_std),
    ("kurtosis", mf_uni.compute_kurtosis),
    ("hjorth_mobility", mf_uni.compute_hjorth_mobility),
    ("hjorth_complexity", mf_uni.compute_hjorth_complexity),
    ("rms", mf_uni.compute_rms),
]


def _apply_uni_epoch(
    data: np.ndarray,  # (n_labels, n_times)
    sfreq: float,
    funcs: list[tuple[str, Callable]],
) -> tuple[np.ndarray, list[str]]:
    """Applica funzioni univariate su una singola epoch.

    Parameters
    ----------
    data:
        Array (n_labels, n_times).
    sfreq:
        Sampling frequency.
    funcs:
        Lista (nome, callable) univariate mne-features.

    Returns
    -------
    feat_vec : np.ndarray, shape (n_features,)
    feat_names : list[str]
    """
    n_labels = data.shape[0]
    parts: list[np.ndarray] = []
    names: list[str] = []

    for fname, fn in funcs:
        try:
            vec = fn(data)  # (n_labels,)
        except TypeError:
            vec = fn(sfreq, data)  # funzioni che richiedono sfreq come primo arg
        vec = np.asarray(vec).ravel()

        if vec.shape[0] == n_labels:
            parts.append(vec)
            names.extend(f"{fname}_lbl{i:03d}" for i in range(n_labels))
        else:
            # pow_freq_bands: (n_labels * n_bands,) — già flat
            parts.append(vec)
            n_per = len(vec) // n_labels
            names.extend(
                f"{fname}_band{b}_lbl{i:03d}" for b in range(n_per) for i in range(n_labels)
            )

    return np.concatenate(parts), names


def extract_univariate(
    label_tc: np.ndarray,
    sfreq: float,
    *,
    extra_funcs: list[tuple[str, Callable]] | None = None,
    include_pow_freq_bands: bool = True,
    freq_bands: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Estrae feature univariate da label time courses.

    Parameters
    ----------
    label_tc:
        Array shape (n_epochs, n_labels, n_times).
    sfreq:
        Sampling frequency in Hz.
    extra_funcs:
        Lista aggiuntiva di (nome, callable) da mne-features.
    include_pow_freq_bands:
        Se True, aggiunge `compute_pow_freq_bands` con ``freq_bands``.
    freq_bands:
        Bordi bande per pow_freq_bands. Default: 1-4-8-13-30-45 Hz.

    Returns
    -------
    X_uni : np.ndarray, shape (n_epochs, n_uni_features)
    feature_names : list[str]
    """
    if label_tc.ndim != 3:
        raise ValueError(
            f"label_tc deve essere (n_epochs, n_labels, n_times), got {label_tc.shape}"
        )

    funcs = list(_DEFAULT_UNI_FUNCS)
    if include_pow_freq_bands:
        fb = freq_bands if freq_bands is not None else _DEFAULT_FREQ_BANDS

        def _pow(data: np.ndarray) -> np.ndarray:
            return mf_uni.compute_pow_freq_bands(sfreq, data, freq_bands=fb)

        funcs.append(("pow_freq_bands", _pow))
    if extra_funcs:
        funcs.extend(extra_funcs)

    rows: list[np.ndarray] = []
    names: list[str] = []
    for i, epoch_data in enumerate(label_tc):  # (n_labels, n_times)
        vec, epoch_names = _apply_uni_epoch(epoch_data, sfreq, funcs)
        rows.append(vec)
        if i == 0:
            names = epoch_names

    return np.vstack(rows), names


def flatten_fc(
    fc: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Appiattisce le matrici FC (upper triangle) in un vettore 1D.

    Parameters
    ----------
    fc:
        Dict {band_name: matrix_NxN} — output di compute_fc.

    Returns
    -------
    fc_vec : np.ndarray, shape (n_fc_features,)
    feature_names : list[str]
        Formato: "<band>_fc_<i>_<j>".
    """
    parts: list[np.ndarray] = []
    names: list[str] = []

    for band, mat in fc.items():
        n = mat.shape[0]
        idx_i, idx_j = np.triu_indices(n, k=1)
        parts.append(mat[idx_i, idx_j])
        names.extend(f"{band}_fc_{i:03d}_{j:03d}" for i, j in zip(idx_i, idx_j))

    return np.concatenate(parts), names


def build_X(
    label_tc: np.ndarray,
    sfreq: float,
    fc: dict[str, np.ndarray] | None = None,
    *,
    include_univariate: bool = True,
    include_fc: bool = True,
    extra_uni_funcs: list[tuple[str, Callable]] | None = None,
    include_pow_freq_bands: bool = True,
    freq_bands: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Assembla la matrice feature X per ML.

    Parameters
    ----------
    label_tc:
        Shape (n_epochs, n_labels, n_times).
    sfreq:
        Sampling frequency in Hz.
    fc:
        Dict {band: matrix_NxN}. Obbligatorio se include_fc=True.
    include_univariate:
        Se True, include le feature univariate per-epoch.
    include_fc:
        Se True, include le feature FC flatten (broadcast su tutti epoch).
    extra_uni_funcs:
        Funzioni univariate aggiuntive.
    include_pow_freq_bands:
        Aggiunge pow_freq_bands alle feature univariate.
    freq_bands:
        Override bande per pow_freq_bands.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_features)
    feature_names : list[str]

    Notes
    -----
    No normalization applied — output X is raw feature matrix.
    StandardScaler is applied inside the sklearn Pipeline in ml_dispatcher.py
    (fit-on-train-only to prevent leakage).
    """
    if not include_univariate and not include_fc:
        raise ValueError("Almeno una tra include_univariate e include_fc deve essere True.")
    if include_fc and fc is None:
        raise ValueError("fc obbligatorio quando include_fc=True.")

    n_epochs = label_tc.shape[0]
    blocks: list[np.ndarray] = []
    names: list[str] = []

    if include_univariate:
        X_uni, uni_names = extract_univariate(
            label_tc,
            sfreq,
            extra_funcs=extra_uni_funcs,
            include_pow_freq_bands=include_pow_freq_bands,
            freq_bands=freq_bands,
        )
        blocks.append(X_uni)
        names.extend(uni_names)

    if include_fc and fc is not None:
        fc_vec, fc_names = flatten_fc(fc)
        # broadcast: ogni epoch ha le stesse feature FC (FC calcolata su tutto il set)
        blocks.append(np.tile(fc_vec, (n_epochs, 1)))
        names.extend(fc_names)

    return np.hstack(blocks), names
