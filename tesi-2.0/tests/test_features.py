"""Test STEP 5 features dispatcher — univariate + FC-flatten."""

import numpy as np
import pytest

from features.dispatcher import build_X, extract_univariate, flatten_fc

# ── dati sintetici ────────────────────────────────────────────────────────────
_RNG = np.random.default_rng(1)
_N_EP, _N_LAB, _N_T, _SFREQ = 10, 12, 512, 256.0
_LABEL_TC = _RNG.standard_normal((_N_EP, _N_LAB, _N_T))

_BANDS = ["alpha", "beta"]
_FC: dict[str, np.ndarray] = {
    b: (lambda m: m + m.T)(_RNG.uniform(0, 1, (_N_LAB, _N_LAB))) / 2.0
    for b in _BANDS
}
# Assicura simmetria e diagonale zero
for b in _BANDS:
    np.fill_diagonal(_FC[b], 0.0)


# ── extract_univariate ────────────────────────────────────────────────────────


def test_extract_univariate_shape() -> None:
    """Output shape (n_epochs, n_uni_features)."""
    X, names = extract_univariate(_LABEL_TC, _SFREQ)
    assert X.ndim == 2
    assert X.shape[0] == _N_EP
    assert X.shape[1] == len(names)
    assert len(names) > 0


def test_extract_univariate_finite() -> None:
    """Nessun NaN/Inf nelle feature univariate."""
    X, _ = extract_univariate(_LABEL_TC, _SFREQ)
    assert np.all(np.isfinite(X)), "NaN o Inf trovati nelle feature univariate"


def test_extract_univariate_no_pow_freq_bands() -> None:
    """Con include_pow_freq_bands=False, feature count ridotto."""
    X_full, names_full = extract_univariate(_LABEL_TC, _SFREQ, include_pow_freq_bands=True)
    X_slim, names_slim = extract_univariate(_LABEL_TC, _SFREQ, include_pow_freq_bands=False)
    assert X_slim.shape[1] < X_full.shape[1]
    assert not any("pow_freq_bands" in n for n in names_slim)


def test_extract_univariate_invalid_ndim() -> None:
    """Input 2D deve sollevare ValueError."""
    bad = _RNG.standard_normal((_N_LAB, _N_T))
    with pytest.raises(ValueError, match="n_epochs"):
        extract_univariate(bad, _SFREQ)


# ── flatten_fc ────────────────────────────────────────────────────────────────


def test_flatten_fc_shape() -> None:
    """Output shape: n_bands × n_labels×(n_labels-1)/2."""
    fc_vec, names = flatten_fc(_FC)
    n_edges = _N_LAB * (_N_LAB - 1) // 2
    expected = len(_BANDS) * n_edges
    assert len(fc_vec) == expected, f"atteso {expected}, trovato {len(fc_vec)}"
    assert len(names) == len(fc_vec)


def test_flatten_fc_upper_triangle() -> None:
    """Controlla che siano estratti esattamente gli elementi upper-triangle."""
    fc_single = {"alpha": _FC["alpha"].copy()}
    vec, names = flatten_fc(fc_single)
    # Ricostruisci manualmente upper triangle
    idx_i, idx_j = np.triu_indices(_N_LAB, k=1)
    expected_vec = _FC["alpha"][idx_i, idx_j]
    assert np.allclose(vec, expected_vec)


def test_flatten_fc_names_format() -> None:
    """Nomi feature nel formato '<band>_fc_<i>_<j>'."""
    _, names = flatten_fc(_FC)
    for name in names[:5]:
        parts = name.split("_")
        assert len(parts) >= 4, f"formato nome inatteso: {name}"


# ── build_X ───────────────────────────────────────────────────────────────────


def test_build_X_shape_both() -> None:
    """build_X con uni+FC produce (n_epochs, n_uni+n_fc)."""
    X, names = build_X(_LABEL_TC, _SFREQ, _FC)
    assert X.shape[0] == _N_EP
    assert X.shape[1] == len(names)

    X_uni, _ = extract_univariate(_LABEL_TC, _SFREQ)
    X_fc, _ = flatten_fc(_FC)
    assert X.shape[1] == X_uni.shape[1] + len(X_fc)


def test_build_X_fc_only() -> None:
    """build_X con solo FC: tutte le righe identiche (broadcast)."""
    X, names = build_X(_LABEL_TC, _SFREQ, _FC, include_univariate=False)
    assert X.shape == (_N_EP, len(flatten_fc(_FC)[0]))
    # Tutte le epoch hanno gli stessi valori FC
    assert np.allclose(X[0], X[-1])


def test_build_X_uni_only() -> None:
    """build_X con solo univariate: no FC features nei nomi."""
    X, names = build_X(_LABEL_TC, _SFREQ, include_fc=False)
    assert not any("_fc_" in n for n in names)


def test_build_X_finite() -> None:
    """Nessun NaN/Inf in X."""
    X, _ = build_X(_LABEL_TC, _SFREQ, _FC)
    assert np.all(np.isfinite(X)), "NaN o Inf trovati in X"


def test_build_X_raises_no_source() -> None:
    """ValueError se nessuna sorgente feature attivata."""
    with pytest.raises(ValueError, match="Almeno"):
        build_X(_LABEL_TC, _SFREQ, include_univariate=False, include_fc=False)


def test_build_X_raises_fc_none() -> None:
    """ValueError se include_fc=True ma fc=None."""
    with pytest.raises(ValueError, match="fc obbligatorio"):
        build_X(_LABEL_TC, _SFREQ, fc=None, include_univariate=False, include_fc=True)
