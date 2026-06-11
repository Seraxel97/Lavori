"""Test step 1.3: aperiodic_fooof features."""

from __future__ import annotations

import numpy as np
import pytest

from features.aperiodic_fooof import (
    FEATURE_NAMES,
    compute_aperiodic_batch,
    compute_aperiodic_features,
)

RNG = np.random.default_rng(42)
FREQS = np.linspace(1, 45, 300)


def _make_psd(exp: float = 1.5, offset: float = 2.0, alpha_peak: bool = True) -> np.ndarray:
    """PSD sintetico: aperiodico + opzionale picco alpha."""
    psd = 10 ** (offset - exp * np.log10(FREQS))
    if alpha_peak:
        psd += np.exp(-0.5 * ((FREQS - 10.5) / 1.5) ** 2) * 0.3
    psd += RNG.normal(0, psd * 0.02)
    return np.abs(psd)


@pytest.fixture(scope="module")
def feat_single():
    return compute_aperiodic_features(FREQS, _make_psd())


def test_feature_names(feat_single):
    assert set(feat_single.keys()) == set(FEATURE_NAMES)


def test_no_nan_band_powers(feat_single):
    for key in ("delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"):
        assert not np.isnan(feat_single[key]), f"{key} is NaN"


def test_spectral_entropy_positive(feat_single):
    assert feat_single["spectral_entropy"] > 0.0


def test_band_powers_positive(feat_single):
    for key in ("delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power"):
        assert feat_single[key] > 0.0, f"{key} not positive"


def test_fit_success_rate():
    """Fit success rate deve essere >= 90% su 20 PSD sintetici."""
    psds = [_make_psd(exp=RNG.uniform(0.5, 2.5)) for _ in range(20)]
    X, names = compute_aperiodic_batch(FREQS, np.array(psds))
    assert X.shape == (20, len(FEATURE_NAMES))
    exp_col = names.index("aperiodic_exp")
    success = np.sum(~np.isnan(X[:, exp_col]))
    assert success / 20 >= 0.90, f"Fit success rate {success}/20 < 90%"


def test_batch_shape():
    psds = np.array([_make_psd() for _ in range(5)])
    X, names = compute_aperiodic_batch(FREQS, psds)
    assert X.shape == (5, 8)
    assert names == FEATURE_NAMES


def test_aperiodic_exp_direction():
    """Exp più alto → PSD decade più velocemente."""
    psd_steep = _make_psd(exp=2.5, alpha_peak=False)
    psd_flat = _make_psd(exp=0.5, alpha_peak=False)
    feat_steep = compute_aperiodic_features(FREQS, psd_steep)
    feat_flat = compute_aperiodic_features(FREQS, psd_flat)
    if not np.isnan(feat_steep["aperiodic_exp"]) and not np.isnan(feat_flat["aperiodic_exp"]):
        assert feat_steep["aperiodic_exp"] > feat_flat["aperiodic_exp"]
