"""Tests per S-FC-EXTEND: PLV, ImCoh, multi-banda."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from connectivity.fc_dispatcher import compute_fc


def _mock_label_tc(n_epochs: int = 10, n_labels: int = 10, n_times: int = 100,
                   sfreq: float = 250.0) -> np.ndarray:
    """Genera label_tc random fisiologico (n_epochs, n_labels, n_times)."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, n_times / sfreq, n_times)
    tc = np.zeros((n_epochs, n_labels, n_times))
    for ep in range(n_epochs):
        for lbl in range(n_labels):
            freq = rng.uniform(8.0, 13.0)  # alpha range
            phase = rng.uniform(0, 2 * np.pi)
            tc[ep, lbl] = np.sin(2 * np.pi * freq * t + phase) + rng.normal(0, 0.1, n_times)
    return tc


SFREQ = 250.0
ALL_METRICS = ["wpli", "coh", "plv", "imcoh"]
ALL_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def test_plv_produces_valid_matrix():
    """PLV su mock data produce matrice (n_labels, n_labels) con valori [0, 1]."""
    tc = _mock_label_tc()
    fc = compute_fc(tc, SFREQ, "plv", bands={"alpha": (8.0, 13.0)})
    mat = fc["alpha"]
    assert mat.shape == (10, 10), f"Expected (10,10), got {mat.shape}"
    valid = mat[~np.isnan(mat)]
    assert len(valid) > 0, "Tutti NaN"
    assert float(np.nanmin(mat)) >= 0.0, "PLV < 0"
    assert float(np.nanmax(mat)) <= 1.0 + 1e-6, f"PLV > 1: {np.nanmax(mat)}"


def test_imcoh_produces_valid_matrix():
    """ImCoh produce matrice con valori in [-1, 1]."""
    tc = _mock_label_tc()
    fc = compute_fc(tc, SFREQ, "imcoh", bands={"alpha": (8.0, 13.0)})
    mat = fc["alpha"]
    assert mat.shape == (10, 10)
    assert float(np.nanmin(mat)) >= -1.0 - 1e-6, f"ImCoh < -1: {np.nanmin(mat)}"
    assert float(np.nanmax(mat)) <= 1.0 + 1e-6, f"ImCoh > 1: {np.nanmax(mat)}"


def test_multi_band_returns_all_bands():
    """compute_fc con 4 bande restituisce dict con 4 chiavi."""
    tc = _mock_label_tc()
    fc = compute_fc(tc, SFREQ, "wpli", bands=ALL_BANDS)
    assert set(fc.keys()) == set(ALL_BANDS.keys()), f"Bande mancanti: {set(ALL_BANDS)-set(fc)}"
    for band_name, mat in fc.items():
        assert mat.shape == (10, 10), f"{band_name}: shape {mat.shape} ≠ (10,10)"


def test_all_four_metrics_smoke():
    """Tutte e 4 le metriche girano senza crash su alpha."""
    tc = _mock_label_tc()
    for metric in ALL_METRICS:
        fc = compute_fc(tc, SFREQ, metric, bands={"alpha": (8.0, 13.0)})
        assert "alpha" in fc, f"{metric}: manca chiave 'alpha'"
        mat = fc["alpha"]
        assert mat.shape == (10, 10), f"{metric}: shape {mat.shape}"


def test_multi_metric_multi_band_combinations():
    """4 metric × 4 band = 16 combo producono shape corretta."""
    tc = _mock_label_tc(n_epochs=5, n_labels=8, n_times=80)
    for metric in ALL_METRICS:
        fc = compute_fc(tc, SFREQ, metric, bands=ALL_BANDS)
        assert len(fc) == 4, f"{metric}: {len(fc)} bande invece di 4"
        for band, mat in fc.items():
            assert mat.shape == (8, 8), f"{metric}/{band}: shape {mat.shape}"


def test_idempotency_flag_in_compute_script(tmp_path):
    """npz_out con band nel nome consente idempotenza per file esistenti."""
    from scripts.compute_connectivity_per_epoch import npz_out
    p1 = npz_out("sub-007", "aparc", "EO", "wpli", "alpha")
    p2 = npz_out("sub-007", "aparc", "EO", "wpli", "beta")
    p3 = npz_out("sub-007", "aparc", "EO", "coh", "alpha")
    # Nomi distinti per (metric, band) diversi
    assert p1 != p2, "alpha e beta devono avere path distinti"
    assert p1 != p3, "wpli e coh devono avere path distinti"
    assert "band-alpha" in str(p1)
    assert "band-beta" in str(p2)
    assert "metric-coh" in str(p3)
