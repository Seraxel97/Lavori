"""Test STEP 4 FC dispatcher — 7 metriche × 2 bande + edge cases + symmetry."""

from pathlib import Path

import numpy as np
import pytest

from connectivity.fc_dispatcher import DEFAULT_BANDS, compute_fc, save_fc

# ── dati sintetici condivisi ──────────────────────────────────────────────────
_RNG = np.random.default_rng(0)
_N_EP, _N_LAB, _N_T, _SFREQ = 8, 10, 512, 256.0
_LABEL_TC = _RNG.standard_normal((_N_EP, _N_LAB, _N_T))

_ALL_METRICS = ["coh", "imcoh", "plv", "ciplv", "pli", "wpli", "wpli2_debiased"]
_TWO_BANDS = {"alpha": (8.0, 13.0), "beta": (13.0, 30.0)}


# ── helper ────────────────────────────────────────────────────────────────────


def _run(metric: str, bands: dict | None = None) -> dict[str, np.ndarray]:
    return compute_fc(_LABEL_TC, _SFREQ, metric, bands=bands or _TWO_BANDS)  # type: ignore[arg-type]


# ── 7 metriche × 2 bande: shape + symmetry + finite ─────────────────────────


@pytest.mark.parametrize("metric", _ALL_METRICS)
def test_compute_fc_shape(metric: str) -> None:
    """Output ha le bande richieste e matrici quadrate NxN."""
    fc = _run(metric)
    assert set(fc.keys()) == set(_TWO_BANDS.keys())
    for band, mat in fc.items():
        assert mat.shape == (_N_LAB, _N_LAB), (
            f"{metric}/{band}: atteso ({_N_LAB},{_N_LAB}), trovato {mat.shape}"
        )


@pytest.mark.parametrize("metric", _ALL_METRICS)
def test_compute_fc_symmetric(metric: str) -> None:
    """Matrice FC deve essere simmetrica (mat[i,j] == mat[j,i])."""
    fc = _run(metric)
    for band, mat in fc.items():
        assert np.allclose(mat, mat.T, atol=1e-10), (
            f"{metric}/{band}: matrice non simmetrica, max_err={np.max(np.abs(mat-mat.T)):.2e}"
        )


@pytest.mark.parametrize("metric", _ALL_METRICS)
def test_compute_fc_finite(metric: str) -> None:
    """Nessun NaN/Inf nel risultato FC."""
    fc = _run(metric)
    for band, mat in fc.items():
        assert np.all(np.isfinite(mat)), f"{metric}/{band}: NaN o Inf trovati"


@pytest.mark.parametrize("metric", _ALL_METRICS)
def test_compute_fc_all_default_bands(metric: str) -> None:
    """Con bands=None usa DEFAULT_BANDS (5 bande)."""
    fc = compute_fc(_LABEL_TC, _SFREQ, metric, bands=None)  # type: ignore[arg-type]
    assert set(fc.keys()) == set(DEFAULT_BANDS.keys())


# ── edge cases ────────────────────────────────────────────────────────────────


def test_compute_fc_n_epochs_1() -> None:
    """n_epochs=1 non deve sollevare eccezioni."""
    tc = _RNG.standard_normal((1, _N_LAB, _N_T))
    fc = compute_fc(tc, _SFREQ, "plv", bands=_TWO_BANDS)  # type: ignore[arg-type]
    assert "alpha" in fc
    assert fc["alpha"].shape == (_N_LAB, _N_LAB)


def test_compute_fc_n_labels_2() -> None:
    """n_labels=2 (minimum) deve produrre matrice 2×2."""
    tc = _RNG.standard_normal((_N_EP, 2, _N_T))
    fc = compute_fc(tc, _SFREQ, "plv", bands=_TWO_BANDS)  # type: ignore[arg-type]
    assert fc["alpha"].shape == (2, 2)


def test_compute_fc_fmax_above_nyquist() -> None:
    """fmax > nyquist viene gestito silenziosamente (mne-connectivity clippa)."""
    bands_over = {"gamma_high": (80.0, 200.0)}  # nyquist = 128 Hz
    fc = compute_fc(_LABEL_TC, _SFREQ, "plv", bands=bands_over)  # type: ignore[arg-type]
    mat = fc["gamma_high"]
    assert mat.shape == (_N_LAB, _N_LAB)
    assert np.all(np.isfinite(mat))


def test_compute_fc_invalid_ndim() -> None:
    """Input 2D deve sollevare ValueError."""
    bad_tc = _RNG.standard_normal((_N_LAB, _N_T))
    with pytest.raises(ValueError, match="n_epochs"):
        compute_fc(bad_tc, _SFREQ, "plv", bands=_TWO_BANDS)  # type: ignore[arg-type]


# ── save_fc ───────────────────────────────────────────────────────────────────


def test_save_fc_roundtrip(tmp_path: Path) -> None:
    """save_fc + np.load ritorna le stesse matrici e nomi."""
    fc = _run("wpli")
    names = [f"roi_{i:02d}" for i in range(_N_LAB)]
    out = tmp_path / "test_fc.npz"
    save_fc(fc, names, out)

    loaded = np.load(out, allow_pickle=True)
    assert list(loaded["names"]) == names
    for band in _TWO_BANDS:
        assert np.allclose(loaded[f"fc_{band}"], fc[band])


def test_save_fc_creates_parent(tmp_path: Path) -> None:
    """save_fc crea la directory padre se non esiste."""
    fc = _run("coh")
    names = [f"r{i}" for i in range(_N_LAB)]
    out = tmp_path / "subdir" / "nested" / "fc.npz"
    save_fc(fc, names, out)
    assert out.exists()
