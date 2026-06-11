"""
Smoke test E2E minimale — pipeline sintetica, niente dataset reale.

Copre i 7 step con dati numpy sintetici:
  - STEP 2 proxy  : mne.SourceEstimate con dati random (verifica API)
  - STEP 3 proxy  : label_tc array sintetico (bypassa fsaverage non disponibile in CI)
  - STEP 4        : fc_dispatcher.compute_fc su label_tc sintetico
  - STEP 5        : features.dispatcher.build_X su fc sintetico
  - STEP 6        : ml_training.ml_dispatcher.run_cv su X sintetico
  - import smoke  : tutti i moduli pubblici importano senza errori

Vincoli: NO download dataset, NO fsaverage, tempo < 60s.
"""

import time

import mne
import numpy as np
import pytest

RNG = np.random.default_rng(42)
N_EPOCHS = 20
N_LABELS = 12
N_TIMES = 500
SFREQ = 250.0


# ── Fixtures sintetiche ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_label_tc() -> np.ndarray:
    """label_tc shape (n_epochs, n_labels, n_times) — sintetico."""
    return RNG.standard_normal((N_EPOCHS, N_LABELS, N_TIMES))


@pytest.fixture(scope="module")
def synthetic_y() -> np.ndarray:
    return np.array([0, 1] * (N_EPOCHS // 2))


@pytest.fixture(scope="module")
def synthetic_fc(synthetic_label_tc):
    from connectivity.fc_dispatcher import compute_fc
    return compute_fc(synthetic_label_tc, SFREQ, "wpli", bands={"alpha": (8.0, 13.0)})


@pytest.fixture(scope="module")
def synthetic_X(synthetic_label_tc, synthetic_fc):
    from features.dispatcher import build_X
    X, _ = build_X(synthetic_label_tc, SFREQ, synthetic_fc, include_univariate=False)
    return X


# ── Import smoke ─────────────────────────────────────────────────────────────

def test_imports_all_modules():
    import connectivity.fc_dispatcher  # noqa: F401
    import features.dispatcher  # noqa: F401
    import ml_training.ml_dispatcher  # noqa: F401
    import parcellation.extract_label_tc  # noqa: F401
    import source_reconstruction.finalize_inverse  # noqa: F401
    from common.dispatcher_base import validate_dispatch_key  # noqa: F401
    from common.paths import BIDS_ROOT, DERIV, SUBJECTS_DIR  # noqa: F401


# ── STEP 2 proxy: fake SourceEstimate ────────────────────────────────────────

def test_fake_source_estimate_creation():
    n_src = N_LABELS * 2
    vertices = [np.arange(N_LABELS), np.arange(N_LABELS)]
    data = RNG.standard_normal((n_src, N_TIMES))
    stc = mne.SourceEstimate(
        data, vertices=vertices, tmin=0.0, tstep=1.0 / SFREQ, subject="fsaverage"
    )
    assert stc.data.shape == (n_src, N_TIMES)
    assert stc.tstep == pytest.approx(1.0 / SFREQ)


# ── STEP 3 proxy: extract_label_tc API ───────────────────────────────────────

def test_extract_label_tc_module_api():
    from parcellation.extract_label_tc import (
        _EXCLUDE_KEYWORDS,
        ATLAS_TO_PARC,
    )
    assert "aparc" in ATLAS_TO_PARC
    assert "schaefer200" in ATLAS_TO_PARC
    assert len(_EXCLUDE_KEYWORDS) >= 2


def test_extract_tc_from_files_validates_mutual_exclusion():
    from parcellation.extract_label_tc import extract_tc_from_files
    with pytest.raises(ValueError, match="mutuamente esclusivi"):
        extract_tc_from_files("fake.stc", "aparc", fwd_path="a.fif", src_path="b.fif")
    with pytest.raises(ValueError, match="Fornire esattamente"):
        extract_tc_from_files("fake.stc", "aparc")


# ── STEP 4: fc_dispatcher ────────────────────────────────────────────────────

def test_fc_dispatcher_shapes(synthetic_label_tc):
    from connectivity.fc_dispatcher import compute_fc
    fc = compute_fc(synthetic_label_tc, SFREQ, "wpli", bands={"alpha": (8.0, 13.0)})
    assert "alpha" in fc
    mat = fc["alpha"]
    assert mat.shape == (N_LABELS, N_LABELS)
    assert np.allclose(mat, mat.T), "FC matrix must be symmetric"


def test_fc_dispatcher_multiple_bands(synthetic_label_tc):
    from connectivity.fc_dispatcher import DEFAULT_BANDS, compute_fc
    fc = compute_fc(synthetic_label_tc, SFREQ, "plv", bands=DEFAULT_BANDS)
    assert set(fc.keys()) == set(DEFAULT_BANDS.keys())
    for band, mat in fc.items():
        assert mat.shape == (N_LABELS, N_LABELS), f"band {band}: wrong shape"


def test_fc_dispatcher_rejects_invalid_metric(synthetic_label_tc):
    from connectivity.fc_dispatcher import compute_fc
    with pytest.raises(ValueError, match="Unknown metric"):
        compute_fc(synthetic_label_tc, SFREQ, "bad")  # type: ignore[arg-type]


# ── STEP 5: features dispatcher ──────────────────────────────────────────────

def test_build_X_fc_only(synthetic_label_tc, synthetic_fc):
    from features.dispatcher import build_X
    X, names = build_X(synthetic_label_tc, SFREQ, synthetic_fc, include_univariate=False)
    n_edges = N_LABELS * (N_LABELS - 1) // 2
    assert X.shape == (N_EPOCHS, n_edges), f"expected ({N_EPOCHS}, {n_edges}), got {X.shape}"
    assert len(names) == n_edges
    assert not np.isnan(X).any()


def test_build_X_univariate_only(synthetic_label_tc):
    from features.dispatcher import build_X
    X, names = build_X(synthetic_label_tc, SFREQ, fc=None, include_univariate=True, include_fc=False)
    assert X.shape[0] == N_EPOCHS
    assert X.shape[1] > 0
    assert not np.isnan(X).any()


# ── STEP 6: ml dispatcher ────────────────────────────────────────────────────

def test_run_cv_logreg(synthetic_X, synthetic_y):
    from ml_training.ml_dispatcher import run_cv
    res = run_cv(synthetic_X, synthetic_y, algorithm="logreg", n_splits=5)
    assert 0.0 <= res.ba_mean <= 1.0
    assert res.n_features == synthetic_X.shape[1]
    assert res.n_samples == synthetic_X.shape[0]
    assert len(res.ba_per_fold) == 5


def test_run_cv_rejects_invalid_algorithm(synthetic_X, synthetic_y):
    from ml_training.ml_dispatcher import run_cv
    with pytest.raises(ValueError, match="Unknown algorithm"):
        run_cv(synthetic_X, synthetic_y, algorithm="xgboost")  # type: ignore[arg-type]


# ── Full chain timing guard ───────────────────────────────────────────────────

def test_full_chain_under_60s(synthetic_label_tc, synthetic_y):
    """Catena completa STEP 4→5→6 su dati sintetici deve finire in <60s."""
    from connectivity.fc_dispatcher import compute_fc
    from features.dispatcher import build_X
    from ml_training.ml_dispatcher import run_cv

    t0 = time.monotonic()

    fc = compute_fc(synthetic_label_tc, SFREQ, "wpli", bands={"alpha": (8.0, 13.0)})
    X, _ = build_X(synthetic_label_tc, SFREQ, fc, include_univariate=False)
    res = run_cv(X, synthetic_y, algorithm="logreg", n_splits=5)

    elapsed = time.monotonic() - t0
    assert elapsed < 60.0, f"pipeline took {elapsed:.1f}s, exceeds 60s CI limit"
    assert 0.0 <= res.ba_mean <= 1.0
