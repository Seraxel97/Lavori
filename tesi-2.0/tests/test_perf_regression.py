"""S-59 + S-PERF-INVPARC: Performance regression test — traccia tempi step nel tempo.

Usa dati sintetici per essere CI-friendly e rapido (< 30s).
Confronta i tempi misurati contro le soglie in reports/PERF_BASELINE.json.

Step testati:
  - functional_connectivity (FC wPLI, 1 banda)
  - feature_extraction      (build_X, univariate off)
  - ml_single_algo_2fold    (LogReg, 2-fold)
  - e2e_synthetic_total     (tutti e tre in sequenza)
  - inverse_parcellation    (batch apply_inverse_epochs + extract_tc_batch — reali sub-05, skip se mancanti)
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

_BASELINE = Path("reports/PERF_BASELINE.json")
_RNG = np.random.default_rng(42)

_N_EPOCHS = 20
_N_LABELS = 12
_N_TIMES = 500
_SFREQ = 250.0


@pytest.fixture(scope="module")
def baselines() -> dict:
    assert _BASELINE.exists(), f"{_BASELINE} non trovato (S-48 non completato?)"
    return json.loads(_BASELINE.read_text())["thresholds_s"]


@pytest.fixture(scope="module")
def synthetic_label_tc() -> np.ndarray:
    return _RNG.standard_normal((_N_EPOCHS, _N_LABELS, _N_TIMES))


@pytest.fixture(scope="module")
def synthetic_y() -> np.ndarray:
    return np.array([0, 1] * (_N_EPOCHS // 2))


def test_fc_speed(synthetic_label_tc, baselines):
    """FC wPLI su banda alpha: deve completare entro threshold."""
    from connectivity.fc_dispatcher import compute_fc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        fc = compute_fc(synthetic_label_tc, _SFREQ, "wpli", bands={"alpha": (8.0, 13.0)})
        elapsed = time.perf_counter() - t0

    assert "alpha" in fc, "FC: banda alpha mancante"
    assert fc["alpha"].shape == (_N_LABELS, _N_LABELS), "FC: shape errata"
    threshold = baselines["functional_connectivity"]
    assert elapsed <= threshold, (
        f"FC troppo lenta: {elapsed:.2f}s > soglia {threshold}s"
    )


def test_feature_extraction_speed(synthetic_label_tc, baselines):
    """build_X (FC-only): deve completare entro threshold."""
    from connectivity.fc_dispatcher import compute_fc
    from features.dispatcher import build_X

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = compute_fc(synthetic_label_tc, _SFREQ, "wpli", bands={"alpha": (8.0, 13.0)})

    t0 = time.perf_counter()
    X, feat_names = build_X(synthetic_label_tc, _SFREQ, fc, include_univariate=False)
    elapsed = time.perf_counter() - t0

    assert X.shape[0] == _N_EPOCHS, "build_X: n_samples errato"
    assert len(feat_names) > 0, "build_X: nessuna feature"
    threshold = baselines["feature_extraction"]
    assert elapsed <= threshold, (
        f"Feature extraction troppo lenta: {elapsed:.2f}s > soglia {threshold}s"
    )


def test_ml_single_algo_speed(synthetic_label_tc, synthetic_y, baselines):
    """LogReg 2-fold: deve completare entro threshold."""
    from connectivity.fc_dispatcher import compute_fc
    from features.dispatcher import build_X
    from ml_training.ml_dispatcher import run_all_algorithms

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = compute_fc(synthetic_label_tc, _SFREQ, "wpli", bands={"alpha": (8.0, 13.0)})
    X, _ = build_X(synthetic_label_tc, _SFREQ, fc, include_univariate=False)

    t0 = time.perf_counter()
    results = run_all_algorithms(X, synthetic_y, groups=None, n_splits=2, algorithms=["logreg"])
    elapsed = time.perf_counter() - t0

    assert "logreg" in results, "ML: logreg mancante dai risultati"
    assert 0.0 <= results["logreg"].ba_mean <= 1.0, "ML: ba_mean fuori range"
    threshold = baselines["ml_single_algo_2fold"]
    assert elapsed <= threshold, (
        f"ML (logreg 2-fold) troppo lento: {elapsed:.2f}s > soglia {threshold}s"
    )


def test_e2e_synthetic_total(synthetic_label_tc, synthetic_y, baselines):
    """E2E sintetico completo (FC + features + ML): sotto soglia totale."""
    from connectivity.fc_dispatcher import compute_fc
    from features.dispatcher import build_X
    from ml_training.ml_dispatcher import run_all_algorithms

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = compute_fc(synthetic_label_tc, _SFREQ, "wpli", bands={"alpha": (8.0, 13.0)})
    X, _ = build_X(synthetic_label_tc, _SFREQ, fc, include_univariate=False)
    results = run_all_algorithms(X, synthetic_y, groups=None, n_splits=2, algorithms=["logreg"])
    elapsed = time.perf_counter() - t0

    assert len(results) >= 1
    threshold = baselines["e2e_synthetic_total"]
    assert elapsed <= threshold, (
        f"E2E sintetico totale troppo lento: {elapsed:.2f}s > soglia {threshold}s"
    )


def test_inverse_parcellation_batch_speed(inv_path, epochs_path, baselines):
    """Batch apply_inverse_epochs + extract_tc_batch su dati reali sub-05 (skip se assenti).

    Soglia da PERF_BASELINE.json['thresholds_s']['inverse_parcellation'].
    Baseline pre-refactor: 5.490s (loop per-epoch). Target post-refactor: 1.533s.
    CI threshold: 2.5s (su 10 epoch, ~metà del benchmark standard di 20).
    """
    import warnings

    import mne
    import mne.minimum_norm

    from parcellation.extract_label_tc import extract_tc_batch

    inv_op = mne.minimum_norm.read_inverse_operator(str(inv_path), verbose=False)
    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
    sfreq_orig = epochs.info["sfreq"]
    epochs = epochs.decimate(int(sfreq_orig / 500.0))  # match benchmark sfreq=500 Hz
    epochs = epochs[:10]
    src = inv_op["src"]

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stcs = list(
            mne.minimum_norm.apply_inverse_epochs(
                epochs, inv_op, lambda2=1.0 / 9.0, method="dSPM",
                pick_ori=None, return_generator=False, verbose=False,
            )
        )
    label_tc, names = extract_tc_batch(stcs, "aparc", src)
    elapsed = time.perf_counter() - t0

    assert label_tc.shape == (len(epochs), 68, len(epochs.times)), (
        f"shape attesa ({len(epochs)}, 68, {len(epochs.times)}), trovata {label_tc.shape}"
    )
    assert len(names) == 68
    threshold = baselines["inverse_parcellation"]
    assert elapsed <= threshold, (
        f"Batch inverse+parc troppo lento: {elapsed:.2f}s > soglia {threshold}s"
    )
