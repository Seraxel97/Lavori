"""Test E2E pipeline matchingpennies — smoke test con dati sintetici."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from connectivity.fc_dispatcher import compute_fc
from features.dispatcher import build_X
from ml_training.ml_dispatcher import run_all_algorithms
from pipeline_mne_bids.run_e2e.exec import _build_y
from pipeline_mne_bids.run_e2e.report import write_report as _write_report

# ── fixtures sintetiche ────────────────────────────────────────────────────────

_RNG = np.random.default_rng(7)
_N_EP, _N_LAB, _N_T, _SFREQ = 20, 10, 200, 250.0
_LABEL_TC = _RNG.standard_normal((_N_EP, _N_LAB, _N_T))
_Y = np.array([0] * 10 + [1] * 10)
_FC = compute_fc(_LABEL_TC, _SFREQ, "wpli", bands={"alpha": (8.0, 13.0)})


# ── integration: full feature → ML pipeline ──────────────────────────────────


def test_e2e_pipeline_feature_to_ml() -> None:
    """feature extraction → ML su dati sintetici produce risultato valido."""
    X, names = build_X(_LABEL_TC, _SFREQ, _FC)
    assert X.shape[0] == _N_EP
    assert X.shape[1] == len(names)
    assert X.shape[1] > 0

    results = run_all_algorithms(X, _Y, groups=None, algorithms=["logreg"], n_splits=3)
    assert "logreg" in results
    assert 0.0 <= results["logreg"].ba_mean <= 1.0


def test_e2e_x_shape_consistent_with_fc() -> None:
    """X ha le colonne FC corrette (upper triangle)."""
    n_fc_per_band = _N_LAB * (_N_LAB - 1) // 2
    n_bands = len(_FC)
    X_fc_only, names_fc = build_X(_LABEL_TC, _SFREQ, _FC, include_univariate=False)
    assert X_fc_only.shape[1] == n_bands * n_fc_per_band


def test_e2e_all_algorithms_run() -> None:
    """Tutti e 6 gli algoritmi (incl. logreg_nested) completano senza errori."""
    X, _ = build_X(_LABEL_TC, _SFREQ, _FC)
    results = run_all_algorithms(X, _Y, groups=None, n_splits=3)
    assert set(results.keys()) == {"logreg", "logreg_nested", "svm", "mlp", "rf", "gb"}
    for algo, r in results.items():
        assert 0.0 <= r.ba_mean <= 1.0, f"{algo}: BA fuori range"


def test_e2e_feature_names_unique() -> None:
    """I nomi feature devono essere unici."""
    _, names = build_X(_LABEL_TC, _SFREQ, _FC)
    assert len(names) == len(set(names)), "Nomi feature duplicati trovati"


# ── _write_report ──────────────────────────────────────────────────────────────


def test_write_report_creates_file(tmp_path: Path) -> None:
    """_write_report crea il file MD."""
    summary = {
        "subject": "sub-05",
        "atlas": "aparc",
        "metric": "wpli",
        "n_epochs": 20,
        "n_labels": 10,
        "n_times": 200,
        "sfreq": 250.0,
        "n_features": 50,
        "elapsed_s": 5.0,
        "bands": ["alpha"],
        "results": {
            "logreg": {"ba_mean": 0.55, "ba_std": 0.05, "ba_per_fold": [0.5, 0.55, 0.6]},
        },
    }
    out = tmp_path / "E2E_TEST.md"
    _write_report(summary, out)
    assert out.exists()
    content = out.read_text()
    assert "sub-05" in content
    assert "logreg" in content
    assert "0.550" in content


# ── _build_y helper ────────────────────────────────────────────────────────────


def test_build_y_label_map() -> None:
    """_build_y mappa correttamente raised-left→0, raised-right→1."""
    import mne

    # Crea epochs sintetici con eventi
    rng = np.random.default_rng(3)
    n_ch, n_t = 5, 100
    sfreq = 250.0
    ch_names = [f"EEG{i:03d}" for i in range(n_ch)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")

    events = np.array(
        [
            [100, 0, 1],
            [200, 0, 2],
            [300, 0, 3],
            [400, 0, 4],
        ]
    )
    event_id = {
        "raised-left/match-false": 1,
        "raised-left/match-true": 2,
        "raised-right/match-false": 3,
        "raised-right/match-true": 4,
    }
    data = rng.standard_normal((4, n_ch, n_t)) * 1e-6

    epochs = mne.EpochsArray(data, info, events=events, event_id=event_id, tmin=0.0)
    y = _build_y(epochs)

    assert list(y) == [0, 0, 1, 1], f"atteso [0,0,1,1], trovato {list(y)}"
