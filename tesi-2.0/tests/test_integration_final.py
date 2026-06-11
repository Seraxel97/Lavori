"""S-56: Final integration tests E2E.

test_e2e_matchingpennies_full  — run_e2e su sub-05 reale, verifica ≥1 STC, FC, prediction.
test_mock_dataset_smoke        — pipeline su mock_eceo BIDS (S-29), verifica struct OK.

Skip automatico se dati non disponibili.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pytest

_DERIV = Path("data/derivatives/mne-bids-pipeline")
_SUB05_EEG = _DERIV / "sub-05" / "eeg"
_EPO_FILE = _SUB05_EEG / "sub-05_task-matchingpennies_epo.fif"
_INV_FILE = _SUB05_EEG / "sub-05_inv.fif"

_HAS_REAL_DATA = _EPO_FILE.exists() and _INV_FILE.exists()


# ── Test 1: E2E matchingpennies su dati reali ─────────────────────────────────


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="dati sub-05 non disponibili")
def test_e2e_matchingpennies_full(tmp_path):
    """Pipeline E2E completa su sub-05: ≥1 STC, ≥1 FC band, ≥1 prediction."""
    from pipeline_mne_bids.run_e2e_matchingpennies import run_e2e

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = run_e2e(
            subject="sub-05",
            deriv_root=str(_DERIV),
            atlas="aparc",
            metric="wpli",
            n_epochs_max=8,
            n_cv_splits=2,
            bands={"alpha": (8.0, 13.0)},
            out_report=str(tmp_path / "E2E_MATCHINGPENNIES.md"),
        )
    elapsed = time.time() - t0

    # ≥1 epoch processata (STC prodotta per ogni epoch)
    assert summary["n_epochs"] >= 1, "nessuna epoch processata"

    # ≥1 FC band
    assert len(summary["bands"]) >= 1, "nessuna FC band nel risultato"

    # ≥1 prediction (almeno un algoritmo ML ha girato)
    assert len(summary["results"]) >= 1, "nessun risultato ML"
    for algo, res in summary["results"].items():
        assert "ba_mean" in res, f"{algo}: ba_mean mancante"
        assert 0.0 <= res["ba_mean"] <= 1.0, f"{algo}: ba_mean fuori range [0,1]"

    # n_features > 0
    assert summary["n_features"] > 0, "n_features=0"

    # wall time < 15 min
    assert elapsed < 900, f"E2E troppo lento: {elapsed:.0f}s > 900s"

    # report scritto
    assert (tmp_path / "E2E_MATCHINGPENNIES.md").exists(), "report E2E non scritto"


# ── Test 2: Mock BIDS dataset smoke (S-29) ────────────────────────────────────


def test_mock_dataset_smoke(tmp_path):
    """Genera mock_eceo BIDS, verifica struttura BIDS valida."""
    from scripts.generate_mock_bids import generate_mock_bids, validate_bids

    out = tmp_path / "mock_eceo"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generate_mock_bids(
            n_subjects=2,
            output_dir=out,
            sfreq=250.0,
            duration=3.0,
            verbose=False,
        )

    # dataset_description.json e participants.tsv
    assert (out / "dataset_description.json").exists()
    assert (out / "participants.tsv").exists()

    # almeno 1 soggetto con struttura BIDS corretta
    sub01_eeg = out / "sub-01" / "eeg"
    assert sub01_eeg.exists(), "sub-01/eeg/ mancante"

    # almeno 1 run con file .vhdr
    found_vhdr = list(sub01_eeg.glob("*.vhdr"))
    assert len(found_vhdr) >= 1, "nessun file .vhdr trovato in sub-01/eeg/"

    # validate_bids smoke check
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = validate_bids(out, n_check=2)
    assert ok, "validate_bids: struttura BIDS non leggibile"
