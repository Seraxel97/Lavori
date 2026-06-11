"""S-29: test mock BIDS dataset generation + pipeline STEP 1 smoke.

Test 1: genera 5 soggetti mock → struttura BIDS valida (mne_bids.read_raw_bids non raise).
Test 2: lettura mock BIDS → verifica che i dati siano leggibili per pipeline.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")
mne_bids = pytest.importorskip("mne_bids")

from scripts.generate_mock_bids import (  # noqa: E402
    _RUNS,
    _TASK,
    generate_mock_bids,
    validate_bids,
)

_SFREQ = 250.0
_DURATION_SHORT = 5.0   # secondi — veloce per CI
_N_TEST_SUBJECTS = 5


# ── Test 1: generazione + validazione struttura BIDS ─────────────────────────


def test_generate_mock_bids_structure(tmp_path: Path) -> None:
    """Genera 5 soggetti mock e verifica struttura BIDS (read_raw_bids senza raise)."""
    out = tmp_path / "mock_eceo"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generate_mock_bids(
            n_subjects=_N_TEST_SUBJECTS,
            output_dir=out,
            sfreq=_SFREQ,
            duration=_DURATION_SHORT,
            verbose=False,
        )

    # dataset_description.json e participants.tsv devono esistere
    assert (out / "dataset_description.json").exists(), "dataset_description.json mancante"
    assert (out / "participants.tsv").exists(), "participants.tsv mancante"

    # Verifica struttura per ogni soggetto e run
    for sub_idx in range(1, _N_TEST_SUBJECTS + 1):
        sub_id = f"{sub_idx:02d}"
        for run_id in _RUNS:
            sub_dir = out / f"sub-{sub_id}" / "eeg"
            assert sub_dir.exists(), f"Mancante: sub-{sub_id}/eeg/"
            vhdr = sub_dir / f"sub-{sub_id}_task-{_TASK}_run-{run_id}_eeg.vhdr"
            assert vhdr.exists(), f"Mancante: {vhdr.name}"

    # Validazione lettura via mne_bids (smoke check per tutti i soggetti)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok = validate_bids(out, n_check=_N_TEST_SUBJECTS)
    assert ok, "validate_bids: almeno un soggetto non leggibile"


# ── Test 2: pipeline readback — verifica canali, sfreq, shape ────────────────


def test_mock_bids_readback_properties(tmp_path: Path) -> None:
    """Legge i mock soggetti e verifica proprietà EEG (n_channels, sfreq, durata)."""
    out = tmp_path / "mock_eceo_readback"
    n_subs = 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generate_mock_bids(
            n_subjects=n_subs,
            output_dir=out,
            sfreq=_SFREQ,
            duration=_DURATION_SHORT,
            verbose=False,
        )

    for sub_idx in range(1, n_subs + 1):
        sub_id = f"{sub_idx:02d}"
        for run_id in _RUNS:
            bids_path = mne_bids.BIDSPath(
                subject=sub_id, task=_TASK, run=run_id,
                datatype="eeg", root=str(out),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = mne_bids.read_raw_bids(bids_path, verbose=False)

            assert raw.info["nchan"] == 64, (
                f"sub-{sub_id} run-{run_id}: atteso 64 ch, trovato {raw.info['nchan']}"
            )
            assert raw.info["sfreq"] == _SFREQ, (
                f"sub-{sub_id} run-{run_id}: sfreq atteso {_SFREQ}, trovato {raw.info['sfreq']}"
            )
            n_samples = raw.n_times
            expected_min = int(_SFREQ * _DURATION_SHORT * 0.9)
            assert n_samples >= expected_min, (
                f"sub-{sub_id} run-{run_id}: n_times={n_samples} < {expected_min}"
            )
            # Dati devono essere random (std >> 0)
            raw.load_data(verbose=False)
            data = raw.get_data()
            assert np.std(data) > 0, f"sub-{sub_id} run-{run_id}: dati costanti (std=0)"
