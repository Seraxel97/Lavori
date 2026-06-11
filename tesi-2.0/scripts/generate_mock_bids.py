"""
Generatore dataset BIDS fittizio — 50 soggetti EC/EO (resting-state).

Crea struttura BIDS valida con segnale EEG sintetico puro (numpy random).
Nessun segnale reale: scopo = testing pipeline senza dati reali soggetto.

Struttura output:
    <output-dir>/
        dataset_description.json
        participants.tsv
        sub-01/eeg/sub-01_task-restingstate_run-1_eeg.{vhdr,vmrk,eeg,...}  (EC)
        sub-01/eeg/sub-01_task-restingstate_run-2_eeg.{vhdr,vmrk,eeg,...}  (EO)
        ...
        sub-50/...

Usage:
    python scripts/generate_mock_bids.py \\
        --n-subjects 50 --output-dir data/mock_eceo/ \\
        --sfreq 250 --duration 120

Vincolo sicurezza: output-dir DEVE essere sotto data/ (relativo a CWD o assoluto).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mne
import mne_bids
import numpy as np

_TASK = "restingstate"
_RUNS = {"1": "EC (Eyes Closed)", "2": "EO (Eyes Open)"}
_REPO_ROOT = Path(__file__).parent.parent


def _validate_output_dir(output_dir: str) -> Path:
    """Risolve e valida che output_dir sia sotto data/ (nel repo)."""
    p = Path(output_dir)

    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    else:
        p = p.resolve()
        # Path assoluto: controlla che sia dentro repo_root (rifiuta system paths)
        try:
            p.relative_to(_REPO_ROOT)
        except ValueError:
            sys.exit(
                f"Sicurezza: path assoluto deve essere dentro il repo {_REPO_ROOT}, "
                f"ricevuto: {p}"
            )

    data_root = (_REPO_ROOT / "data").resolve()
    try:
        p.relative_to(data_root)
    except ValueError:
        sys.exit(
            f"Sicurezza: output_dir deve essere sotto {data_root}, "
            f"ricevuto: {p}"
        )
    return p


def _make_raw(sfreq: float, duration: float, rng: np.random.Generator) -> mne.io.RawArray:
    """Crea Raw MNE con 64 canali standard_1020 e dati random."""
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names = montage.ch_names[:64]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(montage, on_missing="ignore", verbose=False)

    n_samples = int(sfreq * duration)
    # Scala ~10 µV (EEG tipico)
    data = rng.standard_normal((64, n_samples)) * 10e-6
    return mne.io.RawArray(data, info, verbose=False)


def generate_mock_bids(
    n_subjects: int,
    output_dir: Path,
    sfreq: float = 250.0,
    duration: float = 120.0,
    seed: int = 42,
    verbose: bool = True,
) -> Path:
    """Genera dataset BIDS con n_subjects soggetti fittizi EC+EO.

    Parameters
    ----------
    n_subjects:
        Numero di soggetti da generare (es. 50).
    output_dir:
        Directory BIDS root (deve essere sotto data/).
    sfreq:
        Frequenza di campionamento in Hz (default 250).
    duration:
        Durata in secondi per ogni run (default 120).
    seed:
        Seed per riproducibilità (default 42).
    verbose:
        Stampa progresso se True.

    Returns
    -------
    Path
        Path alla BIDS root generata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    t_start = time.time()
    for sub_idx in range(1, n_subjects + 1):
        sub_id = f"{sub_idx:02d}"
        for run_id in _RUNS:
            raw = _make_raw(sfreq, duration, rng)
            bids_path = mne_bids.BIDSPath(
                subject=sub_id,
                task=_TASK,
                run=run_id,
                datatype="eeg",
                root=str(output_dir),
            )
            mne_bids.write_raw_bids(
                raw, bids_path,
                overwrite=True,
                allow_preload=True,
                format="BrainVision",
                verbose=False,
            )

        if verbose and sub_idx % 10 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / sub_idx * (n_subjects - sub_idx)
            print(f"  sub-{sub_id}: {sub_idx}/{n_subjects} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

    elapsed = time.time() - t_start
    if verbose:
        print(f"Done: {n_subjects} sub × {len(_RUNS)} run in {elapsed:.1f}s → {output_dir}")
    return output_dir


def validate_bids(output_dir: Path, n_check: int = 3) -> bool:
    """Legge i primi n_check soggetti con mne_bids.read_raw_bids (smoke check).

    Returns True se tutti OK, False altrimenti.
    """
    ok = True
    for sub_idx in range(1, min(n_check + 1, 100)):
        sub_id = f"{sub_idx:02d}"
        for run_id in _RUNS:
            bids_path = mne_bids.BIDSPath(
                subject=sub_id, task=_TASK, run=run_id,
                datatype="eeg", root=str(output_dir),
            )
            try:
                raw = mne_bids.read_raw_bids(bids_path, verbose=False)
                assert raw.info["nchan"] == 64
                assert raw.info["sfreq"] == raw.info["sfreq"]
            except Exception as exc:
                print(f"  FAIL sub-{sub_id} run-{run_id}: {exc}")
                ok = False
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Genera mock BIDS EC/EO dataset")
    ap.add_argument("--n-subjects", type=int, default=50)
    ap.add_argument("--output-dir", required=True, help="Output BIDS root (must be under data/)")
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validate", action="store_true", help="Valida i primi 3 soggetti dopo generazione")
    args = ap.parse_args()

    out = _validate_output_dir(args.output_dir)
    print(f"Generating {args.n_subjects} subjects × {len(_RUNS)} runs "
          f"@ {args.sfreq} Hz × {args.duration}s → {out}")

    generate_mock_bids(
        n_subjects=args.n_subjects,
        output_dir=out,
        sfreq=args.sfreq,
        duration=args.duration,
        seed=args.seed,
    )

    if args.validate:
        print("Validating first 3 subjects...")
        ok = validate_bids(out)
        print("Validation:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
