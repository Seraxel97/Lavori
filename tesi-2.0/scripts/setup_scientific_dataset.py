"""
Dataset selection helper — Scientific dataset setup (ds005385 / LEMON).

Azioni disponibili:
  symlink  — crea symlink data/<dataset>/ → ~/Scrivania/Tesi/data/<dataset>/
  verify   — legge 1 soggetto random con read_raw_bids + stampa report
  prep     — genera config/config_step1_<dataset>.py con path override

Usage:
    python scripts/setup_scientific_dataset.py --dataset ds005385 --action symlink
    python scripts/setup_scientific_dataset.py --dataset LEMON --action verify
    python scripts/setup_scientific_dataset.py --dataset ds005385 --action prep
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_DATA_ROOT = _REPO_ROOT / "data"
_TESI_DATA = Path.home() / "Scrivania" / "Tesi" / "data"

_DATASET_META = {
    "ds005385": {
        "desc": "OpenNeuro ds005385 (EEG resting-state EC/EO, 50 sub)",
        "task": "restingstate",
        "ch_types": ["eeg"],
        "conditions": ["eyes_closed", "eyes_open"],
        "disk_gb": 4.5,
    },
    "LEMON": {
        "desc": "MPI Leipzig Mind-Brain-Body (LEMON) EEG resting-state",
        "task": "restingstate",
        "ch_types": ["eeg"],
        "conditions": ["eyes_closed", "eyes_open"],
        "disk_gb": 8.0,
    },
}


def _validate_data_path(p: Path) -> None:
    """Verifica che il path sia sotto data/ (sicurezza)."""
    try:
        p.resolve().relative_to(_DATA_ROOT.resolve())
    except ValueError:
        sys.exit(f"Sicurezza: il path deve essere sotto {_DATA_ROOT}, ricevuto: {p}")


def action_symlink(dataset: str) -> None:
    """Crea symlink data/<dataset>/ → ~/Scrivania/Tesi/data/<dataset>/."""
    target = _TESI_DATA / dataset
    link = _DATA_ROOT / dataset

    _validate_data_path(link)

    if not target.exists():
        sys.exit(
            f"Sorgente non trovata: {target}\n"
            f"Assicurarsi che il dataset sia scaricato in ~/Scrivania/Tesi/data/{dataset}/."
        )

    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == target.resolve():
            print(f"Symlink già esistente e corretto: {link} → {target}")
            return
        sys.exit(
            f"Path già esistente ma non punta a {target}: {link}\n"
            "Rimuovere manualmente se si vuole ri-creare il symlink."
        )

    link.symlink_to(target)
    print(f"Symlink creato: {link} → {target}")


def action_verify(dataset: str) -> None:
    """Legge 1 soggetto random con mne_bids.read_raw_bids e stampa report."""
    import mne_bids

    bids_root = _DATA_ROOT / dataset
    if not bids_root.exists():
        sys.exit(
            f"Dataset non trovato: {bids_root}\n"
            f"Eseguire prima: --action symlink"
        )

    subjects = mne_bids.get_entity_vals(str(bids_root), "subject")
    if not subjects:
        sys.exit(f"Nessun soggetto trovato in {bids_root}")

    sub = random.choice(subjects)
    meta = _DATASET_META.get(dataset, {})
    task = meta.get("task", "restingstate")

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset} ({meta.get('desc', 'N/A')})")
    print(f"BIDS root: {bids_root}")
    print(f"N soggetti: {len(subjects)}")
    print(f"Soggetto campione: sub-{sub}")

    # Prova a trovare i file disponibili per il soggetto
    bids_path = mne_bids.BIDSPath(
        subject=sub, task=task, datatype="eeg", root=str(bids_root)
    )
    matches = bids_path.match()
    if not matches:
        # Prova senza task specifico
        bids_path = mne_bids.BIDSPath(subject=sub, datatype="eeg", root=str(bids_root))
        matches = bids_path.match()

    if not matches:
        print(f"Nessun file EEG trovato per sub-{sub} nel layout BIDS.")
        return

    bp = matches[0]
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne_bids.read_raw_bids(bp, verbose=False)
        print(f"N canali: {raw.info['nchan']}")
        print(f"sfreq: {raw.info['sfreq']} Hz")
        print(f"Durata: {raw.n_times / raw.info['sfreq']:.1f}s")
        print(f"Tipo canali: {set(raw.get_channel_types())}")
        print("Verifica BIDS: PASS")
    except Exception as exc:
        print(f"Errore lettura sub-{sub}: {exc}")
    print("="*50)


def action_prep(dataset: str) -> None:
    """Genera config/config_step1_<dataset>.py con path override."""
    meta = _DATASET_META.get(dataset, {})
    config_path = _REPO_ROOT / "config" / f"config_step1_{dataset.lower()}.py"

    if config_path.exists():
        print(f"Config già esistente: {config_path}")
        resp = input("Sovrascrivere? [y/N] ").strip().lower()
        if resp != "y":
            print("Annullato.")
            return

    bids_root = _DATA_ROOT / dataset
    task = meta.get("task", "restingstate")
    conditions = meta.get("conditions", [])
    ch_types = meta.get("ch_types", ["eeg"])

    lines = [
        f'"""\nSTEP 1 — {dataset} EEG, preprocessing base.\n\n'
        f'Dataset: {meta.get("desc", dataset)}\n'
        f'Generato automaticamente da scripts/setup_scientific_dataset.py.\n"""\n',
        "from config.config_base import *  # noqa: F401, F403\n",
        "",
        "# ── Path override ────────────────────────────────────────────────────",
        f'bids_root = "{bids_root}"',
        "",
        "# ── Dataset-specific params ──────────────────────────────────────────",
        f'task = "{task}"',
        f"ch_types = {ch_types!r}",
        f"conditions = {conditions!r}",
        "",
        "# ── Rimuovi crop_runs se il dataset ha sessioni brevi ────────────────",
        "# crop_runs = None  # decommentare se necessario",
    ]

    config_path.write_text("\n".join(lines) + "\n")
    print(f"Config generato: {config_path}")
    print(f"  task={task}, ch_types={ch_types}, conditions={conditions}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scientific dataset setup helper")
    ap.add_argument(
        "--dataset", required=True, choices=list(_DATASET_META),
        help="Dataset scientifico da configurare."
    )
    ap.add_argument(
        "--action", required=True, choices=["symlink", "verify", "prep"],
        help="Azione: symlink|verify|prep."
    )
    args = ap.parse_args()

    if args.action == "symlink":
        action_symlink(args.dataset)
    elif args.action == "verify":
        action_verify(args.dataset)
    elif args.action == "prep":
        action_prep(args.dataset)
