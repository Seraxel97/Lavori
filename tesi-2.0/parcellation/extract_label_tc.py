"""
STEP 3 — Estrazione time course per ROI da source estimates (STC).

Wraps `mne.extract_label_time_course` su STC prodotti da STEP 2.
Atlasi supportati (tutti già presenti in fsaverage/label/):
  - "aparc"        : Desikan-Killiany ~68 ROI corticali (aparc.annot)
  - "destrieux"    : Destrieux a2009s ~148 ROI (aparc.a2009s.annot)
  - "schaefer100"  : Schaefer2018 100 parcels x 7 networks
  - "schaefer200"  : Schaefer2018 200 parcels x 7 networks
  - "schaefer400"  : Schaefer2018 400 parcels x 7 networks

Mode di aggregazione: vedi `mne.extract_label_time_course` (default="mean_flip"
per source-level functional connectivity, robusto a sign-flip).

Variabili d'ambiente:
  TESI_SUBJECTS_DIR  : override per il path di fsaverage (default: ~/mne_data/MNE-fsaverage-data)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import mne
import numpy as np

_DEFAULT_SUBJECTS_DIR = Path.home() / "mne_data" / "MNE-fsaverage-data"
SUBJECTS_DIR = Path(os.environ.get("TESI_SUBJECTS_DIR", str(_DEFAULT_SUBJECTS_DIR)))

ATLAS_TO_PARC: dict[str, str] = {
    "aparc": "aparc",
    "destrieux": "aparc.a2009s",
    "schaefer100": "Schaefer2018_100Parcels_7Networks_order",
    "schaefer200": "Schaefer2018_200Parcels_7Networks_order",
    "schaefer400": "Schaefer2018_400Parcels_7Networks_order",
}

AtlasName = Literal["aparc", "destrieux", "schaefer100", "schaefer200", "schaefer400"]

_EXCLUDE_KEYWORDS = ("unknown", "medial_wall", "background+freesurfer")


def get_labels(atlas: AtlasName, *, subject: str = "fsaverage") -> list[mne.Label]:
    """Legge le label di un atlante da fsaverage, escludendo ROI non corticali.

    Parameters
    ----------
    atlas:
        Chiave atlante (una tra: aparc, destrieux, schaefer100, schaefer200, schaefer400).
    subject:
        Subject di riferimento per le annotazioni (default: "fsaverage").

    Returns
    -------
    list[mne.Label]
        Label corticali filtrate (esclusi unknown, medial_wall, background).
    """
    parc = ATLAS_TO_PARC[atlas]
    labels = mne.read_labels_from_annot(
        subject=subject, parc=parc, subjects_dir=str(SUBJECTS_DIR), verbose=False
    )
    return [
        lbl for lbl in labels
        if not any(kw in lbl.name.lower() for kw in _EXCLUDE_KEYWORDS)
    ]


def extract_tc(
    stc: mne.SourceEstimate,
    atlas: AtlasName,
    src: mne.SourceSpaces | str | Path,
    *,
    mode: str = "mean_flip",
    subject: str = "fsaverage",
) -> tuple[np.ndarray, list[str]]:
    """Estrae time course per ROI da uno STC.

    Parameters
    ----------
    stc:
        Source estimate MNE (prodotto da STEP 2 / finalize_inverse).
    atlas:
        Chiave atlante.
    src:
        SourceSpaces o path a file .fif (src o fwd).
    mode:
        Modalità aggregazione MNE (default: "mean_flip").
    subject:
        Subject fsaverage per le label.

    Returns
    -------
    tc : np.ndarray, shape (n_labels, n_times)
    names : list[str]
        Nomi label, ordinati come righe di tc.
    """
    if isinstance(src, (str, Path)):
        src = mne.read_source_spaces(str(src), verbose=False)
    labels = get_labels(atlas, subject=subject)
    tc = mne.extract_label_time_course(
        stc, labels, src, mode=mode, allow_empty=True, verbose=False
    )
    names = [lbl.name for lbl in labels]
    return np.asarray(tc), names


def extract_tc_batch(
    stcs: list[mne.SourceEstimate],
    atlas: AtlasName,
    src: mne.SourceSpaces | str | Path,
    *,
    mode: str = "mean_flip",
    subject: str = "fsaverage",
) -> tuple[np.ndarray, list[str]]:
    """Estrae time course per ROI da lista di STC in una sola call batch.

    Più efficiente di chiamare extract_tc in loop: una sola invocazione a
    mne.extract_label_time_course per tutti gli STC (prepara le label 1×).

    Parameters
    ----------
    stcs:
        Lista di SourceEstimate (uno per epoch).
    atlas:
        Chiave atlante.
    src:
        SourceSpaces o path a file .fif.
    mode:
        Modalità aggregazione MNE (default: "mean_flip").
    subject:
        Subject fsaverage per le label.

    Returns
    -------
    tc : np.ndarray, shape (n_stcs, n_labels, n_times)
    names : list[str]
        Nomi label, ordinati come righe di tc[i].
    """
    if isinstance(src, (str, Path)):
        src = mne.read_source_spaces(str(src), verbose=False)
    labels = get_labels(atlas, subject=subject)
    tc = mne.extract_label_time_course(
        stcs, labels, src, mode=mode, allow_empty=True, verbose=False
    )
    names = [lbl.name for lbl in labels]
    return np.stack(tc, axis=0), names


def extract_tc_from_files(
    stc_path: str | Path,
    atlas: AtlasName,
    *,
    src_path: str | Path | None = None,
    fwd_path: str | Path | None = None,
    mode: str = "mean_flip",
    subject: str = "fsaverage",
) -> tuple[np.ndarray, list[str]]:
    """Estrae time course da percorsi file, con sorgente esplicita fwd o src.

    Esattamente uno tra ``fwd_path`` e ``src_path`` deve essere fornito.

    Parameters
    ----------
    stc_path:
        Path allo STC (stem senza -lh/-rh; MNE legge entrambi gli emisferi).
    atlas:
        Chiave atlante.
    src_path:
        Path a file SourceSpaces (*-src.fif). Esclusivo con fwd_path.
    fwd_path:
        Path a forward solution (*-fwd.fif); il src viene estratto internamente.
        Esclusivo con src_path.
    mode:
        Modalità aggregazione (default: "mean_flip").
    subject:
        Subject fsaverage per le label.

    Returns
    -------
    tc : np.ndarray, shape (n_labels, n_times)
    names : list[str]
    """
    if fwd_path is None and src_path is None:
        raise ValueError("Fornire esattamente uno tra fwd_path e src_path.")
    if fwd_path is not None and src_path is not None:
        raise ValueError("fwd_path e src_path sono mutuamente esclusivi.")

    stc = mne.read_source_estimate(str(stc_path))

    if fwd_path is not None:
        fwd = mne.read_forward_solution(str(fwd_path), verbose=False)
        src = fwd["src"]
    else:
        src = mne.read_source_spaces(str(src_path), verbose=False)

    return extract_tc(stc, atlas, src, mode=mode, subject=subject)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stc", required=True)
    ap.add_argument("--fwd", default=None, help="path forward solution *-fwd.fif")
    ap.add_argument("--src", default=None, help="path source spaces *-src.fif")
    ap.add_argument("--atlas", required=True, choices=list(ATLAS_TO_PARC))
    ap.add_argument("--mode", default="mean_flip")
    args = ap.parse_args()

    tc, names = extract_tc_from_files(
        args.stc,
        args.atlas,
        fwd_path=args.fwd,
        src_path=args.src,
        mode=args.mode,
    )
    print(f"atlas={args.atlas} mode={args.mode}")
    print(f"shape tc=({tc.shape[0]}, {tc.shape[1]})  ({len(names)} labels × n_times)")
    print(f"first 5 labels: {names[:5]}")
    print(f"last  5 labels: {names[-5:]}")
