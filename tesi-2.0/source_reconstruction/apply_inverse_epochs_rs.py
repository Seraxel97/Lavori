"""
STEP 2b — Apply inverse operator su epochs resting-state (sliding 2s).

Funzione principale: `apply_inverse_epochs_rs`
  - Input: Epochs MNE (EC/EO resting-state in finestre 2s) + inverse operator
  - Output: lista di SourceEstimate (uno per epoch)

Propedeutico a S-10 (parcellazione RS) ed EC/EO analysis.

Usage:
    python source_reconstruction/apply_inverse_epochs_rs.py \\
        --subject 05 --task matchingpennies --method dSPM
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Generator
from pathlib import Path

import mne
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

_DEFAULT_DERIV = Path(__file__).parent.parent / "data" / "derivatives" / "mne-bids-pipeline"
DERIV = Path(os.environ.get("TESI_DERIV", str(_DEFAULT_DERIV)))


def apply_inverse_epochs_rs(
    epochs: mne.Epochs,
    inv_op: mne.minimum_norm.InverseOperator,
    *,
    method: str = "dSPM",
    lambda2: float = 1.0 / 9.0,
    pick_ori: str | None = "normal",
    return_generator: bool = False,
) -> list[mne.SourceEstimate] | Generator[mne.SourceEstimate]:
    """Applica inverse operator su epochs resting-state.

    Parameters
    ----------
    epochs:
        Epochs MNE già epochate (es. finestre sliding 2s da RS continuo).
    inv_op:
        Inverse operator pre-calcolato (compatibile con epochs.info).
    method:
        Metodo inverse ('dSPM', 'MNE', 'sLORETA', 'eLORETA').
    lambda2:
        Parametro di regolarizzazione (default 1/9, standard dSPM).
    pick_ori:
        Orientazione da estrarre ('normal' per signed, None per magnitude).
    return_generator:
        Se True, ritorna generator (memory-efficient per dataset grandi).

    Returns
    -------
    list[SourceEstimate] | Generator[SourceEstimate, None, None]
        Lista di STC o generator se return_generator=True.
        Shape data per STC: (n_sources, n_times).
    """
    stcs = apply_inverse_epochs(
        epochs,
        inv_op,
        lambda2=lambda2,
        method=method,
        pick_ori=pick_ori,
        return_generator=return_generator,
        verbose=False,
    )
    if return_generator:
        return stcs
    return list(stcs)


def save_stcs(
    stcs: list[mne.SourceEstimate],
    out_dir: Path,
    prefix: str,
) -> list[Path]:
    """Salva la lista di STC su disco.

    Parameters
    ----------
    stcs:
        Lista di SourceEstimate da salvare.
    out_dir:
        Directory di output (creata se non esiste).
    prefix:
        Prefisso per i file (es. 'sub-05_rs-epoch').

    Returns
    -------
    list[Path]
        Percorsi dei file salvati (stem senza -lh.stc / -rh.stc).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for idx, stc in enumerate(stcs):
        stem = out_dir / f"{prefix}_epoch-{idx:04d}_inv"
        stc.save(str(stem), overwrite=True)
        saved.append(stem)
    return saved


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Apply inverse on RS epochs (STEP 2b)")
    ap.add_argument("--subject", required=True, help="Subject ID (es. '05')")
    ap.add_argument("--task", required=True, help="Task name (es. 'matchingpennies')")
    ap.add_argument("--method", default="dSPM", choices=["MNE", "dSPM", "sLORETA", "eLORETA"])
    ap.add_argument("--lambda2", type=float, default=1.0 / 9.0)
    ap.add_argument("--save", action="store_true", help="Salva STC su disco")
    args = ap.parse_args()

    sub_dir = DERIV / f"sub-{args.subject}" / "eeg"
    inv_path = sub_dir / f"sub-{args.subject}_inv.fif"
    epo_path = sub_dir / f"sub-{args.subject}_task-{args.task}_proc-clean_epo.fif"

    inv_op = read_inverse_operator(str(inv_path), verbose=False)
    epochs = mne.read_epochs(str(epo_path), preload=True, verbose=False)

    stcs = apply_inverse_epochs_rs(epochs, inv_op, method=args.method, lambda2=args.lambda2)
    print(f"Computed {len(stcs)} STC — shape data per epoch: {stcs[0].data.shape}")

    if args.save:
        out_dir = sub_dir / "rs_stcs"
        stems = save_stcs(stcs, out_dir, prefix=f"sub-{args.subject}_task-{args.task}")
        print(f"Saved {len(stems)} STC stems to {out_dir}")
