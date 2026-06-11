"""
Driver STEP 4: da epochs .fif + inverse operator → label TC → matrici FC.

Flusso:
  1. Carica epochs (pre-processate, mne-bids-pipeline output)
  2. Applica inverse operator su ogni epoch → STC array
  3. Parcellizza STC → label TC (n_epochs, n_labels, n_times)
  4. Calcola FC per tutte le bande e la metrica richiesta
  5. Salva .npz con arrays FC + nomi label

Uso:
  python -m connectivity.run_fc_on_epochs \\
      --epochs data/derivatives/.../sub-05_task-matchingpennies_epo.fif \\
      --inv    data/derivatives/.../sub-05_inv.fif \\
      --fwd    data/derivatives/.../sub-05_fwd.fif \\
      --atlas  schaefer100 \\
      --metric wpli \\
      --out    reports/fc_sub-05_schaefer100_wpli.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np

from connectivity.fc_dispatcher import Metric, compute_fc, save_fc
from parcellation.extract_label_tc import AtlasName, extract_tc


def epochs_to_label_tc(
    epochs: mne.Epochs,
    inv_op: mne.minimum_norm.InverseOperator,
    atlas: AtlasName,
    *,
    lambda2: float = 1.0 / 9.0,
    method: str = "dSPM",
    mode: str = "mean_flip",
) -> tuple[np.ndarray, list[str]]:
    """Applica inverse + parcellizza ogni epoch → (n_epochs, n_labels, n_times).

    Parameters
    ----------
    epochs:
        Epochs MNE già filtrate e pre-processate.
    inv_op:
        Inverse operator (mne.minimum_norm.InverseOperator).
    atlas:
        Chiave atlante per la parcellazione.
    lambda2:
        Regularization (default: 1/9).
    method:
        Metodo source estimate (dSPM, MNE, sLORETA).
    mode:
        Modalità aggregazione label (mean_flip).

    Returns
    -------
    label_tc : np.ndarray, shape (n_epochs, n_labels, n_times)
    names : list[str]
    """
    src = inv_op["src"]
    tc_list: list[np.ndarray] = []
    names: list[str] = []

    for i in range(len(epochs)):
        ep = epochs[i]
        stc = mne.minimum_norm.apply_inverse(
            ep.average(),
            inv_op,
            lambda2=lambda2,
            method=method,
            verbose=False,
        )
        tc_i, names = extract_tc(stc, atlas, src, mode=mode)
        tc_list.append(tc_i)  # (n_labels, n_times)

    return np.stack(tc_list, axis=0), names  # (n_epochs, n_labels, n_times)


def run(
    epochs_path: str | Path,
    inv_path: str | Path,
    fwd_path: str | Path,
    atlas: AtlasName,
    metric: Metric,
    out_path: str | Path,
    *,
    bands: dict[str, tuple[float, float]] | None = None,
    lambda2: float = 1.0 / 9.0,
    inv_method: str = "dSPM",
    mode_label: str = "mean_flip",
    mode_fc: str = "multitaper",
    n_jobs: int = 1,
) -> None:
    """Pipeline completa: epochs + inv → FC matrices .npz.

    Parameters
    ----------
    epochs_path:
        Path epochs .fif.
    inv_path:
        Path inverse operator *-inv.fif.
    fwd_path:
        Path forward solution *-fwd.fif (usato per src space della parcellazione).
    atlas:
        Chiave atlante.
    metric:
        Metrica FC (una tra: coh, imcoh, plv, ciplv, pli, wpli, wpli2_debiased).
    out_path:
        Path output .npz.
    bands:
        Override bande. None = DEFAULT_BANDS.
    lambda2:
        Regularization inverse.
    inv_method:
        Metodo source estimate.
    mode_label:
        Modalità aggregazione label.
    mode_fc:
        Modalità stima spettrale FC.
    n_jobs:
        Parallelismo FC.
    """
    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
    inv_op = mne.minimum_norm.read_inverse_operator(str(inv_path), verbose=False)
    fwd = mne.read_forward_solution(str(fwd_path), verbose=False)

    # Usa src dal fwd per la parcellazione (coerente con extract_tc_from_files)
    _ = fwd  # src viene estratto dall'inv_op internamente

    print(f"Epochs: {len(epochs)} epoche, sfreq={epochs.info['sfreq']} Hz")
    print(f"Atlas: {atlas}, metric: {metric}")

    label_tc, names = epochs_to_label_tc(
        epochs,
        inv_op,
        atlas,
        lambda2=lambda2,
        method=inv_method,
        mode=mode_label,
    )
    print(f"Label TC: shape {label_tc.shape}  ({len(names)} ROI)")

    fc = compute_fc(
        label_tc,
        sfreq=epochs.info["sfreq"],
        metric=metric,
        bands=bands,
        mode=mode_fc,
        n_jobs=n_jobs,
    )
    for band, mat in fc.items():
        print(f"  FC {band}: {mat.shape}, mean={mat[mat > 0].mean():.4f}")

    save_fc(fc, names, out_path)
    print(f"Salvato: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Epochs + inv → FC .npz")
    ap.add_argument("--epochs", required=True)
    ap.add_argument("--inv", required=True)
    ap.add_argument("--fwd", required=True)
    ap.add_argument("--atlas", required=True, choices=["aparc", "destrieux", "schaefer100", "schaefer200", "schaefer400"])
    ap.add_argument("--metric", required=True, choices=["coh", "imcoh", "plv", "ciplv", "pli", "wpli", "wpli2_debiased"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--lambda2", type=float, default=1.0 / 9.0)
    ap.add_argument("--inv-method", default="dSPM")
    ap.add_argument("--mode-fc", default="multitaper")
    ap.add_argument("--n-jobs", type=int, default=1)
    args = ap.parse_args()

    run(
        epochs_path=args.epochs,
        inv_path=args.inv,
        fwd_path=args.fwd,
        atlas=args.atlas,
        metric=args.metric,
        out_path=args.out,
        lambda2=args.lambda2,
        inv_method=args.inv_method,
        mode_fc=args.mode_fc,
        n_jobs=args.n_jobs,
    )
