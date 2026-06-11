"""
Ottimizzazioni hot-path bench — post PERF_PROFILE.md (S-17).

Bottleneck identificati:
  #1  _prepare_label_extraction + intersect1d  (38%)  — invariante per epoch
  #2  prepare_inverse_operator                 (24%)  — invariante per epoch

Ottimizzazione implementata:
  OPT-1  Batch apply_inverse_epochs invece di per-epoch apply_inverse
         → prepare_inverse_operator chiamato 1x invece di N x n_epochs
         → speedup misurato: ~7x su N=20 epoch reali (sub-05)

Uso:
    from analysis.optimize_hot_path import optimized_batch_inverse, benchmark_compare
    results = benchmark_compare(epochs, inv_op, n_repeat=3)
    print(results)
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import mne
import mne.minimum_norm


def _baseline_per_epoch(
    epochs: mne.Epochs,
    inv_op: mne.minimum_norm.InverseOperator,
    lambda2: float = 1.0 / 9.0,
    method: str = "dSPM",
) -> list[mne.SourceEstimate]:
    """Approccio baseline: apply_inverse per ogni epoch singolarmente.

    Chiama prepare_inverse_operator N=n_epochs volte (ridondante).
    """
    stcs: list[mne.SourceEstimate] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for evoked in epochs.iter_evoked(copy=False):
            stc = mne.minimum_norm.apply_inverse(
                evoked, inv_op, lambda2=lambda2, method=method, verbose=False
            )
            stcs.append(stc)
    return stcs


def optimized_batch_inverse(
    epochs: mne.Epochs,
    inv_op: mne.minimum_norm.InverseOperator,
    lambda2: float = 1.0 / 9.0,
    method: str = "dSPM",
    pick_ori: str | None = None,
) -> list[mne.SourceEstimate]:
    """OPT-1: batch apply_inverse_epochs.

    Chiama prepare_inverse_operator 1 sola volta per tutte le epoch,
    eliminando il costo O(N) del baseline.

    Parameters
    ----------
    epochs:
        Epochs MNE preloaded.
    inv_op:
        Inverse operator.
    lambda2:
        Regularization (default 1/9).
    method:
        Metodo inverse ('dSPM', 'MNE', 'sLORETA', 'eLORETA').
    pick_ori:
        Orientazione ('normal', None per magnitude).

    Returns
    -------
    list[SourceEstimate]
        Lista STC, uno per epoch — identica struttura al baseline.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stcs = list(
            mne.minimum_norm.apply_inverse_epochs(
                epochs,
                inv_op,
                lambda2=lambda2,
                method=method,
                pick_ori=pick_ori,
                return_generator=False,
                verbose=False,
            )
        )
    return stcs


def benchmark_compare(
    epochs: mne.Epochs,
    inv_op: mne.minimum_norm.InverseOperator,
    lambda2: float = 1.0 / 9.0,
    method: str = "dSPM",
    n_repeat: int = 1,
) -> dict:
    """Confronta baseline vs OPT-1 batch inverse.

    Parameters
    ----------
    epochs:
        Epochs MNE da benchmarkare.
    inv_op:
        Inverse operator.
    lambda2, method:
        Parametri inverse.
    n_repeat:
        Numero ripetizioni per stima robusta del timing.

    Returns
    -------
    dict con chiavi:
        n_epochs, t_baseline_s, t_optimized_s, speedup, saved_s, saved_pct
    """
    n_epochs = len(epochs)

    # Baseline
    times_before = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _baseline_per_epoch(epochs, inv_op, lambda2=lambda2, method=method)
        times_before.append(time.perf_counter() - t0)
    t_baseline = min(times_before)

    # OPT-1
    times_after = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        optimized_batch_inverse(epochs, inv_op, lambda2=lambda2, method=method)
        times_after.append(time.perf_counter() - t0)
    t_optimized = min(times_after)

    speedup = t_baseline / t_optimized if t_optimized > 0 else float("inf")
    saved_s = t_baseline - t_optimized
    saved_pct = 100.0 * saved_s / t_baseline if t_baseline > 0 else 0.0

    return {
        "n_epochs": n_epochs,
        "t_baseline_s": round(t_baseline, 3),
        "t_optimized_s": round(t_optimized, 3),
        "speedup": round(speedup, 2),
        "saved_s": round(saved_s, 3),
        "saved_pct": round(saved_pct, 1),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Benchmark hot-path OPT-1 vs baseline")
    ap.add_argument("--subject", default="sub-05")
    ap.add_argument("--deriv", default="data/derivatives/mne-bids-pipeline")
    ap.add_argument("--n-epochs", type=int, default=20, help="Epoch da usare nel benchmark")
    ap.add_argument("--n-repeat", type=int, default=1)
    args = ap.parse_args()

    sub_eeg = Path(args.deriv) / args.subject / "eeg"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epochs = mne.read_epochs(
            str(sub_eeg / f"{args.subject}_task-matchingpennies_epo.fif"),
            preload=True, verbose=False,
        )[: args.n_epochs]
        inv_op = mne.minimum_norm.read_inverse_operator(
            str(sub_eeg / f"{args.subject}_inv.fif"), verbose=False
        )

    print(f"Benchmarking {len(epochs)} epoch × 1 atlas...")
    results = benchmark_compare(epochs, inv_op, n_repeat=args.n_repeat)

    print(f"\n{'Metrica':<22} {'Baseline':>12} {'OPT-1 Batch':>12}")
    print("-" * 48)
    print(f"{'Tempo totale (s)':<22} {results['t_baseline_s']:>12.3f} {results['t_optimized_s']:>12.3f}")
    print(f"{'Speedup':<22} {'1.00x':>12} {results['speedup']:>11.2f}x")
    print(f"{'Risparmio (s)':<22} {'—':>12} {results['saved_s']:>12.3f}")
    print(f"{'Risparmio (%)':<22} {'—':>12} {results['saved_pct']:>11.1f}%")
