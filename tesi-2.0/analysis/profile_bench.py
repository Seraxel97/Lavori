"""
cProfile del hot path del benchmark S-08.

Sub-set ridotto per profiling rapido: 1 atlas × 2 metriche × 1 algo × 2 bande = 4 run.
Output: reports/PROFILE_STATS.txt (top-30 per cumtime) + stampa top-10 a console.

Uso:
    python analysis/profile_bench.py [--n-epochs-max N] [--out-stats PATH]
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

import mne
import numpy as np

from connectivity.fc_dispatcher import compute_fc
from features.dispatcher import build_X
from ml_training.ml_dispatcher import run_cv
from parcellation.extract_label_tc import extract_tc

# ── Sub-set per profiling ─────────────────────────────────────────────────────
_ATLAS: str = "aparc"          # 68 ROI — più veloce
_METRICS = ["wpli", "plv"]    # 2 metriche
_ALGO: str = "logreg"
_BANDS = {
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
_SUBJECT = "sub-05"
_DERIV = Path("data/derivatives/mne-bids-pipeline")

_LABEL_MAP = {
    "raised-left/match-false": 0,
    "raised-left/match-true": 0,
    "raised-right/match-false": 1,
    "raised-right/match-true": 1,
}


def _build_y(epochs: mne.Epochs) -> np.ndarray:
    event_ids_inv = {v: k for k, v in epochs.event_id.items()}
    return np.array(
        [_LABEL_MAP.get(event_ids_inv.get(ev, ""), -1) for ev in epochs.events[:, 2]],
        dtype=int,
    )


def _bench_subset(n_epochs_max: int = 20) -> None:
    """Esegue il sub-set di benchmark (hot path da profilare)."""
    import mne
    import mne.minimum_norm

    sub_eeg = _DERIV / _SUBJECT / "eeg"

    # Carica epochs
    epo_path = sub_eeg / f"{_SUBJECT}_task-matchingpennies_epo.fif"
    epochs = mne.read_epochs(str(epo_path), preload=True, verbose=False)
    decim = int(epochs.info["sfreq"] / 500.0)
    epochs = epochs.decimate(decim)
    if n_epochs_max:
        epochs = epochs[:n_epochs_max]
    sfreq = epochs.info["sfreq"]

    y = _build_y(epochs)
    valid = y >= 0
    epochs = epochs[valid]
    y = y[valid]

    inv_op = mne.minimum_norm.read_inverse_operator(
        str(sub_eeg / f"{_SUBJECT}_inv.fif"), verbose=False
    )
    src = inv_op["src"]

    # ── HOT PATH 1: apply_inverse + extract_tc per epoch ────────────────────
    tc_list: list[np.ndarray] = []
    names_: list[str] = []
    for i in range(len(epochs)):
        ep = epochs[i]  # single-epoch Epochs object (supports .average())
        stc = mne.minimum_norm.apply_inverse(
            ep.average(), inv_op, lambda2=1.0 / 9.0, method="dSPM", verbose=False
        )
        tc_i, names_ = extract_tc(stc, _ATLAS, src)  # type: ignore[arg-type]
        tc_list.append(tc_i)
    label_tc = np.stack(tc_list, axis=0)

    # ── HOT PATH 2: compute_fc per metrica × banda ───────────────────────────
    for metric in _METRICS:
        for band_name, (fmin, fmax) in _BANDS.items():
            fc = compute_fc(
                label_tc, sfreq, metric,  # type: ignore[arg-type]
                bands={band_name: (fmin, fmax)},
            )

            # ── HOT PATH 3: build_X + run_cv ────────────────────────────────
            X, _ = build_X(label_tc, sfreq, fc, include_univariate=False)
            n_splits = min(3, int(min(np.bincount(y))))
            run_cv(X, y, groups=None, algorithm=_ALGO, n_splits=n_splits)  # type: ignore[arg-type]


def profile_and_report(
    n_epochs_max: int = 20,
    out_stats: Path = Path("reports/PROFILE_STATS.txt"),
    top_n: int = 30,
) -> pstats.Stats:
    """Esegue cProfile sul subset e salva i risultati.

    Parameters
    ----------
    n_epochs_max:
        Numero massimo di epoch (default 20 per run rapido).
    out_stats:
        Path file testo stats (default reports/PROFILE_STATS.txt).
    top_n:
        Numero di funzioni top da salvare (per cumtime).

    Returns
    -------
    pstats.Stats oggetto con le statistiche.
    """
    out_stats = Path(out_stats)
    out_stats.parent.mkdir(parents=True, exist_ok=True)

    print(f"[PERF] profiling subset: atlas={_ATLAS}, metrics={_METRICS}, "
          f"algo={_ALGO}, bande={list(_BANDS)}, n_epochs_max={n_epochs_max}")
    t0 = time.time()

    pr = cProfile.Profile()
    pr.enable()
    _bench_subset(n_epochs_max=n_epochs_max)
    pr.disable()

    elapsed = time.time() - t0
    print(f"[PERF] wall-clock: {elapsed:.1f}s")

    # Dump top-N per cumtime
    buf = io.StringIO()
    stats = pstats.Stats(pr, stream=buf)
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)
    txt = buf.getvalue()

    out_stats.write_text(txt)
    print(f"[PERF] stats salvate in {out_stats}")

    # Stampa top-10 a console
    buf2 = io.StringIO()
    stats2 = pstats.Stats(pr, stream=buf2)
    stats2.sort_stats("cumulative")
    stats2.print_stats(10)
    print("\n[PERF] TOP-10 per cumulative time:\n")
    # Filtra le righe di intestazione e le prime 10 funzioni
    for line in buf2.getvalue().splitlines()[7:17]:
        print(line)

    return stats


def _parse_top_stats(stats_txt: str, n: int = 10) -> list[dict]:
    """Estrae le top-N funzioni dal testo pstats.

    Returns
    -------
    list di dict con keys: ncalls, tottime, cumtime, func
    """
    rows = []
    in_data = False
    for line in stats_txt.splitlines():
        line = line.strip()
        if "ncalls" in line and "tottime" in line:
            in_data = True
            continue
        if not in_data or not line:
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        try:
            rows.append({
                "ncalls": parts[0],
                "tottime": float(parts[1]),
                "percall_tot": float(parts[2]),
                "cumtime": float(parts[3]),
                "percall_cum": float(parts[4]),
                "func": parts[5].strip(),
            })
        except ValueError:
            continue
        if len(rows) >= n:
            break
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="cProfile bench S-08 hot path")
    ap.add_argument("--n-epochs-max", type=int, default=20)
    ap.add_argument("--out-stats", default="reports/PROFILE_STATS.txt")
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    profile_and_report(
        n_epochs_max=args.n_epochs_max,
        out_stats=Path(args.out_stats),
        top_n=args.top_n,
    )
