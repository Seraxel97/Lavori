"""Benchmark tempi per step della pipeline matchingpennies."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import mne
import mne.minimum_norm
import numpy as np

from connectivity.fc_dispatcher import compute_fc
from features.dispatcher import build_X
from ml_training.ml_dispatcher import run_all_algorithms
from parcellation.extract_label_tc import extract_tc_batch


def _build_y(epochs: mne.Epochs) -> np.ndarray:
    """Costruisce array y dagli eventi."""
    label_map = {
        "raised-left/match-false": 0,
        "raised-left/match-true": 0,
        "raised-right/match-false": 1,
        "raised-right/match-true": 1,
    }
    y_list = []
    event_ids_inv = {v: k for k, v in epochs.event_id.items()}
    for ev_code in epochs.events[:, 2]:
        ev_name = event_ids_inv.get(ev_code, "unknown")
        y_list.append(label_map.get(ev_name, -1))
    return np.array(y_list, dtype=int)


def benchmark_pipeline(
    subject: str,
    deriv_root: str | Path,
    atlas: str = "aparc",
    metric: str = "wpli",
) -> dict:
    """Benchmarka tempi per step della pipeline e ritorna dict con risultati."""
    timings = {}
    deriv_root = Path(deriv_root)
    sub_eeg = deriv_root / subject / "eeg"

    # Step 1: Load epochs
    t0 = time.time()
    epo_path = sub_eeg / f"{subject}_task-matchingpennies_epo.fif"
    epochs = mne.read_epochs(str(epo_path), preload=True, verbose=False)
    sfreq_orig = epochs.info["sfreq"]
    epochs = epochs.decimate(int(sfreq_orig / 500.0))
    sfreq = epochs.info["sfreq"]
    epochs = epochs[:min(20, len(epochs))]
    timings["load_epochs"] = round(time.time() - t0, 3)

    # Step 2: Build y
    t0 = time.time()
    y = _build_y(epochs)
    valid_mask = y >= 0
    epochs = epochs[valid_mask]
    y = y[valid_mask]
    timings["build_y"] = round(time.time() - t0, 3)

    n_epochs = len(epochs)

    # Step 3: Inverse + parcellation (batch — apply_inverse_epochs 1×)
    t0 = time.time()
    inv_path = sub_eeg / f"{subject}_inv.fif"
    inv_op = mne.minimum_norm.read_inverse_operator(str(inv_path), verbose=False)
    src = inv_op["src"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stcs = list(
            mne.minimum_norm.apply_inverse_epochs(
                epochs, inv_op, lambda2=1.0 / 9.0, method="dSPM",
                pick_ori=None, return_generator=False, verbose=False,
            )
        )

    label_tc, _ = extract_tc_batch(stcs, atlas, src)
    timings["inverse_parcellation"] = round(time.time() - t0, 3)

    # Step 4: FC
    t0 = time.time()
    fc = compute_fc(
        label_tc,
        sfreq,
        metric,
        bands={"alpha": (8.0, 13.0), "beta": (13.0, 30.0)},
    )
    timings["functional_connectivity"] = round(time.time() - t0, 3)

    # Step 5: Feature extraction
    t0 = time.time()
    X, feat_names = build_X(label_tc, sfreq, fc)
    timings["feature_extraction"] = round(time.time() - t0, 3)

    # Step 6: ML
    t0 = time.time()
    n_splits = min(5, min(np.bincount(y)))
    results_ml = run_all_algorithms(X, y, groups=None, n_splits=n_splits)
    timings["ml_training"] = round(time.time() - t0, 3)

    return {
        "subject": subject,
        "atlas": atlas,
        "metric": metric,
        "n_epochs": n_epochs,
        "timings": timings,
        "ml_algos": list(results_ml.keys()),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Benchmark pipeline steps")
    ap.add_argument("--subject", default="sub-05")
    ap.add_argument("--deriv-root", default="data/eeg_matchingpennies")
    ap.add_argument("--atlas", default="aparc")
    ap.add_argument("--metric", default="wpli")
    ap.add_argument("--output", default="reports/PERF_BENCHMARK_raw.json")
    args = ap.parse_args()

    result = benchmark_pipeline(args.subject, args.deriv_root, args.atlas, args.metric)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Benchmark complete → {args.output}")
    print("Timings:")
    for step, elapsed in result["timings"].items():
        print(f"  {step}: {elapsed:.3f}s")
