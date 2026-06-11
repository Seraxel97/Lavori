"""
Benchmark matrix — STEP 8: 7 FC × 4 atlas × 5 ML × 5 bande su matchingpennies sub-05.

Combinazioni totali (full crossing): 7 × 4 × 5 × 5 = 700 run.

Strategia:
  - Outer loop: atlas (più costoso il setup; label_tc caricato 1× per atlas)
  - Inner loop: band → metric → algorithm
  - Per ogni (atlas, band, metric): computa FC 1× poi esegui 5 algoritmi ML
  - CV: StratifiedKFold k=5 (N=1 soggetto, no cross-subject)
  - Output: reports/BENCH_MATRIX_RESULTS.json + reports/BENCH_MATRIX_SUMMARY.md

Nota: per N=1 soggetto, BA non è scientificamente significativa.
      I numeri reali arrivano su dataset finale (S-07).

Uso:
  python -m pipeline_mne_bids.run_bench_matrix \\
      --subject sub-05 \\
      --deriv data/derivatives/mne-bids-pipeline \\
      --sfreq-target 500 \\
      --n-epochs-max 0   # 0=tutte; abbassa per smoke rapido
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mne
import mne.minimum_norm
import numpy as np

from connectivity.fc_dispatcher import compute_fc
from features.dispatcher import build_X
from ml_training.ml_dispatcher import run_all_algorithms
from parcellation.extract_label_tc import AtlasName, extract_tc

# ── Assi benchmark ────────────────────────────────────────────────────────────

ATLASES: list[AtlasName] = ["aparc", "destrieux", "schaefer100", "schaefer200"]

METRICS = ["coh", "imcoh", "plv", "ciplv", "pli", "wpli", "wpli2_debiased"]

ALGORITHMS = ["logreg", "svm", "mlp", "rf", "gb"]

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

_LABEL_MAP = {
    "raised-left/match-false": 0,
    "raised-left/match-true": 0,
    "raised-right/match-false": 1,
    "raised-right/match-true": 1,
}

_TZ = timezone(timedelta(hours=2))


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _config_hash(subject: str, sfreq_target: float, n_epochs_max: int) -> str:
    cfg = f"{subject}|{sfreq_target}|{n_epochs_max}"
    return hashlib.md5(cfg.encode()).hexdigest()[:8]


def _build_y(epochs: mne.Epochs) -> np.ndarray:
    event_ids_inv = {v: k for k, v in epochs.event_id.items()}
    return np.array(
        [_LABEL_MAP.get(event_ids_inv.get(ev, ""), -1) for ev in epochs.events[:, 2]],
        dtype=int,
    )


def _load_epochs_and_inv(
    sub_eeg: Path,
    subject: str,
    sfreq_target: float,
    n_epochs_max: int,
) -> tuple[mne.Epochs, mne.minimum_norm.InverseOperator, np.ndarray, float]:
    epo_path = sub_eeg / f"{subject}_task-matchingpennies_epo.fif"
    epochs = mne.read_epochs(str(epo_path), preload=True, verbose=False)

    if epochs.info["sfreq"] > sfreq_target:
        decim = int(epochs.info["sfreq"] / sfreq_target)
        epochs = epochs.decimate(decim)
    sfreq = epochs.info["sfreq"]

    if n_epochs_max and n_epochs_max < len(epochs):
        epochs = epochs[:n_epochs_max]

    y = _build_y(epochs)
    valid = y >= 0
    if not valid.all():
        epochs = epochs[valid]
        y = y[valid]

    inv_path = sub_eeg / f"{subject}_inv.fif"
    inv_op = mne.minimum_norm.read_inverse_operator(str(inv_path), verbose=False)
    return epochs, inv_op, y, sfreq


def _compute_label_tc(
    epochs: mne.Epochs,
    inv_op: mne.minimum_norm.InverseOperator,
    atlas: AtlasName,
    lambda2: float = 1.0 / 9.0,
    inv_method: str = "dSPM",
) -> tuple[np.ndarray, list[str]]:
    src = inv_op["src"]
    tc_list: list[np.ndarray] = []
    names_: list[str] = []
    for ep in epochs:
        stc = mne.minimum_norm.apply_inverse(
            ep.average(), inv_op, lambda2=lambda2, method=inv_method, verbose=False
        )
        tc_i, names_ = extract_tc(stc, atlas, src)
        tc_list.append(tc_i)
    return np.stack(tc_list, axis=0), names_


def run_bench(
    subject: str = "sub-05",
    deriv_root: str | Path = "data/derivatives/mne-bids-pipeline",
    *,
    sfreq_target: float = 500.0,
    n_epochs_max: int = 0,
    lambda2: float = 1.0 / 9.0,
    inv_method: str = "dSPM",
    n_cv_splits: int = 5,
    dump_every: int = 50,
    out_json: str | Path | None = None,
    out_summary: str | Path | None = None,
) -> list[dict]:
    """Esegue la matrice di benchmark completa.

    Parameters
    ----------
    subject:
        Subject ID (es. "sub-05").
    deriv_root:
        Root derivati mne-bids-pipeline.
    sfreq_target:
        Sfreq dopo decimazione.
    n_epochs_max:
        Max epoch (0=tutte).
    lambda2:
        Regularization inverse.
    inv_method:
        Metodo source estimate.
    n_cv_splits:
        Fold CV.
    dump_every:
        Dump incrementale JSON ogni N run.
    out_json:
        Path BENCH_MATRIX_RESULTS.json.
    out_summary:
        Path BENCH_MATRIX_SUMMARY.md.

    Returns
    -------
    list di dict run results.
    """
    deriv_root = Path(deriv_root)
    sub_eeg = deriv_root / subject / "eeg"

    if out_json is None:
        out_json = Path("reports") / "BENCH_MATRIX_RESULTS.json"
    if out_summary is None:
        out_summary = Path("reports") / "BENCH_MATRIX_SUMMARY.md"
    out_json = Path(out_json)
    out_summary = Path(out_summary)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    ts_start = datetime.now(_TZ).isoformat()
    meta: dict = {
        "dataset": "matchingpennies",
        "subject": subject,
        "sfreq_target": sfreq_target,
        "n_epochs_max": n_epochs_max,
        "n_runs": len(ATLASES) * len(BANDS) * len(METRICS) * len(ALGORITHMS),
        "ts_start": ts_start,
        "ts_end": None,
        "git_sha": _git_sha(),
        "config_hash": _config_hash(subject, sfreq_target, n_epochs_max),
    }

    print(f"[BENCH] carico epochs + inv per {subject}...")
    epochs, inv_op, y, sfreq = _load_epochs_and_inv(
        sub_eeg, subject, sfreq_target, n_epochs_max
    )
    n_splits = min(n_cv_splits, int(min(np.bincount(y))))
    print(f"[BENCH] {len(epochs)} epoch, sfreq={sfreq:.0f}, classes={np.bincount(y).tolist()}")
    print(f"[BENCH] atlanti×bande×metriche×algo = {meta['n_runs']} run totali")

    runs: list[dict] = []
    run_id = 0

    for atlas in ATLASES:
        print(f"\n[BENCH] === ATLAS: {atlas} ===")
        t_atlas = time.time()

        label_tc, _ = _compute_label_tc(epochs, inv_op, atlas, lambda2, inv_method)
        print(f"[BENCH] label_tc: {label_tc.shape} ({time.time()-t_atlas:.1f}s)")

        for band_name, (fmin, fmax) in BANDS.items():
            for metric in METRICS:
                try:
                    fc = compute_fc(
                        label_tc, sfreq, metric,  # type: ignore[arg-type]
                        bands={band_name: (fmin, fmax)},
                    )
                    X, _ = build_X(label_tc, sfreq, fc, include_univariate=False)
                    n_features = X.shape[1]
                    fc_ok = True
                    fc_err = None
                except Exception as exc:
                    fc_ok = False
                    fc_err = str(exc)
                    X = np.zeros((len(y), 1))
                    n_features = 0

                for algorithm in ALGORITHMS:
                    run_id += 1
                    t_run = time.time()
                    result: dict = {
                        "run_id": run_id,
                        "atlas": atlas,
                        "band": band_name,
                        "metric": metric,
                        "algorithm": algorithm,
                        "n_features": n_features,
                        "balanced_accuracy": None,
                        "ba_std": None,
                        "fit_time_s": None,
                        "error": fc_err,
                    }

                    if fc_ok:
                        try:
                            cv_res = run_all_algorithms(
                                X, y, groups=None,
                                algorithms=[algorithm],  # type: ignore[arg-type]
                                n_splits=n_splits,
                            )[algorithm]
                            elapsed = time.time() - t_run
                            result.update({
                                "balanced_accuracy": round(cv_res.ba_mean, 4),
                                "ba_std": round(cv_res.ba_std, 4),
                                "fit_time_s": round(elapsed, 2),
                                "error": None,
                            })
                        except Exception as exc:
                            result["error"] = str(exc)

                    runs.append(result)

                    if run_id % 10 == 0:
                        print(
                            f"[BENCH] {run_id}/{meta['n_runs']} "
                            f"{atlas}/{band_name}/{metric}/{algorithm}: "
                            f"BA={result['balanced_accuracy']}"
                        )

                    if run_id % dump_every == 0:
                        _dump_json(out_json, meta, runs)

    meta["ts_end"] = datetime.now(_TZ).isoformat()
    _dump_json(out_json, meta, runs)
    _write_summary(out_summary, meta, runs)
    print(f"\n[BENCH] DONE — {run_id} run | json → {out_json} | summary → {out_summary}")
    return runs


def _dump_json(path: Path, meta: dict, runs: list[dict]) -> None:
    data = {"meta": meta, "runs": runs}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _write_summary(path: Path, meta: dict, runs: list[dict]) -> None:
    valid = [r for r in runs if r["balanced_accuracy"] is not None]
    failed = [r for r in runs if r["error"] is not None]

    lines = [
        "# Bench Matrix Summary — matchingpennies",
        "",
        f"**Subject**: {meta['subject']} | **N run**: {meta['n_runs']}  ",
        f"**git**: {meta['git_sha']} | **config**: {meta['config_hash']}  ",
        f"**Start**: {meta['ts_start']} → **End**: {meta['ts_end']}  ",
        "",
        "## Top-10 by Balanced Accuracy",
        "",
        "| BA | Atlas | Band | Metric | Algorithm | N feat |",
        "|----|-------|------|--------|-----------|--------|",
    ]

    top10 = sorted(valid, key=lambda r: r["balanced_accuracy"], reverse=True)[:10]
    for r in top10:
        lines.append(
            f"| {r['balanced_accuracy']:.4f} | {r['atlas']} | {r['band']} "
            f"| {r['metric']} | {r['algorithm']} | {r['n_features']} |"
        )

    # Marginal per asse
    def _marginal(runs: list[dict], key: str) -> dict[str, float]:
        from collections import defaultdict
        sums: dict[str, list[float]] = defaultdict(list)
        for r in runs:
            if r["balanced_accuracy"] is not None:
                sums[r[key]].append(r["balanced_accuracy"])
        return {k: round(sum(v) / len(v), 4) for k, v in sums.items()}

    for axis_key, axis_label in [("metric", "FC Metric"), ("atlas", "Atlas"),
                                  ("algorithm", "Algorithm"), ("band", "Band")]:
        marginals = _marginal(valid, axis_key)
        lines += ["", f"### {axis_label} (marginal BA)"]
        lines += [f"- {k}: {v}" for k, v in sorted(marginals.items(), key=lambda x: -x[1])]

    if failed:
        lines += ["", f"## Failed runs ({len(failed)})", ""]
        for r in failed[:10]:
            lines.append(f"- run {r['run_id']} {r['atlas']}/{r['band']}/{r['metric']}: {r['error']}")

    wall = sum(r["fit_time_s"] or 0 for r in runs)
    lines += ["", "## Wall-clock budget", f"- Totale fit: {wall:.1f}s"]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bench matrix 700 run")
    ap.add_argument("--subject", default="sub-05")
    ap.add_argument("--deriv", default="data/derivatives/mne-bids-pipeline")
    ap.add_argument("--sfreq-target", type=float, default=500.0)
    ap.add_argument("--n-epochs-max", type=int, default=0)
    ap.add_argument("--n-cv-splits", type=int, default=5)
    ap.add_argument("--dump-every", type=int, default=50)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-summary", default=None)
    args = ap.parse_args()

    run_bench(
        subject=args.subject,
        deriv_root=args.deriv,
        sfreq_target=args.sfreq_target,
        n_epochs_max=args.n_epochs_max,
        n_cv_splits=args.n_cv_splits,
        dump_every=args.dump_every,
        out_json=args.out_json,
        out_summary=args.out_summary,
    )
