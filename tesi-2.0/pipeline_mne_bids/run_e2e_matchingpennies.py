"""
E2E pipeline matchingpennies — STEP 1-6 smoke test su sub-05.

Flusso:
  1. Carica epochs (mne-bids-pipeline output), decima a sfreq_target
  2. Applica inverse operator per epoch → STC
  3. Parcellizza STC → label TC (n_epochs, n_labels, n_times)
  4. Compute FC (metrica + bande richieste)
  5. Estrae feature (univariate + FC-flatten)
  6. ML: StratifiedKFold (N=1 soggetto, no cross-subject) + balanced accuracy
  7. Scrive reports/E2E_MATCHINGPENNIES.md

Classi target:
  0 = raised-left (raised-left/match-false + raised-left/match-true)
  1 = raised-right (raised-right/match-false + raised-right/match-true)

Uso:
  python -m pipeline_mne_bids.run_e2e_matchingpennies \\
      --subject sub-05 \\
      --deriv data/derivatives/mne-bids-pipeline \\
      --atlas aparc \\
      --metric wpli \\
      --n-epochs-max 20    # per smoke rapido; 0=tutti
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mne
import mne.minimum_norm
import numpy as np

from connectivity.fc_dispatcher import compute_fc
from features.dispatcher import build_X
from ml_training.ml_dispatcher import run_all_algorithms
from parcellation.extract_label_tc import AtlasName, extract_tc

_LABEL_MAP = {
    "raised-left/match-false": 0,
    "raised-left/match-true": 0,
    "raised-right/match-false": 1,
    "raised-right/match-true": 1,
}


def _build_y(epochs: mne.Epochs) -> np.ndarray:
    """Costruisce array y (0=raised-left, 1=raised-right) dagli eventi."""
    y_list = []
    event_ids_inv = {v: k for k, v in epochs.event_id.items()}
    for ev_code in epochs.events[:, 2]:
        ev_name = event_ids_inv.get(ev_code, "unknown")
        y_list.append(_LABEL_MAP.get(ev_name, -1))
    return np.array(y_list, dtype=int)


def run_e2e(
    subject: str,
    deriv_root: str | Path,
    atlas: AtlasName = "aparc",
    metric: str = "wpli",
    *,
    sfreq_target: float = 500.0,
    n_epochs_max: int = 0,
    bands: dict[str, tuple[float, float]] | None = None,
    lambda2: float = 1.0 / 9.0,
    inv_method: str = "dSPM",
    n_cv_splits: int = 5,
    out_report: str | Path | None = None,
) -> dict:
    """Esegue la pipeline E2E e ritorna un dizionario risultati.

    Parameters
    ----------
    subject:
        Subject ID (es. "sub-05").
    deriv_root:
        Root directory derivati mne-bids-pipeline.
    atlas:
        Atlante parcellazione.
    metric:
        Metrica FC.
    sfreq_target:
        Frequenza di campionamento dopo decimazione (Hz).
    n_epochs_max:
        Massimo numero epoch da usare (0 = tutte).
    bands:
        Override bande FC. None = DEFAULT_BANDS.
    lambda2:
        Regularization inverse.
    inv_method:
        Metodo source estimate.
    n_cv_splits:
        Fold per CV.
    out_report:
        Path del report MD. None = reports/E2E_MATCHINGPENNIES.md.

    Returns
    -------
    dict con campi: subject, atlas, metric, n_epochs, n_features,
        ba_mean_{algo}, ba_std_{algo}, ...
    """
    t_start = time.time()
    deriv_root = Path(deriv_root)
    sub_eeg = deriv_root / subject / "eeg"

    # ── 1. Carica epochs ─────────────────────────────────────────────────────
    epo_path = sub_eeg / f"{subject}_task-matchingpennies_epo.fif"
    epochs = mne.read_epochs(str(epo_path), preload=True, verbose=False)

    # Decima se necessario
    if epochs.info["sfreq"] > sfreq_target:
        decim = int(epochs.info["sfreq"] / sfreq_target)
        epochs = epochs.decimate(decim)
    sfreq = epochs.info["sfreq"]

    if n_epochs_max and n_epochs_max < len(epochs):
        epochs = epochs[:n_epochs_max]

    # ── 2. Label y ────────────────────────────────────────────────────────────
    y = _build_y(epochs)
    valid_mask = y >= 0
    if not valid_mask.all():
        epochs = epochs[valid_mask]
        y = y[valid_mask]

    n_ep = len(epochs)
    print(f"[E2E] {subject} | {n_ep} epoch | sfreq={sfreq:.0f} | atlas={atlas} | metric={metric}")
    print(f"[E2E] classi: {np.bincount(y).tolist()}")

    # ── 3. Inverse + parcellazione ────────────────────────────────────────────
    inv_path = sub_eeg / f"{subject}_inv.fif"
    inv_op = mne.minimum_norm.read_inverse_operator(str(inv_path), verbose=False)
    src = inv_op["src"]

    tc_list: list[np.ndarray] = []
    names_: list[str] = []
    for i in range(len(epochs)):
        evoked = epochs[i].average()
        stc = mne.minimum_norm.apply_inverse(
            evoked, inv_op, lambda2=lambda2, method=inv_method, verbose=False
        )
        tc_i, names_ = extract_tc(stc, atlas, src)
        tc_list.append(tc_i)

    label_tc = np.stack(tc_list, axis=0)  # (n_epochs, n_labels, n_times)
    print(f"[E2E] label TC: {label_tc.shape}")

    # ── 4. FC ────────────────────────────────────────────────────────────────
    active_bands = bands or {
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
    }
    fc = compute_fc(label_tc, sfreq, metric, bands=active_bands)  # type: ignore[arg-type]
    for b, mat in fc.items():
        print(f"[E2E] FC {b}: {mat.shape}, mean_upper={mat[mat > 0].mean():.4f}")

    # ── 5. Feature extraction ──────────────────────────────────────────────────
    X, feat_names = build_X(label_tc, sfreq, fc)
    print(f"[E2E] X: {X.shape} ({len(feat_names)} feature)")

    # ── 6. ML ─────────────────────────────────────────────────────────────────
    n_splits = min(n_cv_splits, min(np.bincount(y)))  # non più split del min class
    results_ml = run_all_algorithms(X, y, groups=None, n_splits=n_splits)

    t_elapsed = time.time() - t_start
    print(f"[E2E] elapsed: {t_elapsed:.1f}s")
    for algo, r in results_ml.items():
        print(f"[E2E]   {algo}: BA={r.ba_mean:.3f} ± {r.ba_std:.3f}")

    # ── 7. Report ─────────────────────────────────────────────────────────────
    summary = {
        "subject": subject,
        "atlas": atlas,
        "metric": metric,
        "n_epochs": n_ep,
        "n_labels": label_tc.shape[1],
        "n_times": label_tc.shape[2],
        "sfreq": sfreq,
        "n_features": X.shape[1],
        "elapsed_s": round(t_elapsed, 2),
        "bands": list(active_bands.keys()),
        "results": {
            algo: {"ba_mean": r.ba_mean, "ba_std": r.ba_std, "ba_per_fold": r.ba_per_fold}
            for algo, r in results_ml.items()
        },
    }

    if out_report is None:
        out_report = Path("reports") / "E2E_MATCHINGPENNIES.md"
    out_report = Path(out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    _write_report(summary, out_report)
    print(f"[E2E] report → {out_report}")

    return summary


def _write_report(summary: dict, path: Path) -> None:
    lines = [
        "# E2E Smoke Test — matchingpennies",
        "",
        f"**Subject**: {summary['subject']}  ",
        f"**Atlas**: {summary['atlas']}  ",
        f"**FC metric**: {summary['metric']}  ",
        f"**Bande**: {', '.join(summary['bands'])}  ",
        f"**N epoch**: {summary['n_epochs']} | sfreq={summary['sfreq']:.0f} Hz  ",
        f"**N labels**: {summary['n_labels']} | N features: {summary['n_features']}  ",
        f"**Elapsed**: {summary['elapsed_s']}s  ",
        "",
        "## Risultati ML (StratifiedKFold)",
        "",
        "| Algoritmo | BA mean | BA std | Fold BAs |",
        "|-----------|---------|--------|----------|",
    ]
    for algo, r in summary["results"].items():
        fold_str = " | ".join(f"{b:.3f}" for b in r["ba_per_fold"])
        lines.append(f"| {algo} | {r['ba_mean']:.3f} | {r['ba_std']:.3f} | {fold_str} |")

    lines += [
        "",
        "> **Nota**: N=1 soggetto → risultati non significativi scientificamente.",
        "> Pipeline smoke test: verifica che tutti i moduli si interfaccino correttamente.",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="E2E pipeline matchingpennies")
    ap.add_argument("--subject", default="sub-05")
    ap.add_argument("--deriv", default="data/derivatives/mne-bids-pipeline")
    ap.add_argument("--atlas", default="aparc", choices=["aparc", "destrieux", "schaefer100", "schaefer200"])
    ap.add_argument("--metric", default="wpli", choices=["coh", "imcoh", "plv", "ciplv", "pli", "wpli", "wpli2_debiased"])
    ap.add_argument("--n-epochs-max", type=int, default=0, help="0=tutte")
    ap.add_argument("--sfreq-target", type=float, default=500.0)
    ap.add_argument("--n-cv-splits", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_e2e(
        subject=args.subject,
        deriv_root=args.deriv,
        atlas=args.atlas,
        metric=args.metric,
        sfreq_target=args.sfreq_target,
        n_epochs_max=args.n_epochs_max,
        n_cv_splits=args.n_cv_splits,
        out_report=args.out,
    )
