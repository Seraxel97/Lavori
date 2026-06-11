"""CLI entry-point E2E pipeline matchingpennies."""

from __future__ import annotations

import argparse

from pipeline_mne_bids.run_e2e.exec import run_e2e


def main() -> None:
    """Punto di ingresso CLI: parse args e lancia run_e2e."""
    ap = argparse.ArgumentParser(description="E2E pipeline matchingpennies")
    ap.add_argument("--subject", default="sub-05")
    ap.add_argument("--deriv", default="data/derivatives/mne-bids-pipeline")
    ap.add_argument(
        "--atlas", default="aparc", choices=["aparc", "destrieux", "schaefer100", "schaefer200"]
    )
    ap.add_argument(
        "--metric",
        default="wpli",
        choices=["coh", "imcoh", "plv", "ciplv", "pli", "wpli", "wpli2_debiased"],
    )
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
