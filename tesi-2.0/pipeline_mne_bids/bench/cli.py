"""CLI argparse per bench matrix — entry point per run_bench_matrix.py."""

from __future__ import annotations

import argparse

from pipeline_mne_bids.bench.matrix import run_bench


def main() -> None:
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


if __name__ == "__main__":
    main()
