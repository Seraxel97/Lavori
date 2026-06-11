"""CLI entrypoint — Aggregate N=15 classification (S-AGG-N15).

Idempotente: se comparison_matrix_N15.json esiste con hash invariato, stampa
"[CACHED]" ed esce. Esecuzione su dati reali si attiva automaticamente quando
Step 5 N=15 è completato (X_*.npz con shape (30, *)).

Usage
-----
    python scripts/run_aggregate_n15.py \\
        --features data/features/ds005385 \\
        --out      data/results/ds005385

Opzioni
-------
    --atlases   aparc schaefer100         (default)
    --metrics   wpli coh                  (default)
    --band      alpha                     (default)
    --classifiers logreg svm_rbf lda      (default)
    --n-perm    1000                       (default)
    --random-state 42                      (default)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate N=15 classification — sprint S-AGG-N15",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--features", required=True, metavar="DIR",
                    help="directory con X_*.npz + y.npy + groups.npy")
    ap.add_argument("--out", required=True, metavar="DIR",
                    help="directory output per comparison_matrix_N15.json")
    ap.add_argument("--atlases", nargs="+", default=["aparc", "schaefer100"],
                    metavar="ATLAS")
    ap.add_argument("--metrics", nargs="+", default=["wpli", "coh"],
                    metavar="METRIC")
    ap.add_argument("--band", default="alpha")
    ap.add_argument("--classifiers", nargs="+", default=["logreg", "svm_rbf", "lda"],
                    metavar="CLF")
    ap.add_argument("--n-perm", type=int, default=1000, dest="n_perm")
    ap.add_argument("--random-state", type=int, default=42, dest="random_state")
    args = ap.parse_args()

    from ml_training.aggregate_n15 import aggregate_classify_n15  # noqa: PLC0415

    result = aggregate_classify_n15(
        features_dir=Path(args.features),
        out_dir=Path(args.out),
        atlases=args.atlases,
        metrics=args.metrics,
        band=args.band,
        classifiers=args.classifiers,
        n_permutations=args.n_perm,
        random_state=args.random_state,
    )

    if result.get("_cached"):
        print(f"[CACHED] Output aggiornato: {result['output_path']}")
        return 0

    if result["winner"] is None:
        print(
            "[WARN] Nessun dato N=15 caricato — Step 5 N=15 non ancora completato.",
            file=sys.stderr,
        )
        print("       Attendi che sonnet-tesi-1 completi S-FULL-N15 e riprova.")
        return 0

    w = result["winner"]
    print(f"Winner: {w['atlas']} × {w['metric']} × {w['classifier']}")
    print(
        f"  BA = {w['ba_mean']:.3f} ± {w['ba_std']:.3f} "
        f"CI[{w['ci_lo']:.3f}, {w['ci_hi']:.3f}]"
    )
    print(f"  p_perm = {w['p_perm']:.4f}  n_sub={w['n_subjects']}  n_feat={w['n_features']}")
    print(f"  Output → {result['output_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
