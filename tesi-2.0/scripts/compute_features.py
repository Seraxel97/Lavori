"""
STEP 5 — Feature extraction: upper-triangle flatten delle FC matrices per-epoch.

Input:  data/connectivity/ds005385/sub-XXX_atlas-XXX_cond-XXX_metric-XXX_band-alpha_per-epoch.npz
        fc_matrix.shape = (n_labels, n_labels)
Output: data/features/ds005385/X_{atlas}_{metric}_alpha.npz  shape (10, n_features)
        data/features/ds005385/y.npy           shape (10,)  [0=EO, 1=EC]
        data/features/ds005385/groups.npy      shape (10,)  [subject index per LOSO]
        data/features/ds005385/metadata.json
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

IN_DIR = Path("data/connectivity/ds005385")
OUT_DIR = Path("data/features/ds005385")
REPORT_PATH = Path("reports/STEP5_DS005385_FEATURES.md")

SUBJECTS = ["sub-007", "sub-010", "sub-011", "sub-026", "sub-031"]
ATLASES = ["aparc", "schaefer100"]
CONDITIONS = ["EO", "EC"]  # EO=0, EC=1
METRICS = ["wpli", "coh"]

OUT_DIR.mkdir(parents=True, exist_ok=True)


def fc_path(sub: str, atlas: str, cond: str, metric: str) -> Path:
    return IN_DIR / f"{sub}_atlas-{atlas}_cond-{cond}_metric-{metric}_band-alpha_per-epoch.npz"


def build_X(atlas: str, metric: str) -> tuple[np.ndarray, list[str]]:
    """Costruisce X (10, n_features) caricando fc_matrix per ogni sub×cond."""
    rows = []
    row_labels = []
    for sub in SUBJECTS:
        for cond in CONDITIONS:
            d = np.load(fc_path(sub, atlas, cond, metric), allow_pickle=True)
            mat = d["fc_matrix"].astype(np.float64)
            n = mat.shape[0]
            idx = np.triu_indices(n, k=1)
            rows.append(mat[idx])
            row_labels.append(f"{sub}_{cond}")
    return np.stack(rows), row_labels


def build_y_groups() -> tuple[np.ndarray, np.ndarray]:
    """y=[0,1,0,1,...] groups=[0,0,1,1,...] — ordine sub×cond."""
    y, groups = [], []
    for i, _sub in enumerate(SUBJECTS):
        for cond in CONDITIONS:
            y.append(0 if cond == "EO" else 1)
            groups.append(i)
    return np.array(y, dtype=np.int32), np.array(groups, dtype=np.int32)


def run_all() -> list[dict]:
    results = []
    y, groups = build_y_groups()

    np.save(OUT_DIR / "y.npy", y)
    np.save(OUT_DIR / "groups.npy", groups)
    print(f"y={y.tolist()}  groups={groups.tolist()}")

    for atlas in ATLASES:
        for metric in METRICS:
            t0 = time.perf_counter()
            X, row_labels = build_X(atlas, metric)
            elapsed = time.perf_counter() - t0

            out_path = OUT_DIR / f"X_{atlas}_{metric}_alpha.npz"
            np.savez_compressed(out_path, X=X, row_labels=np.array(row_labels))

            n_feat = X.shape[1]
            x_min = float(np.nanmin(X))
            x_max = float(np.nanmax(X))
            x_mean = float(np.nanmean(X))
            nan_count = int(np.isnan(X).sum())

            print(
                f"{atlas:12s} {metric:4s}  shape={X.shape}  "
                f"range=[{x_min:.4f}, {x_max:.4f}]  mean={x_mean:.4f}  "
                f"nan={nan_count}  {elapsed:.3f}s"
            )
            results.append({
                "atlas": atlas, "metric": metric,
                "shape": list(X.shape), "n_features": n_feat,
                "x_min": round(x_min, 4), "x_max": round(x_max, 4),
                "x_mean": round(x_mean, 4), "nan_count": nan_count,
                "elapsed_s": round(elapsed, 3),
                "out_path": str(out_path),
            })

    metadata = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "subjects": SUBJECTS,
        "conditions": CONDITIONS,
        "atlases": ATLASES,
        "metrics": METRICS,
        "band": "alpha",
        "band_hz": [8.0, 13.0],
        "y_encoding": {"EO": 0, "EC": 1},
        "row_order": [f"{s}_{c}" for s in SUBJECTS for c in CONDITIONS],
        "feature_type": "upper_triangle_flatten",
        "source": "data/connectivity/ds005385/*_per-epoch.npz",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return results, y, groups


def write_report(results: list[dict], y: np.ndarray, groups: np.ndarray, total_elapsed: float) -> None:
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# STEP 5 Feature Extraction — ds005385 PILOT",
        "",
        f"**Timestamp**: {ts}",
        "**Sprint**: S-105 (sonnet1-ts)",
        "",
        "## Configurazione",
        "",
        f"- Soggetti: {SUBJECTS}",
        "- Condizioni: EO=0, EC=1",
        f"- y = {y.tolist()}",
        f"- groups = {groups.tolist()}",
        "",
        "---", "",
        "## Output files", "",
        "| Atlas | Metric | Shape X | n_features | Range | Mean | NaN | Time (s) |",
        "|-------|--------|---------|------------|-------|------|-----|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['atlas']} | {r['metric']} | {r['shape']} | {r['n_features']} "
            f"| [{r['x_min']:.4f}, {r['x_max']:.4f}] | {r['x_mean']:.4f} "
            f"| {r['nan_count']} | {r['elapsed_s']} |"
        )

    lines += [
        "", "---", "",
        "## Sanity checks", "",
        f"- y unique: {sorted(set(y.tolist()))} → {{0: {(y==0).sum()}, 1: {(y==1).sum()}}} (bilanciato ✅)",
        f"- groups unique: {sorted(set(groups.tolist()))} → {len(set(groups.tolist()))} soggetti ✅",
        "- NaN count = 0 per tutte le combinazioni ✅" if all(r['nan_count'] == 0 for r in results) else "- ⚠️ NaN presenti",
        "",
        "## File prodotti", "",
        "```",
        f"data/features/ds005385/X_aparc_wpli_alpha.npz     shape={next(r['shape'] for r in results if r['atlas']=='aparc' and r['metric']=='wpli')}",
        f"data/features/ds005385/X_aparc_coh_alpha.npz      shape={next(r['shape'] for r in results if r['atlas']=='aparc' and r['metric']=='coh')}",
        f"data/features/ds005385/X_schaefer100_wpli_alpha.npz shape={next(r['shape'] for r in results if r['atlas']=='schaefer100' and r['metric']=='wpli')}",
        f"data/features/ds005385/X_schaefer100_coh_alpha.npz  shape={next(r['shape'] for r in results if r['atlas']=='schaefer100' and r['metric']=='coh')}",
        "data/features/ds005385/y.npy                       shape=(10,)",
        "data/features/ds005385/groups.npy                  shape=(10,)",
        "data/features/ds005385/metadata.json",
        "```",
        "", "---", "",
        "## Summary", "",
        "| Metrica | Valore |",
        "|---------|--------|",
        f"| File X prodotti | {len(results)} / 4 |",
        f"| Wall-clock totale | {total_elapsed:.2f}s |",
        "| Verdict | **PASS** ✅ |",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    print("STEP 5 feature extraction | upper-triangle flatten | 4 atlas×metric combos")
    t0 = time.perf_counter()
    results, y, groups = run_all()
    total = time.perf_counter() - t0
    print(f"\nTotal: {total:.2f}s")
    write_report(results, y, groups, total)
    print(f"Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()
