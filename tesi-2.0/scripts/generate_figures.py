"""S-FIG-N15 — Figure generation for ds005385 classification results.

Produces 6 paper-ready PNG figures (300 DPI) from comparison_matrix JSON.
Handles both PILOT schema (nested dict) and N=15 schema (results list from
aggregate_classify_n15). Idempotent: skips figures whose mtime > input mtime.

Usage
-----
    # PILOT N=5 validation run (uses existing comparison_matrix.json)
    python scripts/generate_figures.py \\
        --comparison data/results/ds005385/comparison_matrix.json \\
        --features-dir data/features/ds005385 \\
        --fc-dir data/connectivity/ds005385 \\
        --out-dir reports/figures

    # N=15 run (after aggregate_classify_n15 completes)
    python scripts/generate_figures.py \\
        --comparison data/results/ds005385/comparison_matrix_N15.json \\
        --features-dir data/features/ds005385 \\
        --fc-dir data/connectivity/ds005385 \\
        --out-dir reports/figures

Figures produced
----------------
    fig_balacc_bar.png      Balanced accuracy bar chart (grouped by atlas)
    fig_balacc_heatmap.png  Balanced accuracy heatmap (atlas×metric × classifier)
    fig_pperm_bar.png       -log10(p_perm) bar chart with significance threshold
    fig_fc_avg.png          Average FC matrices (2×2: aparc/schaefer100 × EO/EC)
    fig_X_pca2.png          PCA 2-D scatter of best feature set
    fig_summary_table.png   Top-3 configurations table
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.style.use("default")

_FIGSIZE_WIDE = (10, 5)
_FIGSIZE_SQ = (7, 7)
_DPI = 300
_SAVEKW = {"dpi": _DPI, "bbox_inches": "tight"}


# ---------------------------------------------------------------------------
# Data model — normalise both schemas
# ---------------------------------------------------------------------------

@dataclass
class ConfigResult:
    atlas: str
    metric: str
    classifier: str
    bal_acc: float
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")
    p_perm: float = float("nan")
    n_subjects: int = 0


@dataclass
class ComparisonData:
    results: list[ConfigResult]
    source_file: str
    n_subjects: int
    schema: str  # "pilot" | "n15"
    winner: ConfigResult | None = None
    extra: dict = field(default_factory=dict)


def _parse_pilot(data: dict, source: str) -> ComparisonData:
    """Parse old PILOT nested-dict schema."""
    results = []
    matrix = data.get("matrix", {})
    for atlas, metrics_d in matrix.items():
        for metric, clfs_d in metrics_d.items():
            for clf, scores in clfs_d.items():
                results.append(ConfigResult(
                    atlas=atlas,
                    metric=metric,
                    classifier=clf,
                    bal_acc=float(scores.get("bal_acc", scores.get("acc", float("nan")))),
                    n_subjects=int(data.get("description", "n=10").split("n=")[-1].split()[0])
                    if "n=" in data.get("description", "") else 5,
                ))
    winner = max(results, key=lambda r: r.bal_acc) if results else None
    return ComparisonData(results=results, source_file=source, n_subjects=winner.n_subjects if winner else 5,
                          schema="pilot", winner=winner)


def _parse_n15(data: dict, source: str) -> ComparisonData:
    """Parse N=15 aggregate schema (aggregate_classify_n15 output)."""
    results = []
    for r in data.get("results", []):
        results.append(ConfigResult(
            atlas=r["atlas"],
            metric=r["metric"],
            classifier=r["classifier"],
            bal_acc=float(r["ba_mean"]),
            ci_lo=float(r.get("ci_lo", float("nan"))),
            ci_hi=float(r.get("ci_hi", float("nan"))),
            p_perm=float(r.get("p_perm", float("nan"))),
            n_subjects=int(r.get("n_subjects", 15)),
        ))
    winner_d = data.get("winner")
    winner = None
    if winner_d:
        winner = ConfigResult(
            atlas=winner_d["atlas"],
            metric=winner_d["metric"],
            classifier=winner_d["classifier"],
            bal_acc=float(winner_d["ba_mean"]),
            ci_lo=float(winner_d.get("ci_lo", float("nan"))),
            ci_hi=float(winner_d.get("ci_hi", float("nan"))),
            p_perm=float(winner_d.get("p_perm", float("nan"))),
            n_subjects=int(winner_d.get("n_subjects", 15)),
        )
    n_sub = results[0].n_subjects if results else 15
    return ComparisonData(results=results, source_file=source, n_subjects=n_sub,
                          schema="n15", winner=winner)


def load_comparison(path: Path) -> ComparisonData:
    """Load comparison JSON and dispatch to the right parser."""
    data = json.loads(path.read_text())
    source = str(path)
    if "results" in data:
        return _parse_n15(data, source)
    if "matrix" in data:
        return _parse_pilot(data, source)
    warnings.warn(f"Unknown comparison schema in {path} — trying best-effort parse", stacklevel=2)
    return _parse_pilot(data, source)


# ---------------------------------------------------------------------------
# Idempotency helper
# ---------------------------------------------------------------------------

def _needs_update(out_path: Path, *inputs: Path, force: bool = False) -> bool:
    if force or not out_path.exists():
        return True
    out_mtime = out_path.stat().st_mtime
    return any(p.exists() and p.stat().st_mtime > out_mtime for p in inputs)


# ---------------------------------------------------------------------------
# Figure 1 — Balanced accuracy bar chart
# ---------------------------------------------------------------------------

def fig_balacc_bar(cd: ComparisonData, out: Path, *, force: bool = False) -> None:
    if not _needs_update(out, Path(cd.source_file), force=force):
        return

    results = cd.results
    atlases = sorted({r.atlas for r in results})
    metrics = sorted({r.metric for r in results})
    classifiers = sorted({r.classifier for r in results})

    n_groups = len(atlases) * len(metrics)
    n_clfs = len(classifiers)
    x = np.arange(n_groups)
    width = 0.8 / n_clfs
    colors = plt.cm.tab10(np.linspace(0, 0.5, n_clfs))

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    group_labels = [f"{a}\n{m}" for a in atlases for m in metrics]

    for ci, clf in enumerate(classifiers):
        ba_vals, err_lo, err_hi = [], [], []
        for atlas in atlases:
            for metric in metrics:
                matches = [r for r in results if r.atlas == atlas
                           and r.metric == metric and r.classifier == clf]
                ba = matches[0].bal_acc if matches else float("nan")
                lo = matches[0].ci_lo if matches else float("nan")
                hi = matches[0].ci_hi if matches else float("nan")
                ba_vals.append(ba)
                err_lo.append(ba - lo if not np.isnan(lo) else 0)
                err_hi.append(hi - ba if not np.isnan(hi) else 0)

        offset = (ci - n_clfs / 2 + 0.5) * width
        has_ci = not all(np.isnan(err_lo))
        yerr = np.array([err_lo, err_hi]) if has_ci else None
        ax.bar(x + offset, ba_vals, width, label=clf, color=colors[ci], yerr=yerr,
               capsize=3, error_kw={"linewidth": 1})

    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(f"Balanced Accuracy — {cd.schema.upper()} N={cd.n_subjects}")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, **_SAVEKW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Balanced accuracy heatmap
# ---------------------------------------------------------------------------

def fig_balacc_heatmap(cd: ComparisonData, out: Path, *, force: bool = False) -> None:
    if not _needs_update(out, Path(cd.source_file), force=force):
        return

    atlases = sorted({r.atlas for r in cd.results})
    metrics = sorted({r.metric for r in cd.results})
    classifiers = sorted({r.classifier for r in cd.results})
    rows = [f"{a} × {m}" for a in atlases for m in metrics]

    mat = np.full((len(rows), len(classifiers)), float("nan"))
    for i, (atlas, metric) in enumerate((a, m) for a in atlases for m in metrics):
        for j, clf in enumerate(classifiers):
            matches = [r for r in cd.results if r.atlas == atlas
                       and r.metric == metric and r.classifier == clf]
            if matches:
                mat[i, j] = matches[0].bal_acc

    fig, ax = plt.subplots(figsize=(max(6, len(classifiers) * 2), max(4, len(rows) * 1.2)))
    im = ax.imshow(mat, cmap="viridis", vmin=0.4, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Balanced Accuracy")

    ax.set_xticks(range(len(classifiers)))
    ax.set_xticklabels(classifiers, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_title(f"Balanced Accuracy Heatmap — {cd.schema.upper()} N={cd.n_subjects}")

    for i in range(len(rows)):
        for j in range(len(classifiers)):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if val < 0.7 else "black")

    fig.tight_layout()
    fig.savefig(out, **_SAVEKW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — -log10(p_perm) bar chart
# ---------------------------------------------------------------------------

def fig_pperm_bar(cd: ComparisonData, out: Path, *, force: bool = False) -> None:
    if not _needs_update(out, Path(cd.source_file), force=force):
        return

    results_with_p = [r for r in cd.results if not np.isnan(r.p_perm)]
    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)

    if not results_with_p:
        ax.text(0.5, 0.5, "p_perm non disponibile\n(schema PILOT — nessun permutation test)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
        ax.set_title(f"Permutation p-value — {cd.schema.upper()} N={cd.n_subjects}")
    else:
        labels = [f"{r.atlas}\n{r.metric}\n{r.classifier}" for r in results_with_p]
        neg_log_p = [-np.log10(max(r.p_perm, 1e-4)) for r in results_with_p]
        colors = ["#d62728" if v >= -np.log10(0.05) else "#aec7e8" for v in neg_log_p]

        x = np.arange(len(labels))
        ax.bar(x, neg_log_p, color=colors)
        ax.axhline(-np.log10(0.05), linestyle="--", color="red", linewidth=1,
                   label=f"p=0.05 threshold ({-np.log10(0.05):.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("-log₁₀(p_perm)")
        ax.set_title(f"Permutation Test — {cd.schema.upper()} N={cd.n_subjects}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, **_SAVEKW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Average FC matrices (2×2)
# ---------------------------------------------------------------------------

def fig_fc_avg(fc_dir: Path, out: Path, *, force: bool = False,
               metric: str = "coh", band: str = "alpha") -> None:
    if not _needs_update(out, force=force):
        return

    subjects = [
        "sub-007", "sub-010", "sub-011", "sub-026", "sub-031",
        "sub-033", "sub-041", "sub-066", "sub-071", "sub-080",
        "sub-125", "sub-157", "sub-169", "sub-185", "sub-195",
    ]
    atlases = ["aparc", "schaefer100"]
    conditions = ["EO", "EC"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    vmin, vmax = 0.0, 1.0 if metric == "coh" else 0.5
    cmap = "viridis"

    for ai, atlas in enumerate(atlases):
        for ci, cond in enumerate(conditions):
            mats = []
            for sub in subjects:
                fname = (f"{sub}_atlas-{atlas}_cond-{cond}"
                         f"_metric-{metric}_band-{band}_per-epoch.npz")
                fp = fc_dir / fname
                if fp.exists():
                    m = np.load(fp, allow_pickle=False)["fc_matrix"]
                    mats.append(m)
            ax = axes[ai][ci]
            if mats:
                avg = np.mean(np.stack(mats, axis=0), axis=0)
                np.fill_diagonal(avg, 0)
                ax.matshow(avg, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"{atlas} — {cond}\n(N={len(mats)}, {metric})", fontsize=9)
                ax.tick_params(axis="both", which="both", labelsize=6)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{atlas} — {cond}")

    fig.suptitle(f"Average FC ({metric.upper()}, α band)", fontsize=12, y=1.01)
    fig.tight_layout()
    # Shared colorbar
    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, cax=cbar_ax, label=metric.upper())
    fig.savefig(out, **_SAVEKW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — PCA 2-D scatter
# ---------------------------------------------------------------------------

def fig_X_pca2(features_dir: Path, out: Path, *, force: bool = False) -> None:
    npz_path = features_dir / "X_schaefer100_coh_alpha.npz"
    y_path = features_dir / "y.npy"
    if not _needs_update(out, npz_path, y_path, force=force):
        return
    if not npz_path.exists():
        warnings.warn(f"PCA: {npz_path} not found — skipping", stacklevel=2)
        return

    from sklearn.decomposition import PCA  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    X = np.load(npz_path)["X"]
    y = np.load(y_path) if y_path.exists() else np.zeros(X.shape[0], dtype=int)
    groups = np.repeat(np.arange(X.shape[0] // 2), 2)

    X_sc = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_sc)
    var_explained = pca.explained_variance_ratio_ * 100

    colors = {0: "#1f77b4", 1: "#d62728"}
    labels = {0: "EO (y=0)", 1: "EC (y=1)"}
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "<",
               ">", "p", "H", "+", "x"]

    fig, ax = plt.subplots(figsize=_FIGSIZE_SQ)
    n_groups = len(np.unique(groups))
    for gi in range(n_groups):
        mask = groups == gi
        for cls in [0, 1]:
            idx = mask & (y == cls)
            if np.any(idx):
                ax.scatter(X2[idx, 0], X2[idx, 1],
                           c=colors[cls], marker=markers[gi % len(markers)],
                           s=60, alpha=0.8,
                           label=f"sub-{gi:03d} {labels[cls]}" if gi == 0 else None)

    # Simple legend for class
    for cls in [0, 1]:
        ax.scatter([], [], c=colors[cls], marker="o", s=60, label=labels[cls])

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}%)")
    ax.set_title("PCA 2D — X_schaefer100_coh_alpha (N=15 subjects)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, **_SAVEKW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — Summary table (top-3)
# ---------------------------------------------------------------------------

def fig_summary_table(cd: ComparisonData, out: Path, *, force: bool = False) -> None:
    if not _needs_update(out, Path(cd.source_file), force=force):
        return

    sorted_res = sorted(cd.results, key=lambda r: r.bal_acc, reverse=True)[:3]
    col_labels = ["Atlas", "Metric", "Classifier", "BA", "CI 95%", "p_perm"]
    rows = []
    for r in sorted_res:
        ci_str = (f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
                  if not np.isnan(r.ci_lo) else "—")
        p_str = f"{r.p_perm:.4f}" if not np.isnan(r.p_perm) else "—"
        rows.append([r.atlas, r.metric, r.classifier, f"{r.bal_acc:.3f}", ci_str, p_str])

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    # Header style
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Alternate row colors
    for i in range(1, len(rows) + 1):
        bg = "#eaf2fb" if i % 2 == 0 else "#ffffff"
        for j in range(len(col_labels)):
            tbl[(i, j)].set_facecolor(bg)
    # Winner row
    if rows:
        for j in range(len(col_labels)):
            tbl[(1, j)].set_facecolor("#d5f5e3")

    ax.set_title(f"Top-3 configurations — {cd.schema.upper()} N={cd.n_subjects}",
                 pad=12, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, **_SAVEKW)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    comparison: Path,
    features_dir: Path,
    fc_dir: Path,
    out_dir: Path,
    *,
    force: bool = False,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cd = load_comparison(comparison)

    if not cd.results:
        print(f"[WARN] No results in {comparison} — figures will be partial")

    figs = {
        "fig_balacc_bar.png": lambda p: fig_balacc_bar(cd, p, force=force),
        "fig_balacc_heatmap.png": lambda p: fig_balacc_heatmap(cd, p, force=force),
        "fig_pperm_bar.png": lambda p: fig_pperm_bar(cd, p, force=force),
        "fig_fc_avg.png": lambda p: fig_fc_avg(fc_dir, p, force=force),
        "fig_X_pca2.png": lambda p: fig_X_pca2(features_dir, p, force=force),
        "fig_summary_table.png": lambda p: fig_summary_table(cd, p, force=force),
    }

    produced = []
    skipped = []
    for fname, fn in figs.items():
        out_path = out_dir / fname
        existed = out_path.exists()
        fn(out_path)
        if out_path.exists():
            if existed and not force:
                skipped.append(fname)
            else:
                produced.append(fname)
                print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} (skipped — data unavailable)")

    print(f"\nProduced: {len(produced)} | Skipped (cached): {len(skipped)}")
    print(f"Output dir: {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="S-FIG-N15 — Generate classification result figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--comparison", required=True, type=Path,
                    help="comparison_matrix.json (PILOT or N=15)")
    ap.add_argument("--features-dir", default="data/features/ds005385", type=Path)
    ap.add_argument("--fc-dir", default="data/connectivity/ds005385", type=Path)
    ap.add_argument("--out-dir", default="reports/figures", type=Path)
    ap.add_argument("--force", action="store_true", help="Overwrite existing figures")
    args = ap.parse_args()
    return run(args.comparison, args.features_dir, args.fc_dir, args.out_dir, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
