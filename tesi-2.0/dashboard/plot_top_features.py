"""
STEP 7b — Brain visualization figures (headless, matplotlib Agg backend).

Genera PNG per top ROI e top FC edges da risultati benchmark.
Backend Agg forzato: nessun display richiesto, compatibile CI/server headless.

Usage:
    python dashboard/plot_top_features.py \\
        --bench-results reports/BENCH_MATRIX_RESULTS.json \\
        --atlas aparc \\
        --output reports/figures/
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Force Agg backend before matplotlib.pyplot import (headless/CI compatibility)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402, I001
import numpy as np  # noqa: I001


_DEFAULT_DPI = 300
_DEFAULT_FIGSIZE_ROI = (10, 5)
_DEFAULT_FIGSIZE_EDGE = (8, 7)


def plot_top_roi(
    scores: dict[str, float],
    atlas: str,
    *,
    n_top: int = 10,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Barchart orizzontale dei top ROI per importanza (BA contribution).

    Parameters
    ----------
    scores:
        Dict {roi_name: importance_value}. Valori positivi = discriminativi.
    atlas:
        Nome atlante (usato nell'etichetta del grafico).
    n_top:
        Numero di top ROI da visualizzare.
    output_path:
        Path file PNG di output.
    title:
        Titolo custom. Se None usa default.

    Returns
    -------
    Path
        Path del file PNG salvato.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_items = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
    top_items = sorted_items[:n_top]
    names = [item[0] for item in top_items]
    values = [item[1] for item in top_items]

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE_ROI)
    colors = ["#d73027" if v >= 0 else "#4575b4" for v in values]
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Importanza (Balanced Accuracy contribution)", fontsize=10)
    ax.set_title(title or f"Top {n_top} ROI discriminativi — atlas: {atlas}", fontsize=11)
    ax.invert_yaxis()

    # Annotazione valori
    for bar, val in zip(bars, values):
        ax.text(
            val + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=7,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_top_edges(
    fc_importance: np.ndarray,
    names: list[str],
    *,
    n_top: int = 20,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Heatmap NxN con top edges FC evidenziati.

    Parameters
    ----------
    fc_importance:
        Array shape (N, N) con importanza per ogni coppia ROI.
        Simmetrico; diagonale ignorata.
    names:
        Lista di N nomi ROI (etichette assi).
    n_top:
        Numero di top edges da evidenziare con stella.
    output_path:
        Path file PNG di output.
    title:
        Titolo custom.

    Returns
    -------
    Path
        Path del file PNG salvato.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(names)
    mat = np.array(fc_importance, dtype=float)
    if mat.shape != (n, n):
        raise ValueError(f"fc_importance shape {mat.shape} != ({n},{n})")

    # Trova top edges (upper triangle)
    triu_idx = np.triu_indices(n, k=1)
    triu_vals = mat[triu_idx]
    top_k = min(n_top, len(triu_vals))
    top_flat_idx = np.argpartition(np.abs(triu_vals), -top_k)[-top_k:]
    top_mask = np.zeros((n, n), dtype=bool)
    for idx in top_flat_idx:
        i, j = triu_idx[0][idx], triu_idx[1][idx]
        top_mask[i, j] = True
        top_mask[j, i] = True

    fig, ax = plt.subplots(figsize=_DEFAULT_FIGSIZE_EDGE)
    vmax = np.nanpercentile(np.abs(mat), 95) or 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="Importanza FC", shrink=0.8)

    # Overlay stelle per top edges
    top_rows, top_cols = np.where(top_mask & np.triu(np.ones((n, n), dtype=bool), k=1))
    ax.scatter(top_cols, top_rows, marker="*", color="gold", s=30, zorder=5, label=f"Top {top_k} edges")

    # Etichette assi (ridotte se N > 20)
    if n <= 20:
        ax.set_xticks(range(n))
        ax.set_xticklabels(names, rotation=90, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=7)
    else:
        step = max(1, n // 10)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels([names[i] for i in range(0, n, step)], rotation=90, fontsize=7)
        ax.set_yticks(range(0, n, step))
        ax.set_yticklabels([names[i] for i in range(0, n, step)], fontsize=7)

    ax.set_title(title or f"FC edge importance — top {top_k} evidenziati", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=_DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_bench_results(json_path: Path) -> tuple[dict[str, float], np.ndarray, list[str]]:
    """Carica BENCH_MATRIX_RESULTS.json e restituisce scores ROI + FC importance.

    Se il file e' un placeholder (n_runs=0), genera dati sintetici per smoke test.

    Returns
    -------
    roi_scores : dict {roi_name: float}
    fc_importance : np.ndarray shape (N, N)
    roi_names : list[str]
    """
    data = json.loads(json_path.read_text())
    runs = data.get("runs", [])

    if not runs:
        # Placeholder: genera dati sintetici aparc-like (68 ROI)
        from parcellation.extract_label_tc import get_labels
        try:
            labels = get_labels("aparc")
            roi_names = [lbl.name for lbl in labels]
        except Exception:
            roi_names = [f"roi-{i:02d}" for i in range(20)]
        n = len(roi_names)
        rng = np.random.default_rng(42)
        roi_scores = {name: float(rng.standard_normal()) * 0.05 for name in roi_names}
        fc_importance = rng.standard_normal((n, n)) * 0.03
        fc_importance = (fc_importance + fc_importance.T) / 2
        np.fill_diagonal(fc_importance, 0)
        return roi_scores, fc_importance, roi_names

    # Aggrega runs reali: media BA per ROI (stub — dipende da struttura reale)
    roi_scores: dict[str, float] = {}
    all_names: list[str] = []
    for run in runs:
        names = run.get("roi_names", [])
        scores = run.get("roi_importance", [])
        for name, score in zip(names, scores):
            roi_scores[name] = roi_scores.get(name, 0) + score / len(runs)
            if name not in all_names:
                all_names.append(name)

    n = len(all_names)
    fc_importance = np.zeros((n, n))
    for run in runs:
        fi = run.get("fc_importance")
        if fi is not None:
            fc_importance += np.array(fi) / len(runs)

    return roi_scores, fc_importance, all_names


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate brain visualization figures (headless)")
    ap.add_argument("--bench-results", default="reports/BENCH_MATRIX_RESULTS.json",
                    help="Path BENCH_MATRIX_RESULTS.json")
    ap.add_argument("--atlas", default="aparc", choices=["aparc", "destrieux", "schaefer100", "schaefer200"])
    ap.add_argument("--n-top-roi", type=int, default=10)
    ap.add_argument("--n-top-edges", type=int, default=20)
    ap.add_argument("--output", default="reports/figures/", help="Output directory per PNG")
    args = ap.parse_args()

    bench_path = Path(args.bench_results)
    out_dir = Path(args.output)

    roi_scores, fc_importance, roi_names = _load_bench_results(bench_path)

    roi_png = plot_top_roi(
        roi_scores, args.atlas,
        n_top=args.n_top_roi,
        output_path=out_dir / f"top_roi_{args.atlas}.png",
    )
    print(f"Saved ROI figure: {roi_png}")

    edge_png = plot_top_edges(
        fc_importance, roi_names,
        n_top=args.n_top_edges,
        output_path=out_dir / f"top_edges_{args.atlas}.png",
    )
    print(f"Saved edge figure: {edge_png}")
