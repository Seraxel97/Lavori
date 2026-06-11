"""Publication-quality figure export via matplotlib."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_DPI = 300
_FONT_SIZE = 10
_FMT = Literal["png", "svg", "pdf"]


def _apply_pub_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": _FONT_SIZE,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": _DPI,
    })


def export_confusion_matrix(
    cm: list[list[int]],
    path: Path | str,
    fmt: _FMT = "png",
    labels: list[str] | None = None,
) -> Path:
    _apply_pub_style()
    labels = labels or ["EO", "EC"]
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(cm_arr, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (LOSO aggregate)", fontsize=_FONT_SIZE)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    total = cm_arr.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_arr[i,j]}\n({cm_arr[i,j]/total*100:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_arr[i, j] > total * 0.4 else "black")
    fig.tight_layout()
    out = Path(str(path) if not str(path).endswith(f".{fmt}") else path)
    fig.savefig(out, dpi=_DPI, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return out


def export_roc_curve(
    fpr: list[float],
    tpr: list[float],
    auc_val: float,
    path: Path | str,
    fmt: _FMT = "png",
) -> Path:
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(4.0, 3.5))
    ax.plot(fpr, tpr, color="#2166ac", lw=2, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Chance")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2166ac")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (LOSO aggregate)", fontsize=_FONT_SIZE)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = Path(str(path))
    fig.savefig(out, dpi=_DPI, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return out


def export_permutation_null(
    null_distribution: list[float],
    observed_ba: float,
    path: Path | str,
    fmt: _FMT = "png",
    p_perm: float | None = None,
) -> Path:
    from analysis.stats_utility import format_pvalue

    _apply_pub_style()
    null = np.array(null_distribution)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.hist(null, bins=40, color="#4575b4", alpha=0.75, edgecolor="white", lw=0.3,
            label=f"Null (n={len(null)})")
    _p_label = format_pvalue(p_perm, len(null)) if p_perm is not None else ""
    ax.axvline(observed_ba, color="#d73027", lw=2.0,
               label=f"Observed BA={observed_ba:.3f}" + (f"\n{_p_label}" if _p_label else ""))
    ax.axvline(null.mean(), color="gray", lw=1.2, linestyle="--", label=f"Null mean={null.mean():.3f}")
    ax.set_xlabel("Balanced Accuracy (permuted labels)")
    ax.set_ylabel("Count")
    ax.set_title("Permutation Null Distribution", fontsize=_FONT_SIZE)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(str(path))
    fig.savefig(out, dpi=_DPI, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return out


def export_learning_curve(
    ba_per_fold: list[float],
    path: Path | str,
    fmt: _FMT = "png",
) -> Path:
    _apply_pub_style()
    x = list(range(1, len(ba_per_fold) + 1))
    mean_ba = float(np.mean(ba_per_fold))
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.plot(x, ba_per_fold, "o-", color="#4dac26", ms=3, lw=1.2, label="BA per fold")
    ax.axhline(mean_ba, color="#d73027", lw=1.5, ls="--", label=f"Mean={mean_ba:.3f}")
    ax.axhline(0.5, color="gray", lw=1.0, ls=":", label="Chance")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fold (subject withheld)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("BA per LOSO fold", fontsize=_FONT_SIZE)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    out = Path(str(path))
    fig.savefig(out, dpi=_DPI, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return out


def fig_to_bytes(fig: plt.Figure, fmt: _FMT = "png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=_DPI, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def export_all(
    result: dict,
    out_dir: Path,
    fmt: _FMT = "png",
) -> list[Path]:
    """Export all available figures for a result dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{result.get('atlas','?')}_{result.get('metric','?')}_{result.get('band','?')}_{result.get('clf','?')}"
    exported = []
    if result.get("confusion"):
        p = export_confusion_matrix(result["confusion"], out_dir / f"{stem}_cm.{fmt}", fmt=fmt)
        exported.append(p)
    if result.get("roc_fpr") and result.get("roc_auc") is not None:
        p = export_roc_curve(result["roc_fpr"], result["roc_tpr"], result["roc_auc"],
                              out_dir / f"{stem}_roc.{fmt}", fmt=fmt)
        exported.append(p)
    if result.get("null_distribution"):
        p = export_permutation_null(result["null_distribution"], result["ba"],
                                     out_dir / f"{stem}_perm.{fmt}", fmt=fmt,
                                     p_perm=result.get("p_perm"))
        exported.append(p)
    if result.get("ba_per_fold"):
        p = export_learning_curve(result["ba_per_fold"], out_dir / f"{stem}_lc.{fmt}", fmt=fmt)
        exported.append(p)
    return exported
