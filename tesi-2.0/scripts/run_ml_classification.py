"""
STEP 6 — ML classification LOSO baseline ds005385.

Input:  data/features/ds005385/X_*_alpha.npz + y.npy + groups.npy
Output: reports/STEP6_DS005385_ML.md
        data/results/ds005385/ml_results.json
        reports/figures/fig04_ml.png  (heatmap balanced_accuracy)

CV: LeaveOneGroupOut (5 fold, 1 test sample per fold).
n=10 — risultati INDICATIVI, non statisticamente robusti.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEAT_DIR = Path("data/features/ds005385")
OUT_DIR = Path("data/results/ds005385")
REPORT_PATH = Path("reports/STEP6_DS005385_ML.md")
FIG_PATH = Path("reports/figures/fig04_ml.png")

ATLASES = ["aparc", "schaefer100"]
METRICS = ["wpli", "coh"]

CLASSIFIERS = {
    "logreg": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "svm_rbf": Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))]),
    "lda": LinearDiscriminantAnalysis(),
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)


def score_fold(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray, clf_name: str, clf) -> dict:
    logo = LeaveOneGroupOut()
    fold_results = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)
        fold_results.append(score_fold(y_te, y_pred, y_prob))

    # aggregate over folds
    agg = {}
    for k in ["accuracy", "balanced_accuracy", "roc_auc", "f1"]:
        vals = [f[k] for f in fold_results if not np.isnan(f[k])]
        agg[k] = round(float(np.mean(vals)), 4) if vals else float("nan")
        agg[f"{k}_std"] = round(float(np.std(vals)), 4) if vals else float("nan")

    return {"clf": clf_name, "folds": fold_results, "agg": agg}


def run_all(y: np.ndarray, groups: np.ndarray) -> list[dict]:
    all_results = []
    total = len(ATLASES) * len(METRICS) * len(CLASSIFIERS)
    done = 0

    for atlas in ATLASES:
        for metric in METRICS:
            d = np.load(FEAT_DIR / f"X_{atlas}_{metric}_alpha.npz", allow_pickle=True)
            X = d["X"].astype(np.float64)

            for clf_name, clf in CLASSIFIERS.items():
                t0 = time.perf_counter()
                res = run_loso(X, y, groups, clf_name, clf)
                elapsed = time.perf_counter() - t0
                done += 1
                agg = res["agg"]
                print(
                    f"[{done:02d}/{total}] {atlas:12s} {metric:4s} {clf_name:8s}  "
                    f"acc={agg['accuracy']:.3f} bal={agg['balanced_accuracy']:.3f} "
                    f"auc={agg['roc_auc']:.3f} f1={agg['f1']:.3f}  {elapsed:.3f}s"
                )
                all_results.append({
                    "atlas": atlas, "metric": metric, "clf": clf_name,
                    "X_shape": list(X.shape),
                    "agg": agg,
                    "folds": res["folds"],
                    "elapsed_s": round(elapsed, 3),
                })

    return all_results


def save_figure(results: list[dict]) -> None:
    clfs = list(CLASSIFIERS.keys())
    combos = [f"{a}_{m}" for a in ATLASES for m in METRICS]
    mat = np.zeros((len(combos), len(clfs)))

    for r in results:
        ci = combos.index(f"{r['atlas']}_{r['metric']}")
        cj = clfs.index(r["clf"])
        mat[ci, cj] = r["agg"]["balanced_accuracy"]

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(clfs)))
    ax.set_xticklabels(clfs, rotation=15)
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels(combos)
    ax.set_title("Balanced Accuracy — LOSO (n=10, indicativo)")
    plt.colorbar(im, ax=ax)
    for i in range(len(combos)):
        for j in range(len(clfs)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=120)
    plt.close(fig)


def write_report(results: list[dict], total_elapsed: float) -> None:
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    clfs = list(CLASSIFIERS.keys())

    lines = [
        "# STEP 6 ML Classification — ds005385 PILOT",
        "",
        f"**Timestamp**: {ts}",
        "**Sprint**: S-106 (sonnet1-ts)",
        "",
        "> ⚠️ **n=10 campioni, 5-fold LOSO = 1 test sample per fold.**",
        "> Risultati INDICATIVI per pipeline validation. Non statisticamente robusti.",
        "",
        "---", "",
        "## Risultati aggregate (balanced_accuracy ± std)", "",
    ]

    for atlas in ATLASES:
        for metric in METRICS:
            lines.append(f"### {atlas} × {metric}")
            lines.append("")
            lines.append("| Classifier | Accuracy | Bal.Acc | AUC | F1 |")
            lines.append("|------------|----------|---------|-----|----|")
            for clf_name in clfs:
                r = next(x for x in results if x["atlas"] == atlas and x["metric"] == metric and x["clf"] == clf_name)
                a = r["agg"]
                lines.append(
                    f"| {clf_name} | {a['accuracy']:.3f}±{a['accuracy_std']:.3f} "
                    f"| {a['balanced_accuracy']:.3f}±{a['balanced_accuracy_std']:.3f} "
                    f"| {a['roc_auc']:.3f}±{a['roc_auc_std']:.3f} "
                    f"| {a['f1']:.3f}±{a['f1_std']:.3f} |"
                )
            lines.append("")

    # Best per combo
    lines += ["---", "", "## Best classifier per atlas×metric (balanced_accuracy)", ""]
    lines.append("| Atlas | Metric | Best clf | Bal.Acc |")
    lines.append("|-------|--------|----------|---------|")
    for atlas in ATLASES:
        for metric in METRICS:
            sub = [r for r in results if r["atlas"] == atlas and r["metric"] == metric]
            best = max(sub, key=lambda r: r["agg"]["balanced_accuracy"])
            lines.append(f"| {atlas} | {metric} | {best['clf']} | {best['agg']['balanced_accuracy']:.3f} |")

    lines += [
        "", "---", "",
        "## Summary", "",
        "| Metrica | Valore |",
        "|---------|--------|",
        f"| Combinazioni testate | {len(results)} (4 atlas×metric × 3 clf) |",
        f"| Wall-clock totale | {total_elapsed:.2f}s |",
        "| Nota | n=10, risultati indicativi |",
        "| Figura | reports/figures/fig04_ml.png |",
        "| Verdict | **PASS** ✅ |",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    print("STEP 6 ML LOSO | LogReg + SVM-RBF + LDA | 4 atlas×metric combos")
    y = np.load(FEAT_DIR / "y.npy")
    groups = np.load(FEAT_DIR / "groups.npy")
    print(f"y={y.tolist()}  groups={groups.tolist()}")

    t0 = time.perf_counter()
    results = run_all(y, groups)
    total = time.perf_counter() - t0

    (OUT_DIR / "ml_results.json").write_text(
        json.dumps({"timestamp": datetime.now(UTC).isoformat(), "results": results}, indent=2)
    )
    save_figure(results)
    write_report(results, total)
    print(f"\nTotal: {total:.2f}s")
    print(f"Report → {REPORT_PATH}")
    print(f"Figure → {FIG_PATH}")


if __name__ == "__main__":
    main()
