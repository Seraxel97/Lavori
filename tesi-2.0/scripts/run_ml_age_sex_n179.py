"""FASE 3 — ML età + sesso su N=179 (ds005385).

Eseguire DOPO run_pipeline_n30.py --n-subjects 179 --atlases aparc
che produce X_aparc_*.npz (shape 358×2278).

Usage:
    python scripts/run_ml_age_sex_n179.py [--n-perm 1000] [--n-boot 1000]
    python scripts/run_ml_age_sex_n179.py --dry-run   # solo check input

Protocol:
    Winner FASE 1: X_aparc_plv_theta (2278-dim)
    Sesso: LogReg-L2, GroupKFold K=5 outer K=3 inner, balanced_accuracy
    Età:   RF,        GroupKFold K=5 outer K=3 inner, MAE + R²
    GT:    8-dim scalari da FC 68×68 ricostruita da X_aparc_plv_theta
    Perm:  subject-level (shuffle soggetti, broadcast via groups)
    Boot:  cluster subject-level (n_boot, seed=42)
    FDR-BH su 4 test simultanei
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.fdr_helper import apply_fdr
from config.labels_phenotype import load_phenotype
from config.subjects_whitelist_n179 import SUBJECT_WHITELIST_N179 as SUBJECTS_N179
from features.graph_theory import compute_graph_metrics

FEAT_DIR = Path("data/features/ds005385")
RESULTS_DIR = Path("data/results/ds005385")
REPORT_DIR = Path("reports")
N_ROI = 68  # aparc


# ── Helpers ────────────────────────────────────────────────────────────────────

def _update_metadata_n179(subjects: list[str]) -> None:
    """Aggiorna metadata.json con lista N=179 per ml_sex/ml_age compatibility."""
    p = FEAT_DIR / "metadata.json"
    meta = json.loads(p.read_text())
    meta["subjects"] = subjects
    meta["n_subjects"] = len(subjects)
    meta["n_samples"] = len(subjects) * 2
    meta["timestamp_n179"] = datetime.now(UTC).isoformat(timespec="seconds")
    p.write_text(json.dumps(meta, indent=2))
    print(f"  metadata.json aggiornato: N={len(subjects)}")


def _load_data(feat_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carica X, y_age, y_sex, groups dalla whitelist N=179."""
    X_file = FEAT_DIR / f"{feat_name}.npz"
    if not X_file.exists():
        raise FileNotFoundError(f"Feature file mancante: {X_file}\n"
                                "Eseguire prima: run_pipeline_n30.py --n-subjects 179 --atlases aparc")
    arr = np.load(X_file)
    X = arr["X"].astype(np.float64)

    n_expected = len(SUBJECTS_N179) * 2
    if X.shape[0] != n_expected:
        raise ValueError(f"X.shape[0]={X.shape[0]} atteso {n_expected} "
                         f"(N={len(SUBJECTS_N179)}×2 cond). "
                         "Pipeline N=179 non ancora completata?")

    groups = np.repeat(np.arange(len(SUBJECTS_N179)), 2)
    y_age_per_sub, y_sex_per_sub = load_phenotype(SUBJECTS_N179)
    y_age = y_age_per_sub[groups]
    y_sex = y_sex_per_sub[groups]
    return X, y_age, y_sex, groups, y_age_per_sub


def _permute_subject_labels(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    unique_groups = np.unique(groups)
    y_per_sub = np.array([y[groups == g][0] for g in unique_groups])
    y_perm_per_sub = rng.permutation(y_per_sub)
    y_perm = np.empty_like(y)
    for i, g in enumerate(unique_groups):
        y_perm[groups == g] = y_perm_per_sub[i]
    return y_perm


def _cluster_bootstrap_mae_r2(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, n_boot: int, seed: int = 42
) -> tuple[list[float], list[float]]:
    unique_groups = np.unique(groups)
    n_sub = len(unique_groups)
    rng = np.random.default_rng(seed)
    boot_mae, boot_r2 = [], []
    for _ in range(n_boot):
        sel = rng.choice(unique_groups, n_sub, replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sel])
        boot_mae.append(mean_absolute_error(y_true[idx], y_pred[idx]))
        boot_r2.append(r2_score(y_true[idx], y_pred[idx]))
    return boot_mae, boot_r2


def _cluster_bootstrap_ba(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, n_boot: int, seed: int = 42
) -> list[float]:
    unique_groups = np.unique(groups)
    n_sub = len(unique_groups)
    rng = np.random.default_rng(seed)
    boot_ba = []
    for _ in range(n_boot):
        sel = rng.choice(unique_groups, n_sub, replace=True)
        idx = np.concatenate([np.where(groups == g)[0] for g in sel])
        boot_ba.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    return boot_ba


# ── ML Sesso ───────────────────────────────────────────────────────────────────

def run_cv_sex(X, y, groups, outer_k=5, inner_k=3):
    outer_cv = GroupKFold(n_splits=outer_k)
    ba_folds, y_pred_all, y_true_all = [], np.empty(len(y)), np.empty(len(y))
    for train_idx, test_idx in outer_cv.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_tr = groups[train_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        inner_cv = GroupKFold(n_splits=min(inner_k, len(np.unique(g_tr))))
        clf = GridSearchCV(
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               solver="liblinear", random_state=42),
            param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
            cv=inner_cv, scoring="balanced_accuracy", refit=True,
        )
        clf.fit(X_tr_s, y_tr, groups=g_tr)
        y_pred = clf.predict(X_te_s)
        ba_folds.append(balanced_accuracy_score(y_te, y_pred))
        y_pred_all[test_idx] = y_pred
        y_true_all[test_idx] = y_te
    return {
        "ba_folds": [float(b) for b in ba_folds],
        "ba_mean": float(np.mean(ba_folds)),
        "ba_std": float(np.std(ba_folds)),
        "y_pred": y_pred_all,
        "y_true": y_true_all,
    }


def run_ml_sex(X, y, groups, n_perm, n_boot):
    t0 = time.perf_counter()
    cv_res = run_cv_sex(X, y, groups)
    ba_real = cv_res["ba_mean"]

    rng = np.random.default_rng(42)
    null_bas = []
    for _ in range(n_perm):
        y_shuf = _permute_subject_labels(y, groups, rng)
        null_bas.append(run_cv_sex(X, y_shuf, groups)["ba_mean"])
    p_perm = (sum(1 for b in null_bas if b >= ba_real) + 1) / (n_perm + 1)

    boot_ba = _cluster_bootstrap_ba(cv_res["y_true"], cv_res["y_pred"], groups, n_boot)
    elapsed = time.perf_counter() - t0
    return {
        "ba_mean": ba_real,
        "ba_std": cv_res["ba_std"],
        "ba_ci95": [float(np.percentile(boot_ba, 2.5)), float(np.percentile(boot_ba, 97.5))],
        "p_perm": float(p_perm),
        "null_ba_mean": float(np.mean(null_bas)),
        "n_perm": n_perm,
        "n_boot": n_boot,
        "elapsed_s": round(elapsed),
    }


# ── ML Età ─────────────────────────────────────────────────────────────────────

def run_cv_age(X, y, groups, outer_k=5, inner_k=3):
    outer_cv = GroupKFold(n_splits=outer_k)
    y_pred_all, y_true_all = np.empty(len(y)), np.empty(len(y))
    for train_idx, test_idx in outer_cv.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        reg.fit(X_tr_s, y_tr)
        y_pred_all[test_idx] = reg.predict(X_te_s)
        y_true_all[test_idx] = y_te
    mae = float(mean_absolute_error(y_true_all, y_pred_all))
    r2 = float(r2_score(y_true_all, y_pred_all))
    return {"mae": mae, "r2": r2, "y_pred": y_pred_all, "y_true": y_true_all,
            "brain_age_gap": float(np.mean(y_pred_all - y_true_all))}


def run_ml_age(X, y, groups, n_perm, n_boot):
    t0 = time.perf_counter()
    cv_res = run_cv_age(X, y, groups)
    r2_real = cv_res["r2"]

    rng = np.random.default_rng(42)
    null_r2 = []
    for _ in range(n_perm):
        y_shuf = _permute_subject_labels(y, groups, rng)
        null_r2.append(run_cv_age(X, y_shuf, groups)["r2"])
    p_perm = (sum(1 for r in null_r2 if r >= r2_real) + 1) / (n_perm + 1)

    boot_mae, boot_r2 = _cluster_bootstrap_mae_r2(
        cv_res["y_true"], cv_res["y_pred"], groups, n_boot)
    elapsed = time.perf_counter() - t0
    return {
        "mae": cv_res["mae"],
        "r2": r2_real,
        "brain_age_gap": cv_res["brain_age_gap"],
        "mae_ci95": [float(np.percentile(boot_mae, 2.5)), float(np.percentile(boot_mae, 97.5))],
        "r2_ci95": [float(np.percentile(boot_r2, 2.5)), float(np.percentile(boot_r2, 97.5))],
        "p_perm": float(p_perm),
        "null_r2_mean": float(np.mean(null_r2)),
        "n_perm": n_perm,
        "n_boot": n_boot,
        "elapsed_s": round(elapsed),
    }


# ── GT features da X_aparc_plv_theta ──────────────────────────────────────────

def build_gt_features(X: np.ndarray, n_roi: int = N_ROI) -> np.ndarray:
    """Ricava X_gt (n_samples, 8) da X_fc (n_samples, n_roi*(n_roi-1)/2).

    Ricostruisce matrice n_roi×n_roi simmetrica per ogni sample,
    calcola 8 scalari GT: mean_degree, mean_strength, clustering_coeff,
    path_length, global_efficiency, local_efficiency, modularity_q, small_worldness.
    """
    n_samples = X.shape[0]
    idx_triu = np.triu_indices(n_roi, k=1)
    gt_rows = []
    for i in range(n_samples):
        mat = np.zeros((n_roi, n_roi))
        mat[idx_triu] = X[i]
        mat = mat + mat.T
        metrics = compute_graph_metrics(mat, threshold=0.20)
        gt_rows.append([
            metrics["mean_degree"], metrics["mean_strength"],
            metrics["clustering_coeff"], metrics["path_length"],
            metrics["global_efficiency"], metrics["local_efficiency"],
            metrics["modularity_q"], metrics["small_worldness"],
        ])
    X_gt = np.array(gt_rows, dtype=np.float64)
    X_gt = np.nan_to_num(X_gt, nan=0.0)
    return X_gt


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(results: dict, path: Path) -> None:
    ns = results["n_subjects"]
    ts = results["timestamp"]
    sx = results["sex_baseline"]
    ag = results["age_baseline"]
    sx_gt = results["sex_gt"]
    ag_gt = results["age_gt"]
    fdr = results["fdr"]

    lines = [
        "# FASE 3 — Risultati ML Età + Sesso N=179",
        "",
        f"**Data**: {ts}",
        f"**Coorte**: ds005385 N={ns} (whitelist determinististica seed=43)",
        "**Feature winner**: X_aparc_plv_theta (2278-dim PLV theta aparc)",
        f"**Protocollo**: GroupKFold K=5 outer / K=3 inner, perm subject-level n={sx['n_perm']}, "
        f"boot cluster-subject n={sx['n_boot']}",
        "",
        "## 1. Classificazione Sesso — baseline 2278-dim",
        "",
        "| Metrica | Valore | CI95 | p_perm |",
        "|---------|--------|------|--------|",
        f"| BA | **{sx['ba_mean']:.3f}** | [{sx['ba_ci95'][0]:.3f}, {sx['ba_ci95'][1]:.3f}] "
        f"| {sx['p_perm']:.4f} |",
        f"| Null BA (mean) | {sx['null_ba_mean']:.3f} | — | — |",
        "",
        f"Significativo (α=0.05): **{'SÌ' if sx['p_perm'] < 0.05 else 'NO'}**",
        "Letteratura (Kollia 2022, N=100): BA≈0.70-0.75. N=100 FASE 1: BA=0.713.",
        "",
        "## 2. Regressione Età — baseline 2278-dim",
        "",
        "| Metrica | Valore | CI95 |",
        "|---------|--------|------|",
        f"| MAE (anni) | **{ag['mae']:.2f}** | [{ag['mae_ci95'][0]:.2f}, {ag['mae_ci95'][1]:.2f}] |",
        f"| R² | **{ag['r2']:.3f}** | [{ag['r2_ci95'][0]:.3f}, {ag['r2_ci95'][1]:.3f}] |",
        f"| Brain-age gap (mean) | {ag['brain_age_gap']:.2f} anni | — |",
        f"| p_perm (R²) | {ag['p_perm']:.4f} | — |",
        "",
        f"Significativo (α=0.05): **{'SÌ' if ag['p_perm'] < 0.05 else 'NO'}**",
        "Letteratura (Franck 2019, N>200): MAE≈8-12 anni. N=100 FASE 1: MAE=12.52.",
        "",
        "## 3. Graph-Theory 8-dim vs Baseline — FDR-BH",
        "",
        "| Confronto | Score | p_raw | p_fdr | Sig? |",
        "|-----------|-------|-------|-------|------|",
    ]

    labels = ["sex_baseline", "sex_gt", "age_baseline", "age_gt"]
    for lbl in labels:
        p_raw = fdr["p_raw"][lbl]
        p_fdr_ = fdr["p_fdr"][lbl]
        sig = "✅" if fdr["reject"][lbl] else "❌"
        if lbl == "sex_baseline":
            score = f"BA={sx['ba_mean']:.3f}"
        elif lbl == "sex_gt":
            score = f"BA={sx_gt['ba_mean']:.3f}"
        elif lbl == "age_baseline":
            score = f"R²={ag['r2']:.3f}"
        else:
            score = f"R²={ag_gt['r2']:.3f}"
        lines.append(f"| {lbl} | {score} | {p_raw:.4f} | {p_fdr_:.4f} | {sig} |")

    lines += [
        "",
        "## 4. Confronto N=100 vs N=179",
        "",
        "| Target | Metrica | N=100 FASE 1 | N=179 FASE 3 | Δ |",
        "|--------|---------|-------------|-------------|---|",
        f"| Sesso | BA | 0.713 | {sx['ba_mean']:.3f} | {sx['ba_mean']-0.713:+.3f} |",
        f"| Età | MAE (anni) | 12.52 | {ag['mae']:.2f} | {ag['mae']-12.52:+.2f} |",
        f"| Età | R² | 0.097 | {ag['r2']:.3f} | {ag['r2']-0.097:+.3f} |",
        "",
        "## 5. Limitazioni",
        "",
        "- Sorgenti su fsaverage (no MRI individuale): distorsione spaziale per-soggetto.",
        "- PLV sensibile a volume conduction residuo (mitigato da parcellazione aparc).",
        "- schaefer100 non utilizzato (incompleto su N=100, non ripetuto su N=179).",
        f"- n_perm={sx['n_perm']}: p_perm ≥ 1/(n_perm+1) = {1/(sx['n_perm']+1):.5f}.",
        "",
        "---",
        f"*Generato da scripts/run_ml_age_sex_n179.py — {ts}*",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"  Report → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--feat-name", default="X_aparc_plv_theta")
    ap.add_argument("--dry-run", action="store_true", help="Solo verifica input, non esegue ML")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    print(f"FASE 3 ML età+sesso — N={len(SUBJECTS_N179)}, "
          f"feat={args.feat_name}, n_perm={args.n_perm}, n_boot={args.n_boot}")

    if args.dry_run:
        # Solo verifica esistenza feature file, non modifica metadata.json
        X_file = FEAT_DIR / f"{args.feat_name}.npz"
        if not X_file.exists():
            print(f"DRY RUN ERROR: {X_file} mancante")
            sys.exit(1)
        arr = np.load(X_file)
        X_dry = arr["X"]
        n_expected = len(SUBJECTS_N179) * 2
        if X_dry.shape[0] != n_expected:
            print(f"DRY RUN WARN: X.shape[0]={X_dry.shape[0]}, atteso {n_expected} "
                  "(pipeline N=179 non ancora completata)")
        else:
            print(f"DRY RUN OK — X.shape={X_dry.shape}, pipeline N=179 completata")
        return

    # Aggiorna metadata.json (solo se non dry-run)
    _update_metadata_n179(list(SUBJECTS_N179))

    # Carica data
    print("Caricamento features…")
    X, y_age, y_sex, groups, y_age_per_sub = _load_data(args.feat_name)
    print(f"  X.shape={X.shape}, N_sub={len(SUBJECTS_N179)}, "
          f"y_sex F/M={sum(y_sex[::2]==0)}/{sum(y_sex[::2]==1)}, "
          f"y_age mean={y_age.mean():.1f}±{y_age.std():.1f}")

    # GT features
    print("\nCalcolo GT features (8-dim)…")
    t_gt = time.perf_counter()
    X_gt = build_gt_features(X, N_ROI)
    print(f"  X_gt.shape={X_gt.shape} in {time.perf_counter()-t_gt:.0f}s")
    np.save(FEAT_DIR / "X_aparc_plv_theta_gt.npy", X_gt)

    # ML sesso baseline
    print(f"\n[1/4] ML sesso baseline (n_perm={args.n_perm})…")
    sex_baseline = run_ml_sex(X, y_sex, groups, args.n_perm, args.n_boot)
    print(f"  BA={sex_baseline['ba_mean']:.3f} ± {sex_baseline['ba_std']:.3f} "
          f"p={sex_baseline['p_perm']:.4f} ({sex_baseline['elapsed_s']}s)")

    # ML età baseline
    print(f"\n[2/4] ML età baseline (n_perm={args.n_perm})…")
    age_baseline = run_ml_age(X, y_age, groups, args.n_perm, args.n_boot)
    print(f"  MAE={age_baseline['mae']:.2f} R²={age_baseline['r2']:.3f} "
          f"p={age_baseline['p_perm']:.4f} ({age_baseline['elapsed_s']}s)")

    # ML sesso GT
    print(f"\n[3/4] ML sesso GT 8-dim (n_perm={args.n_perm})…")
    sex_gt = run_ml_sex(X_gt, y_sex, groups, args.n_perm, args.n_boot)
    print(f"  BA_gt={sex_gt['ba_mean']:.3f} p={sex_gt['p_perm']:.4f}")

    # ML età GT
    print(f"\n[4/4] ML età GT 8-dim (n_perm={args.n_perm})…")
    age_gt = run_ml_age(X_gt, y_age, groups, args.n_perm, args.n_boot)
    print(f"  MAE_gt={age_gt['mae']:.2f} R²_gt={age_gt['r2']:.3f} p={age_gt['p_perm']:.4f}")

    # FDR-BH su 4 test
    p_raw = {
        "sex_baseline": sex_baseline["p_perm"],
        "sex_gt": sex_gt["p_perm"],
        "age_baseline": age_baseline["p_perm"],
        "age_gt": age_gt["p_perm"],
    }
    keys = list(p_raw.keys())
    p_corr, reject = apply_fdr([p_raw[k] for k in keys])
    fdr_results = {
        "p_raw": {k: p_raw[k] for k in keys},
        "p_fdr": {k: float(p_corr[i]) for i, k in enumerate(keys)},
        "reject": {k: bool(reject[i]) for i, k in enumerate(keys)},
        "alpha": 0.05,
        "method": "fdr_bh",
    }
    print("\nFDR-BH risultati:")
    for k in keys:
        print(f"  {k}: p_raw={p_raw[k]:.4f} p_fdr={fdr_results['p_fdr'][k]:.4f} "
              f"{'✅' if fdr_results['reject'][k] else '❌'}")

    # Assembla risultati
    results = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_subjects": len(SUBJECTS_N179),
        "n_samples": len(SUBJECTS_N179) * 2,
        "feat_name": args.feat_name,
        "protocol": {
            "outer_k": 5, "inner_k": 3, "n_perm": args.n_perm,
            "n_boot": args.n_boot, "perm_type": "subject_level",
            "boot_type": "cluster_subject_level",
        },
        "sex_baseline": sex_baseline,
        "age_baseline": age_baseline,
        "sex_gt": sex_gt,
        "age_gt": age_gt,
        "fdr": fdr_results,
        "comparison_n100": {
            "sex_ba_n100": 0.713, "age_mae_n100": 12.52, "age_r2_n100": 0.097,
        },
    }

    out_json = RESULTS_DIR / "ml_age_sex_n179.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nJSON → {out_json}")

    write_report(results, REPORT_DIR / "AGE_SEX_FASE3_N179.md")
    print("\n[FASE3_ML_DONE]")


if __name__ == "__main__":
    main()
