"""Confronto feature: X_baseline (FC 2278-dim) vs X_graphtheory (8-dim) per ML sesso/età.

Rigore: stesso nested GroupKFold, permutation, FDR-BH su confronto multiplo.
Output: reports/feature_comparison_n50.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import fdrcorrection

from features.graph_theory import compute_graph_metrics
from ml_training.ml_age import load_data as load_age_data
from ml_training.ml_age import run_cv as run_cv_age
from ml_training.ml_sex import load_data as load_sex_data
from ml_training.ml_sex import run_cv as run_cv_sex

FEAT_DIR = Path(__file__).parent.parent / "data" / "features" / "ds005385"
REPORT_DIR = Path(__file__).parent.parent / "reports"
N_NODES_APARC = 68
N_PERM = 50
OUTER_K = 5

GT_FEATURE_ORDER = [
    "clustering_coeff",
    "global_efficiency",
    "local_efficiency",
    "mean_degree",
    "mean_strength",
    "modularity_q",
    "path_length",
    "small_worldness",
]


def build_X_graphtheory(
    feat_name: str = "X_aparc_plv_theta",
    n_nodes: int = N_NODES_APARC,
) -> np.ndarray:
    """Costruisce X_gt (n_samples, 8) da X_fc (n_samples, n_fc_flat).

    Ricostruisce W (n_nodes×n_nodes) per ogni riga e calcola 8 metriche GT.
    path_length=inf e small_worldness=nan vengono sostituiti con valori mediani.
    """
    X_file = FEAT_DIR / f"{feat_name}.npz"
    arr = np.load(X_file)
    X_fc = arr[arr.files[0]]

    idx_i, idx_j = np.triu_indices(n_nodes, k=1)
    rows = []
    for x in X_fc:
        W = np.zeros((n_nodes, n_nodes))
        W[idx_i, idx_j] = x
        W = W + W.T
        m = compute_graph_metrics(W)
        rows.append([m[k] for k in GT_FEATURE_ORDER])

    X_gt = np.array(rows, dtype=float)

    # Imputa inf e NaN con la mediana della colonna
    for col in range(X_gt.shape[1]):
        col_data = X_gt[:, col]
        finite_mask = np.isfinite(col_data)
        if finite_mask.sum() > 0:
            median_val = np.median(col_data[finite_mask])
            X_gt[~finite_mask, col] = median_val
    return X_gt


def _permute_subject_labels(
    y: np.ndarray, groups: np.ndarray, rng
) -> np.ndarray:
    """Permuta label a livello soggetto (non sample) — stessa logica di ml_sex/ml_age."""
    unique_groups = np.unique(groups)
    y_per_sub = np.array([y[groups == g][0] for g in unique_groups])
    y_perm_per_sub = rng.permutation(y_per_sub)
    y_perm = np.empty_like(y)
    for i, g in enumerate(unique_groups):
        y_perm[groups == g] = y_perm_per_sub[i]
    return y_perm


def _permutation_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    task: str,
    n_perm: int = N_PERM,
) -> tuple[float, float, float]:
    """Returns (score_real, null_mean, p_value). Permutation a livello soggetto."""
    if task == "sex":
        real_res = run_cv_sex(X, y, groups, "logreg", outer_k=OUTER_K)
        score_real = real_res["ba_mean"]
        null_scores = []
        rng = np.random.default_rng(42)
        for _ in range(n_perm):
            y_shuf = _permute_subject_labels(y, groups, rng)
            null_scores.append(
                run_cv_sex(X, y_shuf, groups, "logreg", outer_k=OUTER_K)["ba_mean"]
            )
    else:
        # RF invece di ElasticNet: converge su p≫n, R²>0 dimostrato su N=50
        real_res = run_cv_age(X, y, groups, "rf", outer_k=OUTER_K)
        score_real = real_res["r2"]
        null_scores = []
        rng = np.random.default_rng(42)
        for _ in range(n_perm):
            y_shuf = _permute_subject_labels(y, groups, rng)
            null_scores.append(
                run_cv_age(X, y_shuf, groups, "rf", outer_k=OUTER_K)["r2"]
            )

    null_mean = float(np.mean(null_scores))
    p_value = (sum(1 for s in null_scores if s >= score_real) + 1) / (n_perm + 1)
    return score_real, null_mean, p_value


def run_comparison(
    feat_name: str = "X_aparc_plv_theta",
    n_perm: int = N_PERM,
) -> dict:
    """Confronto completo baseline vs GT per sesso e età."""
    # Carica dati
    X_baseline, y_sex, groups = load_sex_data(feat_name)
    X_age, y_age, _ = load_age_data(feat_name)

    print("Costruendo X_graphtheory (può richiedere ~30s)...")
    X_gt = build_X_graphtheory(feat_name)

    results: dict = {
        "feat_name": feat_name,
        "n_samples": int(X_baseline.shape[0]),
        "n_subjects": int(len(np.unique(groups))),
        "n_features_baseline": int(X_baseline.shape[1]),
        "n_features_gt": int(X_gt.shape[1]),
        "gt_feature_names": GT_FEATURE_ORDER,
        "comparisons": {},
    }

    # 4 confronti: sex_baseline, sex_gt, age_baseline, age_gt
    configs = [
        ("sex_baseline", X_baseline, y_sex, "sex"),
        ("sex_gt", X_gt, y_sex, "sex"),
        ("age_baseline", X_baseline, y_age, "age"),
        ("age_gt", X_gt, y_age, "age"),
    ]

    p_values = []
    scores = {}
    for name, X, y, task in configs:
        print(f"Running {name}...")
        score, null_mean, p_val = _permutation_cv(X, y, groups, task, n_perm)
        scores[name] = {"score": score, "null_mean": null_mean, "p_raw": p_val}
        p_values.append(p_val)

    # FDR-BH correction
    _, p_fdr = fdrcorrection(p_values, alpha=0.05, method="indep")

    for i, (name, _, _, task) in enumerate(configs):
        metric = "ba_mean" if task == "sex" else "r2"
        results["comparisons"][name] = {
            "metric": metric,
            "score": float(scores[name]["score"]),
            "null_mean": float(scores[name]["null_mean"]),
            "p_raw": float(scores[name]["p_raw"]),
            "p_fdr": float(p_fdr[i]),
            "significant_fdr": bool(p_fdr[i] < 0.05),
        }

    return results


def save_results(results: dict, path: Path | None = None) -> Path:
    """Salva risultati confronto feature su JSON."""
    if path is None:
        REPORT_DIR.mkdir(exist_ok=True)
        path = REPORT_DIR / "feature_comparison_n50.json"
    path.write_text(json.dumps(results, indent=2))
    return path


if __name__ == "__main__":
    res = run_comparison()
    p = save_results(res)
    print(f"\nSalvato: {p}")
    print("\n=== CONFRONTO FEATURE ===")
    for name, r in res["comparisons"].items():
        sig = "✓ sig" if r["significant_fdr"] else "  ns"
        print(
            f"  {name}: {r['metric']}={r['score']:.3f} "
            f"p_raw={r['p_raw']:.4f} p_fdr={r['p_fdr']:.4f} {sig}"
        )
