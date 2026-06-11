"""ML step 1.5 — Predizione età (regressione) da FC features EEG.

Nested GroupKFold (outer K=5, inner K=3).
Regressori: ElasticNet, SVR, RandomForestRegressor.
Metriche: MAE, R², bootstrap CI (n_boot=1000), brain-age gap.
Confronto vs dummy (mean predictor).
Permutation test (n_perm=200, shuffle y_age).
Output: reports/ml_age_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEAT_DIR = Path(__file__).parent.parent / "data" / "features" / "ds005385"
REPORT_DIR = Path(__file__).parent.parent / "reports"

OUTER_K = 5
INNER_K = 3
N_PERM = 200
N_BOOT = 1000
FEAT_NAME = "X_aparc_plv_theta"


def load_data(feat_name: str = FEAT_NAME) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carica X, y_age, groups."""
    import json as _json

    from config.labels_phenotype import load_phenotype

    X_file = FEAT_DIR / f"{feat_name}.npz"
    arr = np.load(X_file)
    X = arr[arr.files[0]]
    groups = np.load(FEAT_DIR / "groups.npy")
    meta = _json.loads((FEAT_DIR / "metadata.json").read_text())
    y_age_per_sub, _ = load_phenotype(meta["subjects"])
    y_age = y_age_per_sub[groups]
    return X, y_age, groups


def _make_regressor(name: str, inner_cv: GroupKFold):
    if name == "elasticnet":
        return GridSearchCV(
            ElasticNet(max_iter=5000, random_state=42),
            param_grid={
                "alpha": [0.01, 0.1, 1.0, 10.0],
                "l1_ratio": [0.2, 0.5, 0.8],
            },
            cv=inner_cv,
            scoring="neg_mean_absolute_error",
            refit=True,
        )
    elif name == "svr":
        return GridSearchCV(
            SVR(kernel="rbf"),
            param_grid={"C": [0.1, 1.0, 10.0], "gamma": ["scale"]},
            cv=inner_cv,
            scoring="neg_mean_absolute_error",
            refit=True,
        )
    elif name == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif name == "dummy":
        return DummyRegressor(strategy="mean")
    else:
        raise ValueError(f"Unknown regressor: {name}")


def _cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_boot: int = N_BOOT,
    rng_seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Bootstrap CI a livello soggetto (cluster bootstrap).

    Ricampiona soggetti (non sample) per rispettare la struttura dati:
    2 sample/soggetto sono correlati → ignorarli sottostima l'incertezza.
    """
    unique_groups = np.unique(groups)
    n_subjects = len(unique_groups)
    rng = np.random.default_rng(rng_seed)
    boot_mae_list = []
    boot_r2_list = []
    for _ in range(n_boot):
        sel_groups = rng.choice(unique_groups, n_subjects, replace=True)
        boot_idx = np.concatenate([np.where(groups == g)[0] for g in sel_groups])
        boot_mae_list.append(mean_absolute_error(y_true[boot_idx], y_pred[boot_idx]))
        boot_r2_list.append(r2_score(y_true[boot_idx], y_pred[boot_idx]))
    return boot_mae_list, boot_r2_list


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    reg_name: str,
    outer_k: int = OUTER_K,
) -> dict:
    """Esegue outer CV e restituisce metriche aggregate."""
    outer_cv = GroupKFold(n_splits=outer_k)
    y_pred_all = np.empty(len(y))
    y_true_all = np.empty(len(y))

    for train_idx, test_idx in outer_cv.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_tr = groups[train_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        inner_cv = GroupKFold(n_splits=min(INNER_K, len(np.unique(g_tr))))
        reg = _make_regressor(reg_name, inner_cv)
        if hasattr(reg, "fit") and hasattr(reg, "cv"):
            reg.fit(X_tr_s, y_tr, groups=g_tr)
        else:
            reg.fit(X_tr_s, y_tr)

        y_pred_all[test_idx] = reg.predict(X_te_s)
        y_true_all[test_idx] = y_te

    mae = float(mean_absolute_error(y_true_all, y_pred_all))
    r2 = float(r2_score(y_true_all, y_pred_all))
    brain_age_gap = float(np.mean(y_pred_all - y_true_all))

    # Cluster bootstrap CI (resample soggetti, non sample)
    boot_mae, boot_r2 = _cluster_bootstrap_ci(y_true_all, y_pred_all, groups)

    return {
        "mae": mae,
        "r2": r2,
        "brain_age_gap": brain_age_gap,
        "mae_ci95": [
            float(np.percentile(boot_mae, 2.5)),
            float(np.percentile(boot_mae, 97.5)),
        ],
        "r2_ci95": [
            float(np.percentile(boot_r2, 2.5)),
            float(np.percentile(boot_r2, 97.5)),
        ],
    }


def _permute_subject_labels(y: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    """Permuta le label a livello soggetto (non sample).

    Sesso/età sono costanti per soggetto — la scambiabilità è tra soggetti,
    non tra sample. Shuffle i valori per-soggetto, poi broadcast via groups.
    """
    unique_groups = np.unique(groups)
    y_per_sub = np.array([y[groups == g][0] for g in unique_groups])
    y_perm_per_sub = rng.permutation(y_per_sub)
    y_perm = np.empty_like(y)
    for i, g in enumerate(unique_groups):
        y_perm[groups == g] = y_perm_per_sub[i]
    return y_perm


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    reg_name: str,
    n_perm: int = N_PERM,
    outer_k: int = OUTER_K,
) -> tuple[list[float], float, float]:
    """Permutation test su R²."""
    r2_real = run_cv(X, y, groups, reg_name, outer_k)["r2"]
    rng = np.random.default_rng(42)
    null_r2 = []
    for _ in range(n_perm):
        y_shuf = _permute_subject_labels(y, groups, rng)
        null_r2.append(run_cv(X, y_shuf, groups, reg_name, outer_k)["r2"])
    p_value = (sum(1 for r in null_r2 if r >= r2_real) + 1) / (n_perm + 1)
    return null_r2, r2_real, p_value


def run_ml_age(
    feat_name: str = FEAT_NAME,
    reg_names: list[str] | None = None,
    n_perm: int = N_PERM,
) -> dict:
    """Pipeline completa ML età."""
    if reg_names is None:
        reg_names = ["elasticnet", "svr", "rf", "dummy"]
    X, y_age, groups = load_data(feat_name)
    results: dict = {
        "feat_name": feat_name,
        "n_samples": int(len(y_age)),
        "n_subjects": int(len(np.unique(groups))),
        "y_age_mean": float(np.mean(y_age)),
        "y_age_std": float(np.std(y_age)),
        "regressors": {},
    }
    for reg_name in reg_names:
        cv_res = run_cv(X, y_age, groups, reg_name)
        if reg_name != "dummy":
            null_r2, r2_real, p_value = permutation_test(
                X, y_age, groups, reg_name, n_perm
            )
        else:
            null_r2, p_value = [], float("nan")
        results["regressors"][reg_name] = {
            **cv_res,
            "p_value": float(p_value),
            "null_r2_mean": float(np.mean(null_r2)) if null_r2 else float("nan"),
            "n_perm": n_perm,
        }
    return results


def save_results(results: dict, path: Path | None = None) -> Path:
    if path is None:
        REPORT_DIR.mkdir(exist_ok=True)
        path = REPORT_DIR / "ml_age_results.json"
    path.write_text(json.dumps(results, indent=2))
    return path


if __name__ == "__main__":
    res = run_ml_age()
    p = save_results(res)
    print(f"Salvato: {p}")
    for reg, r in res["regressors"].items():
        print(f"  {reg}: MAE={r['mae']:.2f} R²={r['r2']:.3f} p={r['p_value']:.4f}")
