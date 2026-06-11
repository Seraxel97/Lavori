"""ML runner component — GroupKFold enforced, no leakage.

Wrappa ml_training/ con garanzie:
  - GroupKFold subject-level (ogni soggetto escluso dal test in ogni fold)
  - Bootstrap CI (BCa) su ba_per_fold
  - Permutation test opzionale
  - Nessuna possibilità di train su soggetto presente nel test set

Nota scientifica: n_perm=0 salta il permutation test per velocità interattiva;
  risultati scientifici ufficiali usano n_perm≥200 (vedi EXPERIMENTS_N100.md).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_CLF_NAMES = ("logreg", "svm", "mlp", "rf", "gb")
_REG_NAMES = ("ridge", "rf_reg", "svr", "mlp_reg")


def _build_classifier(name: str):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    _map = {
        "logreg": LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver="liblinear"),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
        "rf": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "gb": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "lda": LinearDiscriminantAnalysis(),
    }
    return _map.get(name, _map["logreg"])


def _build_regressor(name: str):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR

    _map = {
        "ridge": Ridge(alpha=1.0),
        "rf_reg": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "svr": SVR(kernel="rbf"),
        "mlp_reg": MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
    }
    return _map.get(name, _map["ridge"])


def run_classification(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_name: str = "logreg",
    n_perm: int = 0,
    n_boot: int = 200,
) -> dict[str, Any]:
    """Classificazione GroupKFold con bootstrap CI e opzionale permutation test.

    Parameters
    ----------
    X:   feature matrix (n_samples × n_features)
    y:   etichette binarie (n_samples,)
    groups: subject ID array (n_samples,) — impone GroupKFold
    clf_name: nome classificatore
    n_perm: numero permutazioni (0 = skip)
    n_boot: numero bootstrap resample per CI

    Returns
    -------
    dict con: ba, ba_std, ci_lo, ci_hi, p_perm, null_distribution,
              ba_per_fold, confusion, roc_fpr, roc_tpr, roc_auc,
              n_features, n_samples, n_subjects, wall_clock_s
    """
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    n_splits = int(np.unique(groups).shape[0])
    base_clf = _build_classifier(clf_name)
    pipe = Pipeline([("sc", StandardScaler()), ("clf", base_clf)])
    gkf = GroupKFold(n_splits=n_splits)

    ba_folds, y_true_all, y_score_all = [], [], []
    conf = np.zeros((2, 2), dtype=int)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        _assert_no_leakage(groups, train_idx, test_idx)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])
        ba_folds.append(float(balanced_accuracy_score(y[test_idx], y_pred)))
        conf += confusion_matrix(y[test_idx], y_pred, labels=[0, 1])
        if hasattr(pipe, "predict_proba"):
            scores = pipe.predict_proba(X[test_idx])[:, 1]
        else:
            scores = pipe.decision_function(X[test_idx])
        y_true_all.extend(y[test_idx].tolist())
        y_score_all.extend(scores.tolist())

    ba_arr = np.array(ba_folds)
    ci_lo, ci_hi = _bootstrap_ci(ba_arr, n_boot=n_boot)

    fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
    auc_val = float(roc_auc_score(y_true_all, y_score_all))

    null_dist: list[float] = []
    p_perm: float | None = None
    if n_perm > 0:
        obs_ba = float(ba_arr.mean())
        rng = np.random.default_rng(42)
        null_bas = []
        for _ in range(n_perm):
            y_perm = rng.permutation(y)
            fold_bas = []
            for train_idx, test_idx in gkf.split(X, y_perm, groups=groups):
                pipe.fit(X[train_idx], y_perm[train_idx])
                y_pred = pipe.predict(X[test_idx])
                fold_bas.append(float(balanced_accuracy_score(y_perm[test_idx], y_pred)))
            null_bas.append(float(np.mean(fold_bas)))
        null_dist = null_bas
        p_perm = float((np.sum(np.array(null_bas) >= obs_ba) + 1) / (n_perm + 1))

    return {
        "ba": float(ba_arr.mean()),
        "ba_std": float(ba_arr.std()),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_perm": p_perm,
        "n_perm": n_perm,
        "null_distribution": null_dist,
        "ba_per_fold": list(ba_arr),
        "confusion": conf.tolist(),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "roc_auc": auc_val,
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "n_subjects": int(np.unique(groups).shape[0]),
        "wall_clock_s": time.time() - t0,
        "clf_name": clf_name,
    }


def run_regression(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    reg_name: str = "ridge",
    n_perm: int = 0,
    n_boot: int = 200,
) -> dict[str, Any]:
    """Regressione GroupKFold per target continuo (età).

    Returns dict con: mae, r2, ci_mae_lo, ci_mae_hi, p_perm, wall_clock_s.
    """
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    n_splits = int(np.unique(groups).shape[0])
    base_reg = _build_regressor(reg_name)
    pipe = Pipeline([("sc", StandardScaler()), ("reg", base_reg)])
    gkf = GroupKFold(n_splits=n_splits)

    mae_folds, y_true_all, y_pred_all = [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        _assert_no_leakage(groups, train_idx, test_idx)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[test_idx])
        mae_folds.append(float(mean_absolute_error(y[test_idx], y_pred)))
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(y_pred.tolist())

    mae_arr = np.array(mae_folds)
    ci_lo, ci_hi = _bootstrap_ci(mae_arr, n_boot=n_boot)
    r2 = float(r2_score(y_true_all, y_pred_all))

    p_perm: float | None = None
    if n_perm > 0:
        obs_mae = float(mae_arr.mean())
        rng = np.random.default_rng(42)
        null_maes = []
        for _ in range(n_perm):
            y_perm = rng.permutation(y)
            fold_maes = []
            for train_idx, test_idx in gkf.split(X, y_perm, groups=groups):
                pipe.fit(X[train_idx], y_perm[train_idx])
                y_pred = pipe.predict(X[test_idx])
                fold_maes.append(float(mean_absolute_error(y_perm[test_idx], y_pred)))
            null_maes.append(float(np.mean(fold_maes)))
        # p-value: proporzione permutazioni con MAE <= osservato (null ≤ obs per regressione)
        p_perm = float((np.sum(np.array(null_maes) <= obs_mae) + 1) / (n_perm + 1))

    return {
        "mae": float(mae_arr.mean()),
        "mae_std": float(mae_arr.std()),
        "ci_mae_lo": float(ci_lo),
        "ci_mae_hi": float(ci_hi),
        "r2": r2,
        "p_perm": p_perm,
        "n_perm": n_perm,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "n_features": int(X.shape[1]),
        "n_samples": int(X.shape[0]),
        "n_subjects": int(np.unique(groups).shape[0]),
        "wall_clock_s": time.time() - t0,
        "reg_name": reg_name,
    }


def _assert_no_leakage(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Verifica runtime che nessun soggetto appare sia in train che in test."""
    train_groups = set(groups[train_idx].tolist())
    test_groups = set(groups[test_idx].tolist())
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Subject leakage rilevato: {overlap}")


def _bootstrap_ci(
    arr: np.ndarray, n_boot: int = 200, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap percentile CI (semplificato; BCa in stats_utility per risultati ufficiali)."""
    rng = np.random.default_rng(42)
    boot_means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi
