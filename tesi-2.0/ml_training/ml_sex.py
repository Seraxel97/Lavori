"""ML step 1.4 — Classificazione sesso da FC features EEG.

Nested GroupKFold (outer K=5, inner K=3).
Classificatori: logreg-L2, RF, SVM con class_weight='balanced'.
Metriche: balanced_accuracy, AUC-ROC.
Permutation test (n_perm=200, shuffle y_sex).
Output: reports/ml_sex_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEAT_DIR = Path(__file__).parent.parent / "data" / "features" / "ds005385"
REPORT_DIR = Path(__file__).parent.parent / "reports"

OUTER_K = 5
INNER_K = 3
N_PERM = 200
FEAT_NAME = "X_aparc_plv_theta"


def load_data(feat_name: str = FEAT_NAME) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carica X, y_sex, groups."""
    import json as _json

    from config.labels_phenotype import load_phenotype

    X_file = FEAT_DIR / f"{feat_name}.npz"
    arr = np.load(X_file)
    X = arr[arr.files[0]]

    groups = np.load(FEAT_DIR / "groups.npy")
    meta = _json.loads((FEAT_DIR / "metadata.json").read_text())
    subjects = meta["subjects"]

    _, y_sex_per_sub = load_phenotype(subjects)
    y_sex = y_sex_per_sub[groups]

    return X, y_sex, groups


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_name: str,
    outer_k: int = OUTER_K,
) -> dict:
    """Esegue outer CV e restituisce metriche per fold."""
    outer_cv = GroupKFold(n_splits=outer_k)
    ba_folds = []
    auc_folds = []

    for train_idx, test_idx in outer_cv.split(X, y, groups=groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        g_tr = groups[train_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Costruisci inner classifier
        inner_cv = GroupKFold(n_splits=min(INNER_K, len(np.unique(g_tr))))
        if clf_name == "logreg":
            clf = GridSearchCV(
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
                cv=inner_cv,
                scoring="balanced_accuracy",
                refit=True,
            )
            clf.fit(X_tr_s, y_tr, groups=g_tr)
        elif clf_name == "rf":
            clf = RandomForestClassifier(
                n_estimators=100, class_weight="balanced", random_state=42
            )
            clf.fit(X_tr_s, y_tr)
        elif clf_name == "svm":
            clf = GridSearchCV(
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=42,
                ),
                param_grid={"C": [0.1, 1.0, 10.0]},
                cv=inner_cv,
                scoring="balanced_accuracy",
                refit=True,
            )
            clf.fit(X_tr_s, y_tr, groups=g_tr)
        else:
            raise ValueError(f"Unknown classifier: {clf_name}")

        y_pred = clf.predict(X_te_s)
        ba_folds.append(balanced_accuracy_score(y_te, y_pred))
        try:
            y_prob = clf.predict_proba(X_te_s)[:, 1]
            auc_folds.append(roc_auc_score(y_te, y_prob))
        except Exception:
            auc_folds.append(float("nan"))

    return {
        "ba_folds": [float(b) for b in ba_folds],
        "ba_mean": float(np.mean(ba_folds)),
        "ba_std": float(np.std(ba_folds)),
        "auc_mean": float(np.nanmean(auc_folds)),
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
    clf_name: str,
    n_perm: int = N_PERM,
    outer_k: int = OUTER_K,
) -> tuple[list[float], float, float]:
    """Permutation test: shuffle y → distribuzione null di BA."""
    ba_real = run_cv(X, y, groups, clf_name, outer_k)["ba_mean"]
    rng = np.random.default_rng(42)
    null_bas = []
    for _ in range(n_perm):
        y_shuf = _permute_subject_labels(y, groups, rng)
        res = run_cv(X, y_shuf, groups, clf_name, outer_k)
        null_bas.append(res["ba_mean"])
    p_value = (sum(1 for b in null_bas if b >= ba_real) + 1) / (n_perm + 1)
    return null_bas, ba_real, p_value


def run_ml_sex(
    feat_name: str = FEAT_NAME,
    clf_names: list[str] | None = None,
    n_perm: int = N_PERM,
) -> dict:
    """Pipeline completa ML sesso."""
    if clf_names is None:
        clf_names = ["logreg", "rf", "svm"]
    X, y_sex, groups = load_data(feat_name)
    results: dict = {
        "feat_name": feat_name,
        "n_samples": int(len(y_sex)),
        "n_subjects": int(len(np.unique(groups))),
        "class_counts": {
            "F(0)": int(np.sum(y_sex == 0)),
            "M(1)": int(np.sum(y_sex == 1)),
        },
        "classifiers": {},
    }
    for clf_name in clf_names:
        cv_res = run_cv(X, y_sex, groups, clf_name)
        null_bas, ba_real, p_value = permutation_test(
            X, y_sex, groups, clf_name, n_perm
        )
        results["classifiers"][clf_name] = {
            **cv_res,
            "p_value": float(p_value),
            "null_ba_mean": float(np.mean(null_bas)),
            "n_perm": n_perm,
        }
    return results


def save_results(results: dict, path: Path | None = None) -> Path:
    """Salva risultati ML su JSON."""
    if path is None:
        REPORT_DIR.mkdir(exist_ok=True)
        path = REPORT_DIR / "ml_sex_results.json"
    path.write_text(json.dumps(results, indent=2))
    return path


if __name__ == "__main__":
    res = run_ml_sex()
    p = save_results(res)
    print(f"Salvato: {p}")
    for clf, r in res["classifiers"].items():
        print(f"  {clf}: BA={r['ba_mean']:.3f}±{r['ba_std']:.3f}  p={r['p_value']:.4f}")
