"""Cross-dataset evaluation: train ds005385 → test ds004504.

Protocollo:
- Train: ds005385 N=100 (il miglior modello da step 1.6 N100)
- Test: ds004504 N=<N> soggetti con features complete
- NO data leakage: scaler fit su train, transform su test
- Feature: X_aparc_plv_theta (miglior combo da N50/N100 ds005385)
- Task: sex (classificazione BA) + age (regressione MAE)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

DS005385_FEAT = Path("data/features/ds005385")
DS004504_FEAT = Path("data/features/ds004504")
REPORT_PATH = Path("reports/CROSS_DATASET_TRANSFER.md")

FEAT_NAME = "X_aparc_plv_theta"   # miglior combo da step 1.6 N100


def load_ds005385(feat_name: str = FEAT_NAME):
    """Carica features ds005385 N=100 (con 2 condizioni EO/EC)."""
    from config.labels_phenotype import load_phenotype
    arr = np.load(DS005385_FEAT / f"{feat_name}.npz")
    X = arr[arr.files[0]]
    meta = json.loads((DS005385_FEAT / "metadata.json").read_text())
    subjects_list = meta["subjects"]  # lista dei label "sub-XXX_EO", "sub-XXX_EC"
    n_samples = meta.get("n_samples", X.shape[0])
    _n_subjects = meta.get("n_subjects", n_samples // 2)

    # Estrai subject ID unici (senza _EO/_EC)
    unique_subs = []
    for s in subjects_list:
        base = s.split("_")[0]  # "sub-XXX"
        if base not in unique_subs:
            unique_subs.append(base)

    groups = np.load(DS005385_FEAT / "groups.npy")
    y_age_base, y_sex_base = load_phenotype(unique_subs)

    # Replica labels per ogni condizione (EO+EC = 2x per soggetto)
    y_age = np.repeat(y_age_base, 2)[:n_samples]
    y_sex = np.repeat(y_sex_base, 2)[:n_samples]

    return X, y_age, y_sex, groups


def load_ds004504(feat_name: str = FEAT_NAME):
    """Carica features ds004504."""
    from config.labels_phenotype_ds004504 import load_phenotype_ds004504
    arr = np.load(DS004504_FEAT / f"{feat_name}.npz")
    X = arr[arr.files[0]]
    meta = json.loads((DS004504_FEAT / "metadata.json").read_text())
    subjects = meta["subjects"]
    groups = np.load(DS004504_FEAT / "groups.npy")
    y_age, y_sex = load_phenotype_ds004504(subjects)
    return X, y_age, y_sex, groups


def run_transfer():
    """Esegui transfer learning e salva report."""
    print("Loading ds005385 (train)...")
    X_train, y_age_tr, y_sex_tr, _ = load_ds005385()
    print(f"  Train: X={X_train.shape}, sex={np.bincount(y_sex_tr)}, age=[{y_age_tr.min():.0f},{y_age_tr.max():.0f}]")

    print("Loading ds004504 (test)...")
    X_test, y_age_te, y_sex_te, _ = load_ds004504()
    print(f"  Test:  X={X_test.shape}, sex={np.bincount(y_sex_te)}, age=[{y_age_te.min():.0f},{y_age_te.max():.0f}]")

    # SCALER: fit su train ONLY
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # SEX — logreg (come step 1.6 N100)
    clf_sex = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
    clf_sex.fit(X_train_sc, y_sex_tr)
    y_pred_sex = clf_sex.predict(X_test_sc)
    ba_sex = balanced_accuracy_score(y_sex_te, y_pred_sex)
    print(f"Transfer sex BA = {ba_sex:.3f}")

    # AGE — RF (come step 1.6 N100)
    reg_age = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    reg_age.fit(X_train_sc, y_age_tr)
    y_pred_age = reg_age.predict(X_test_sc)
    mae_age = mean_absolute_error(y_age_te, y_pred_age)
    print(f"Transfer age MAE = {mae_age:.2f} anni")

    # In-dataset baseline (train=test, leave-one-sub-out stima approssimata)
    from sklearn.metrics import balanced_accuracy_score as bac
    from sklearn.metrics import mean_absolute_error as mae_fn
    from sklearn.model_selection import GroupKFold
    groups_tr = np.load(DS005385_FEAT / "groups.npy")
    cv = GroupKFold(n_splits=5)
    bas_in, maes_in = [], []
    for tr_idx, te_idx in cv.split(X_train_sc, y_sex_tr, groups_tr):
        Xtr, Xte = X_train_sc[tr_idx], X_train_sc[te_idx]
        s = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = s.transform(Xtr), s.transform(Xte)
        clf_s = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000, random_state=42)
        clf_s.fit(Xtr_s, y_sex_tr[tr_idx])
        bas_in.append(bac(y_sex_tr[te_idx], clf_s.predict(Xte_s)))
        reg_a = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        reg_a.fit(Xtr_s, y_age_tr[tr_idx])
        maes_in.append(mae_fn(y_age_tr[te_idx], reg_a.predict(Xte_s)))
    ba_in = float(np.mean(bas_in))
    mae_in = float(np.mean(maes_in))

    # Salva report
    report = f"""# Cross-Dataset Transfer: ds005385 → ds004504

**Data**: {__import__('datetime').date.today()}
**Feature**: {FEAT_NAME}
**Train**: ds005385 N={X_train.shape[0]} osservazioni (N=100 soggetti)
**Test**: ds004504 N={X_test.shape[0]} osservazioni (N={len(np.unique(np.load(DS004504_FEAT / 'groups.npy')))} soggetti)

## Risultati Transfer

| Task | Metrica | Transfer (train=ds005385, test=ds004504) | In-dataset ds005385 (5-fold CV) |
|------|---------|------------------------------------------|---------------------------------|
| Sesso | BA | {ba_sex:.3f} | {ba_in:.3f} |
| Età | MAE (anni) | {mae_age:.2f} | {mae_in:.2f} |

## Interpretazione

- **Sesso transfer BA = {ba_sex:.3f}**: {'sopra' if ba_sex > 0.6 else 'vicino a'} chance (0.5). {'Generalizzazione positiva.' if ba_sex > 0.6 else 'Generalizzazione limitata.'}
- **Età transfer MAE = {mae_age:.2f} anni**: confronto con in-dataset {mae_in:.2f} anni.
- Gap transfer vs in-dataset: sex BA {ba_in - ba_sex:+.3f} | age MAE {mae_age - mae_in:+.2f}

## Note metodologiche

- Scaler fit SOLO su train (ds005385), transform su test (ds004504) — no data leakage
- Feature: {FEAT_NAME} — miglior combo da analisi N100 ds005385
- Dataset eterogeni: ds005385 = adulti sani ({y_age_tr.min():.0f}–{y_age_tr.max():.0f} anni), ds004504 = Alzheimer + controlli ({y_age_te.min():.0f}–{y_age_te.max():.0f} anni)
"""
    REPORT_PATH.write_text(report)
    print(f"Report salvato: {REPORT_PATH}")

    return {"ba_sex_transfer": round(ba_sex, 3), "mae_age_transfer": round(mae_age, 2),
            "ba_sex_indataset": round(ba_in, 3), "mae_age_indataset": round(mae_in, 2)}


if __name__ == "__main__":
    results = run_transfer()
    print("DONE:", results)
