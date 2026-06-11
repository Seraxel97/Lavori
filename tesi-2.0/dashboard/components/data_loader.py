"""Data loader component — accesso ai dati reali della pipeline ds005385 + ds004504.

Carica dati da:
  - data/raw/ds005385/participants.tsv  → fenotipi (età, sesso)
  - data/raw/ds004504/participants.tsv  → fenotipi LEMON (età, sesso, gruppo, MMSE)
  - data/features/ds005385/            → matrici feature X_*.npz
  - data/features/ds004504/            → matrici feature ds004504 (cross-dataset)
  - data/connectivity/ds005385/        → matrici FC per-soggetto per-condizione
  - reports/                           → risultati ML precomputed + CROSS_DATASET_TRANSFER.md
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_FEAT_DIR = _ROOT / "data" / "features" / "ds005385"
_CONN_DIR = _ROOT / "data" / "connectivity" / "ds005385"
_PARTICIPANTS_TSV = _ROOT / "data" / "raw" / "ds005385" / "participants.tsv"
_REPORTS_DIR = _ROOT / "reports"
_META_FILE = _FEAT_DIR / "metadata.json"


@lru_cache(maxsize=1)
def load_metadata() -> dict[str, Any]:
    if not _META_FILE.exists():
        return {}
    return json.loads(_META_FILE.read_text())


@lru_cache(maxsize=1)
def load_cohort_df(n_subjects: int | None = None) -> pd.DataFrame:
    """Carica DataFrame cohort con età, sesso dai soggetti N=100 whitelist.

    Se participants.tsv non disponibile, restituisce un DataFrame con solo subject_id.
    """
    meta = load_metadata()
    subjects: list[str] = meta.get("subjects", [])
    if not subjects:
        from config.subjects_whitelist_n100 import WHITELIST_N100

        subjects = list(WHITELIST_N100)

    if n_subjects is not None:
        subjects = subjects[:n_subjects]

    rows: list[dict[str, Any]] = []
    pheno: dict[str, dict] = {}

    if _PARTICIPANTS_TSV.exists():
        tsv = pd.read_csv(_PARTICIPANTS_TSV, sep="\t", dtype=str)
        tsv = tsv.set_index("participant_id")
        for sub in subjects:
            if sub in tsv.index:
                row = tsv.loc[sub]
                pheno[sub] = {
                    "sex": row.get("sex", "?"),
                    "age": _safe_float(row.get("age", "")),
                    "handedness": row.get("handedness", "?"),
                }

    for sub in subjects:
        p = pheno.get(sub, {})
        rows.append(
            {
                "participant_id": sub,
                "age": p.get("age", float("nan")),
                "sex": p.get("sex", "?"),
                "handedness": p.get("handedness", "?"),
                "has_EO": _has_conn(sub, "EO"),
                "has_EC": _has_conn(sub, "EC"),
            }
        )

    return pd.DataFrame(rows)


def _safe_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def _has_conn(subject: str, cond: str, atlas: str = "aparc", metric: str = "plv", band: str = "theta") -> bool:
    fname = f"{subject}_atlas-{atlas}_cond-{cond}_metric-{metric}_band-{band}_per-epoch.npz"
    return (_CONN_DIR / fname).exists()


def list_subjects_with_connectivity(
    atlas: str = "aparc", metric: str = "plv", band: str = "theta", cond: str = "EC"
) -> list[str]:
    """Elenco soggetti con file connectivity per la combo data."""
    pattern = f"*_atlas-{atlas}_cond-{cond}_metric-{metric}_band-{band}_per-epoch.npz"
    files = sorted(_CONN_DIR.glob(pattern))
    return [f.name.split("_")[0] for f in files]


def load_connectivity_matrix(
    subject: str,
    atlas: str = "aparc",
    metric: str = "plv",
    band: str = "theta",
    cond: str = "EC",
) -> np.ndarray:
    """Carica matrice FC (n_roi × n_roi) per un soggetto.

    Raises FileNotFoundError se il file non esiste.
    """
    fname = f"{subject}_atlas-{atlas}_cond-{cond}_metric-{metric}_band-{band}_per-epoch.npz"
    path = _CONN_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Connectivity file non trovato: {path}")
    data = np.load(path, allow_pickle=True)
    return data["fc_matrix"]


def load_feature_matrix(
    atlas: str,
    metric: str,
    band: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    """Carica X, y, groups, roi_names per la combo data."""
    from dashboard.utils.data_loader import load_feature_matrix as _load

    return _load(atlas, metric, band, dataset_id="ds005385")


def list_available_combos() -> list[dict[str, str]]:
    """Elenco combo (atlas, metric, band) con file feature disponibili."""
    from dashboard.utils.data_loader import list_available_combos as _list

    return _list(dataset_id="ds005385")


def load_ml_sex_results(n100: bool = True) -> dict[str, Any]:
    """Carica risultati ML sesso (N=100 se disponibili, altrimenti N=50)."""
    fname = "ml_sex_results_n100.json" if n100 else "ml_sex_results.json"
    path = _REPORTS_DIR / fname
    if not path.exists():
        path = _REPORTS_DIR / "ml_sex_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_ml_age_results(n100: bool = True) -> dict[str, Any]:
    """Carica risultati ML età (N=100 se disponibili, altrimenti N=50)."""
    fname = "ml_age_results_n100.json" if n100 else "ml_age_results.json"
    path = _REPORTS_DIR / fname
    if not path.exists():
        path = _REPORTS_DIR / "ml_age_results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_bench_matrix() -> dict[str, Any]:
    """Carica BENCH_MATRIX_RESULTS.json (tutte le combo ML)."""
    path = _REPORTS_DIR / "BENCH_MATRIX_RESULTS.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# ds004504 (LEMON) — cross-dataset support
# ---------------------------------------------------------------------------

_FEAT_DIR_LEMON = _ROOT / "data" / "features" / "ds004504"
_PARTICIPANTS_TSV_LEMON = _ROOT / "data" / "raw" / "ds004504" / "participants.tsv"
_META_FILE_LEMON = _FEAT_DIR_LEMON / "metadata.json"

_SEX_MAP_LEMON = {"F": 0, "M": 1}


@lru_cache(maxsize=1)
def load_cohort_df_lemon() -> pd.DataFrame:
    """Carica DataFrame cohort ds004504 (LEMON) con età, sesso, gruppo, MMSE."""
    if not _META_FILE_LEMON.exists():
        return pd.DataFrame()
    meta = json.loads(_META_FILE_LEMON.read_text())
    subjects: list[str] = meta.get("subjects", [])

    pheno: dict[str, dict] = {}
    if _PARTICIPANTS_TSV_LEMON.exists():
        tsv = pd.read_csv(_PARTICIPANTS_TSV_LEMON, sep="\t", dtype=str)
        tsv = tsv.set_index("participant_id")
        for sub in subjects:
            if sub in tsv.index:
                row = tsv.loc[sub]
                pheno[sub] = {
                    "sex": row.get("Gender", "?"),
                    "age": _safe_float(row.get("Age", "")),
                    "group": row.get("Group", "?"),
                    "mmse": _safe_float(row.get("MMSE", "")),
                }

    rows = []
    for sub in subjects:
        p = pheno.get(sub, {})
        rows.append(
            {
                "participant_id": sub,
                "age": p.get("age", float("nan")),
                "sex": p.get("sex", "?"),
                "group": p.get("group", "?"),
                "mmse": p.get("mmse", float("nan")),
                "dataset": "ds004504",
            }
        )
    return pd.DataFrame(rows)


def list_available_combos_lemon() -> list[dict[str, str]]:
    """Elenco combo feature disponibili per ds004504."""
    if not _FEAT_DIR_LEMON.exists():
        return []
    combos = []
    for f in sorted(_FEAT_DIR_LEMON.glob("X_*.npz")):
        parts = f.stem[2:].split("_")
        if len(parts) == 3:
            combos.append({"atlas": parts[0], "metric": parts[1], "band": parts[2]})
    return combos


def load_feature_matrix_lemon(
    atlas: str, metric: str, band: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str] | None]:
    """Carica X, y, groups per ds004504 (LEMON).

    y = sex (0=F, 1=M) allineato ai soggetti con feature.
    groups = subject index (1 campione per soggetto).
    """
    npz_path = _FEAT_DIR_LEMON / f"X_{atlas}_{metric}_{band}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Feature file LEMON non trovato: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    roi_names: list[str] | None = data["row_labels"].tolist() if "row_labels" in data else None

    meta = json.loads(_META_FILE_LEMON.read_text()) if _META_FILE_LEMON.exists() else {}
    subjects: list[str] = meta.get("subjects", [])

    # Costruisci y_sex e groups
    y_list, groups_list = [], []
    if _PARTICIPANTS_TSV_LEMON.exists():
        tsv = pd.read_csv(_PARTICIPANTS_TSV_LEMON, sep="\t", dtype=str).set_index("participant_id")
        for i, sub in enumerate(subjects[: X.shape[0]]):
            sex_str = tsv.loc[sub, "Gender"] if sub in tsv.index else "?"
            y_list.append(_SEX_MAP_LEMON.get(sex_str, 0))
            groups_list.append(i)
    else:
        y_list = [0] * X.shape[0]
        groups_list = list(range(X.shape[0]))

    y = np.array(y_list, dtype=int)
    groups = np.array(groups_list, dtype=int)
    return X, y, groups, roi_names


def load_cross_dataset_transfer_report() -> dict[str, Any]:
    """Carica CROSS_DATASET_TRANSFER.md come dict strutturato."""
    path = _REPORTS_DIR / "CROSS_DATASET_TRANSFER.md"
    if not path.exists():
        return {}
    text = path.read_text()
    # Estrai valori chiave dal markdown
    result: dict[str, Any] = {"raw_text": text}
    import re

    # BA sesso
    m = re.search(r"Sesso.*?BA.*?(\d+\.\d+).*?\|.*?(\d+\.\d+)", text)
    if m:
        result["sex_ba_transfer"] = float(m.group(1))
        result["sex_ba_indataset"] = float(m.group(2))
    # MAE età
    m2 = re.search(r"Età.*?MAE.*?(\d+\.\d+).*?\|.*?(\d+\.\d+)", text)
    if m2:
        result["age_mae_transfer"] = float(m2.group(1))
        result["age_mae_indataset"] = float(m2.group(2))
    return result


def run_cross_dataset_transfer(
    atlas: str = "aparc",
    metric: str = "plv",
    band: str = "theta",
    clf_name: str = "logreg",
    target: str = "sex",
) -> dict[str, Any]:
    """Train su ds005385, test su ds004504 — nessun leakage.

    Parameters
    ----------
    target: "sex" o "age"

    Returns dict con: score, train_n, test_n, clf_name, target, feature.
    """
    import time

    from sklearn.metrics import balanced_accuracy_score, mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()

    # Train data (ds005385)
    X_train, _, groups_train, _ = load_feature_matrix(atlas, metric, band)

    # Carica y train da phenotype
    meta = load_metadata()
    subjects_train = meta.get("subjects", [])
    if _PARTICIPANTS_TSV.exists():
        tsv_train = pd.read_csv(_PARTICIPANTS_TSV, sep="\t", dtype=str).set_index("participant_id")
        y_train_list = []
        for sub in subjects_train:
            sex = tsv_train.loc[sub, "sex"] if sub in tsv_train.index else "?"
            y_train_list.append(0 if sex == "F" else 1)
        # Ogni soggetto ha 2 campioni (EO + EC)
        y_train_sex = np.repeat(np.array(y_train_list, dtype=int), 2)[: X_train.shape[0]]

        age_list = []
        for sub in subjects_train:
            try:
                age_list.append(float(tsv_train.loc[sub, "age"]))
            except (KeyError, ValueError):
                age_list.append(float("nan"))
        y_train_age = np.repeat(np.array(age_list, dtype=float), 2)[: X_train.shape[0]]
    else:
        y_train_sex = np.zeros(X_train.shape[0], dtype=int)
        y_train_age = np.full(X_train.shape[0], 50.0)

    y_train = y_train_sex if target == "sex" else y_train_age
    valid_mask = ~np.isnan(y_train.astype(float))
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]

    # Test data (ds004504)
    X_test, y_test_sex, _, _ = load_feature_matrix_lemon(atlas, metric, band)
    if target == "age":
        if _PARTICIPANTS_TSV_LEMON.exists():
            tsv_lemon = pd.read_csv(_PARTICIPANTS_TSV_LEMON, sep="\t", dtype=str).set_index("participant_id")
            meta_lemon = json.loads(_META_FILE_LEMON.read_text()) if _META_FILE_LEMON.exists() else {}
            subs_lemon = meta_lemon.get("subjects", [])
            y_test = np.array(
                [float(tsv_lemon.loc[s, "Age"]) if s in tsv_lemon.index else float("nan")
                 for s in subs_lemon[: X_test.shape[0]]]
            )
        else:
            y_test = np.full(X_test.shape[0], 65.0)
    else:
        y_test = y_test_sex

    # Fit su train, predict su test — scaler fit SOLO su train
    from dashboard.components.ml_runner import _build_classifier, _build_regressor  # noqa: PLC0415

    if target == "sex":
        base = _build_classifier(clf_name)
    else:
        base = _build_regressor(clf_name + "_reg" if clf_name == "rf" else "ridge")

    pipe = Pipeline([("sc", StandardScaler()), ("est", base)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    if target == "sex":
        score = float(balanced_accuracy_score(y_test, y_pred))
        metric_name = "BA"
    else:
        valid = ~np.isnan(y_test)
        score = float(mean_absolute_error(y_test[valid], y_pred[valid]))
        metric_name = "MAE"

    return {
        "score": score,
        "metric_name": metric_name,
        "target": target,
        "clf_name": clf_name,
        "feature": f"X_{atlas}_{metric}_{band}",
        "train_dataset": "ds005385",
        "test_dataset": "ds004504",
        "train_n": int(X_train.shape[0]),
        "test_n": int(X_test.shape[0]),
        "wall_clock_s": time.time() - t0,
    }
