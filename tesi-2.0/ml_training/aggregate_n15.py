"""STEP 7 — Aggregate classification on N=15 subjects (ds005385).

Implements LOSO GroupKFold-15 CV over all (atlas × metric × classifier)
configurations. Outputs comparison_matrix_N15.json with balanced_accuracy,
bootstrap CI, and empirical p-value for each configuration.

Shape guard: skips any feature file where X.shape[0] != 30 (30 = 15 sub × 2 cond).
Fires a WARNING if shape is (10, *) — pilot N=5 data still present, Step 5 N=15
not yet complete.

Idempotent: if output JSON exists and feature hashes are unchanged, returns cached result.

Extended classifier set (superset of ml_dispatcher.Algorithm):
  - "logreg"  : LogisticRegression (C=1.0)
  - "svm_rbf" : SVC(kernel="rbf", C=1.0) — same as ml_dispatcher "svm"
  - "lda"     : LinearDiscriminantAnalysis
  - All ml_dispatcher.Algorithm values are also accepted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analysis.stats_utility import bootstrap_ci

_logger = logging.getLogger(__name__)

_N_SUBJECTS_FULL = 15
_EXPECTED_SAMPLES = _N_SUBJECTS_FULL * 2  # 30 = 15 sub × 2 cond (EO/EC)

_LOCAL_CLASSIFIERS: dict[str, object] = {
    "logreg": LogisticRegression(max_iter=2000, C=1.0, random_state=42),
    "svm_rbf": SVC(kernel="rbf", C=1.0, probability=False, random_state=42),
    "lda": LinearDiscriminantAnalysis(),
}


def _build_pipeline(clf_name: str) -> Pipeline:
    """Return a StandardScaler + classifier Pipeline for clf_name.

    Handles local extended classifiers first, then delegates to
    ml_dispatcher._make_pipeline for standard Algorithm names.
    """
    if clf_name in _LOCAL_CLASSIFIERS:
        import copy
        clf = copy.deepcopy(_LOCAL_CLASSIFIERS[clf_name])
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    # Delegate to ml_dispatcher for: svm, mlp, rf, gb
    from ml_training.ml_dispatcher import _make_pipeline  # noqa: PLC0415
    return _make_pipeline(clf_name)


@dataclass
class AggResult:
    """Result for one (atlas, metric, band, classifier) configuration."""

    atlas: str
    metric: str
    band: str
    classifier: str
    ba_mean: float
    ba_std: float
    ci_lo: float
    ci_hi: float
    p_perm: float
    n_subjects: int
    n_features: int


def _loso_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_name: str,
    *,
    n_splits: int,
) -> tuple[float, float, list[float]]:
    """LOSO GroupKFold CV. Returns (ba_mean, ba_std, ba_per_fold)."""
    gkf = GroupKFold(n_splits=n_splits)
    bas: list[float] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_idx, test_idx in gkf.split(X, y, groups):
            pipe = _build_pipeline(clf_name)
            pipe.fit(X[train_idx], y[train_idx])
            bas.append(float(balanced_accuracy_score(y[test_idx], pipe.predict(X[test_idx]))))
    arr = np.array(bas)
    return float(arr.mean()), float(arr.std()), bas


def _permutation_p(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    clf_name: str,
    observed_ba: float,
    *,
    n_permutations: int,
    n_splits: int,
    random_state: int,
) -> float:
    """Empirical p-value: P(null_BA >= observed_ba) over label permutations."""
    rng = np.random.default_rng(random_state)
    null_bas = np.zeros(n_permutations)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_permutations):
            ba, _, _ = _loso_cv(X, rng.permutation(y), groups, clf_name, n_splits=n_splits)
            null_bas[i] = ba
    return float(np.mean(null_bas >= observed_ba))


def _load_features(
    features_dir: Path,
    atlas: str,
    metric: str,
    band: str,
    expected_samples: int = _EXPECTED_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load (X, y, groups) for one (atlas, metric, band) config.

    Returns None if the npz is absent or the shape guard fires.
    Shape guard: X.shape[0] must equal expected_samples.
    """
    npz_path = features_dir / f"X_{atlas}_{metric}_{band}.npz"
    y_path = features_dir / "y.npy"
    groups_path = features_dir / "groups.npy"

    if not npz_path.exists():
        return None
    if not y_path.exists() or not groups_path.exists():
        _logger.warning("y.npy / groups.npy mancanti in %s — skip", features_dir)
        return None

    X = np.load(npz_path)["X"]
    y = np.load(y_path)
    groups = np.load(groups_path)

    n_sub_expected = expected_samples // 2
    if X.shape[0] != expected_samples:
        _logger.warning(
            "Shape guard: %s ha %d campioni, attesi %d (N=%d). "
            "Step 5 non ancora completato — skip.",
            npz_path.name, X.shape[0], expected_samples, n_sub_expected,
        )
        return None

    return X, y, groups


def _file_hash(path: Path) -> str:
    """SHA-256 (16 hex chars) of a file, or '' if absent."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _compute_hashes(
    features_dir: Path,
    atlases: list[str],
    metrics: list[str],
    band: str,
    *,
    bands: list[str] | None = None,
) -> dict:
    _bands = bands if bands is not None else [band]
    hashes: dict[str, str] = {}
    for _b in _bands:
        for a in atlases:
            for m in metrics:
                key = f"{a}_{m}_{_b}"
                hashes[key] = _file_hash(features_dir / f"X_{a}_{m}_{_b}.npz")
    return hashes


def aggregate_classify_n15(
    features_dir: Path,
    out_dir: Path,
    atlases: list[str] | None = None,
    metrics: list[str] | None = None,
    band: str = "alpha",
    bands: list[str] | None = None,
    classifiers: list[str] | None = None,
    n_permutations: int = 1000,
    random_state: int = 42,
    n_subjects: int | None = None,
) -> dict:
    """LOSO GroupKFold-15 classification on all N=15 subjects.

    Loads X_<atlas>_<metric>_<band>.npz + y.npy + groups.npy from features_dir,
    runs LOSO with GroupKFold(n_splits=n_unique_groups) for each (atlas × metric ×
    classifier) combination, computes bal_acc + empirical p-value (n_permutations)
    + bootstrap CI (1000 resamples), saves comparison_matrix_N15.json to out_dir.

    Shape guard: silently skips configs where X.shape[0] != 30, logging a WARNING.
    This allows the module to be called before Step 5 N=15 is complete without
    crashing — it simply returns {"results": [], "winner": None}.

    Idempotent: if out_dir/comparison_matrix_N15.json already exists and all
    feature file hashes match, the cached result is returned immediately.

    Parameters
    ----------
    features_dir:
        Directory containing X_*.npz + y.npy + groups.npy.
    out_dir:
        Output directory for comparison_matrix_N15.json.
    atlases:
        Atlas names to evaluate (default: ["aparc", "schaefer100"]).
    metrics:
        FC metrics to evaluate (default: ["wpli", "coh"]).
    band:
        Frequency band key in filename (default: "alpha").
    classifiers:
        Classifier names (default: ["logreg", "svm_rbf", "lda"]).
    n_permutations:
        Number of label permutations for p-value (default: 1000).
    random_state:
        RNG seed for permutation + bootstrap reproducibility.

    Returns
    -------
    dict with keys:
        "results"      : list[dict] — one dict per evaluated configuration
        "winner"       : dict — config with highest ba_mean (None if no data)
        "output_path"  : str — path to comparison_matrix_N15.json
        "_cached"      : bool (present only if result was loaded from cache)
    """
    atlases = atlases or ["aparc", "schaefer100"]
    metrics = metrics or ["wpli", "coh"]
    classifiers = classifiers or ["logreg", "svm_rbf", "lda", "rf", "mlp", "gb"]
    _bands = bands if bands is not None else [band]
    features_dir = Path(features_dir)
    out_dir = Path(out_dir)
    _expected_samples = n_subjects * 2 if n_subjects is not None else _EXPECTED_SAMPLES
    out_path = out_dir / "comparison_matrix_N15.json"

    hashes = _compute_hashes(features_dir, atlases, metrics, band, bands=bands)

    # Idempotency check
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
            if existing.get("_hashes") == hashes:
                return {
                    "results": existing["results"],
                    "winner": existing["winner"],
                    "output_path": str(out_path),
                    "_cached": True,
                }
        except (json.JSONDecodeError, KeyError):
            pass

    all_results: list[AggResult] = []

    for _band in _bands:
        for atlas in atlases:
            for metric in metrics:
                loaded = _load_features(features_dir, atlas, metric, _band, expected_samples=_expected_samples)
                if loaded is None:
                    continue
                X, y, groups = loaded
                n_splits = int(len(np.unique(groups)))

                for clf_name in classifiers:
                    ba_mean, ba_std, ba_folds = _loso_cv(X, y, groups, clf_name, n_splits=n_splits)
                    ci_lo, ci_hi = bootstrap_ci(
                        ba_folds, n_boot=1000, alpha=0.05, random_state=random_state,
                    )
                    p_perm = _permutation_p(
                        X, y, groups, clf_name, ba_mean,
                        n_permutations=n_permutations,
                        n_splits=n_splits,
                        random_state=random_state,
                    )
                    all_results.append(AggResult(
                        atlas=atlas,
                        metric=metric,
                        band=_band,
                        classifier=clf_name,
                        ba_mean=round(ba_mean, 4),
                        ba_std=round(ba_std, 4),
                        ci_lo=round(ci_lo, 4),
                        ci_hi=round(ci_hi, 4),
                        p_perm=round(p_perm, 4),
                        n_subjects=n_splits,
                        n_features=X.shape[1],
                    ))

    if not all_results:
        return {"results": [], "winner": None, "output_path": str(out_path)}

    winner = max(all_results, key=lambda r: r.ba_mean)

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "_hashes": hashes,
        "_meta": {
            "bands": _bands,
            "n_permutations": n_permutations,
            "random_state": random_state,
            "n_subjects": winner.n_subjects,
            "atlases": atlases,
            "metrics": metrics,
            "classifiers": classifiers,
        },
        "results": [asdict(r) for r in all_results],
        "winner": asdict(winner),
    }
    out_path.write_text(json.dumps(payload, indent=2))

    return {
        "results": payload["results"],
        "winner": payload["winner"],
        "output_path": str(out_path),
    }
