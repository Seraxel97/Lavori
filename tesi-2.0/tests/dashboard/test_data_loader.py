"""Test data_loader component — usa fixture reali (no mock).

Verifica:
  - load_cohort_df() restituisce DataFrame con colonne attese
  - list_subjects_with_connectivity() trova soggetti reali
  - load_connectivity_matrix() carica matrice (68, 68)
  - list_available_combos() lista combo esistenti
  - load_feature_matrix() carica X con shape corretta
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.components.data_loader import (  # noqa: E402
    list_available_combos,
    list_subjects_with_connectivity,
    load_cohort_df,
    load_connectivity_matrix,
    load_feature_matrix,
    load_metadata,
)


def test_load_metadata_returns_dict():
    meta = load_metadata()
    assert isinstance(meta, dict)
    # Se metadata.json esiste deve avere chiavi attese
    if meta:
        assert "n_subjects" in meta or "subjects" in meta


def test_load_cohort_df_columns():
    df = load_cohort_df()
    assert "participant_id" in df.columns
    assert "has_EC" in df.columns
    assert "has_EO" in df.columns
    assert len(df) > 0


def test_load_cohort_df_n_subjects():
    df10 = load_cohort_df(n_subjects=10)
    assert len(df10) == 10


def test_list_subjects_with_connectivity_returns_list():
    subjects = list_subjects_with_connectivity(atlas="aparc", metric="plv", band="theta", cond="EC")
    assert isinstance(subjects, list)
    # Deve trovare soggetti reali (ds005385 processato)
    assert len(subjects) > 0, "Nessun soggetto con connectivity aparc×plv×theta×EC trovato"


def test_load_connectivity_matrix_shape():
    subjects = list_subjects_with_connectivity(atlas="aparc", metric="plv", band="theta", cond="EC")
    if not subjects:
        pytest.skip("Nessun soggetto con connectivity disponibile")
    W = load_connectivity_matrix(subjects[0], atlas="aparc", metric="plv", band="theta", cond="EC")
    assert isinstance(W, np.ndarray)
    assert W.ndim == 2
    assert W.shape[0] == W.shape[1]
    assert W.shape[0] > 0


def test_load_connectivity_matrix_finite():
    subjects = list_subjects_with_connectivity(atlas="aparc", metric="plv", band="theta", cond="EC")
    if not subjects:
        pytest.skip("Nessun soggetto con connectivity disponibile")
    W = load_connectivity_matrix(subjects[0], atlas="aparc", metric="plv", band="theta", cond="EC")
    assert np.all(np.isfinite(W)), "Matrice FC contiene NaN o Inf"


def test_load_connectivity_matrix_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_connectivity_matrix(
            "sub-NONEXISTENT", atlas="aparc", metric="plv", band="theta", cond="EC"
        )


def test_list_available_combos_nonempty():
    combos = list_available_combos()
    assert isinstance(combos, list)
    assert len(combos) > 0
    for c in combos:
        assert "atlas" in c
        assert "metric" in c
        assert "band" in c


def test_load_feature_matrix_shape():
    combos = list_available_combos()
    if not combos:
        pytest.skip("Nessuna combo feature disponibile")
    c = combos[0]
    X, y, groups, roi_names = load_feature_matrix(c["atlas"], c["metric"], c["band"])
    assert X.ndim == 2
    assert y.ndim == 1
    assert groups.ndim == 1
    assert X.shape[0] == y.shape[0] == groups.shape[0]
    assert X.shape[0] > 0


def test_load_feature_matrix_no_leakage_possible():
    """Verifica che groups contenga più di 1 soggetto unico (GroupKFold possibile)."""
    combos = list_available_combos()
    if not combos:
        pytest.skip("Nessuna combo feature disponibile")
    c = combos[0]
    _, _, groups, _ = load_feature_matrix(c["atlas"], c["metric"], c["band"])
    n_unique = len(np.unique(groups))
    assert n_unique >= 2, f"Solo {n_unique} soggetto unico — GroupKFold impossibile"
