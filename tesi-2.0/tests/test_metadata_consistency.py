"""Test F-D: metadata.json consistency with y.npy and groups.npy (ds005385).

Asserts:
- meta['n_subjects'] == len(np.unique(groups))
- meta['n_samples'] == len(y)
- len(meta['row_order']) == len(y)
- row_order matches groups/y ordering (EO=0, EC=1)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FEAT_DIR = Path(__file__).parent.parent / "data" / "features" / "ds005385"


@pytest.fixture
def meta() -> dict:
    path = FEAT_DIR / "metadata.json"
    assert path.exists(), f"metadata.json non trovato: {path}"
    return json.loads(path.read_text())


@pytest.fixture
def y() -> np.ndarray:
    return np.load(FEAT_DIR / "y.npy")


@pytest.fixture
def groups() -> np.ndarray:
    return np.load(FEAT_DIR / "groups.npy")


def test_n_subjects_matches_unique_groups(meta: dict, groups: np.ndarray) -> None:
    """meta['n_subjects'] deve corrispondere al numero di soggetti unici in groups."""
    assert meta["n_subjects"] == len(np.unique(groups)), (
        f"n_subjects={meta['n_subjects']} != unique groups={len(np.unique(groups))}"
    )


def test_n_samples_matches_y_length(meta: dict, y: np.ndarray) -> None:
    """meta['n_samples'] deve corrispondere alla lunghezza di y."""
    assert meta["n_samples"] == len(y), (
        f"n_samples={meta['n_samples']} != len(y)={len(y)}"
    )


def test_row_order_length_matches_y(meta: dict, y: np.ndarray) -> None:
    """len(meta['row_order']) deve corrispondere alla lunghezza di y."""
    assert len(meta["row_order"]) == len(y), (
        f"len(row_order)={len(meta['row_order'])} != len(y)={len(y)}"
    )


def test_row_order_group_alignment(meta: dict, groups: np.ndarray, y: np.ndarray) -> None:
    """row_order[i] deve essere '<subject>_<cond>' coerente con groups[i] e y[i]."""
    subjects = meta["subjects"]
    y_encoding: dict[str, int] = meta["y_encoding"]  # {'EO': 0, 'EC': 1}
    # Invert encoding: int -> condition label
    enc_inv = {v: k for k, v in y_encoding.items()}

    for i, label in enumerate(meta["row_order"]):
        parts = label.rsplit("_", 1)
        assert len(parts) == 2, f"row_order[{i}]='{label}' non ha formato '<sub>_<cond>'"
        sub_id, cond = parts
        expected_sub = subjects[groups[i]]
        assert sub_id == expected_sub, (
            f"row_order[{i}]: soggetto '{sub_id}' != atteso '{expected_sub}' (group={groups[i]})"
        )
        expected_cond = enc_inv[int(y[i])]
        assert cond == expected_cond, (
            f"row_order[{i}]: condizione '{cond}' != atteso '{expected_cond}' (y={y[i]})"
        )


def test_subjects_list_length(meta: dict) -> None:
    """meta['subjects'] deve avere esattamente n_subjects elementi."""
    assert len(meta["subjects"]) == meta["n_subjects"], (
        f"len(subjects)={len(meta['subjects'])} != n_subjects={meta['n_subjects']}"
    )


def test_no_nan_in_y(y: np.ndarray) -> None:
    """y.npy non deve contenere NaN."""
    assert not np.any(np.isnan(y.astype(float))), "y.npy contiene NaN"


def test_y_binary(y: np.ndarray) -> None:
    """y deve contenere solo valori binari {0, 1}."""
    unique_vals = set(y.tolist())
    assert unique_vals <= {0, 1}, f"y contiene valori non binari: {unique_vals}"
