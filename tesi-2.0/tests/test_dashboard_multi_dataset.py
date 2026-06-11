"""Test dashboard multi-dataset: list_available_datasets + validate_dataset_schema.

Usa tmp_path pytest per creare filesystem mock con 2 fake dataset.
NON richiede Streamlit running.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dashboard.utils.data_loader import (
    list_available_datasets,
    validate_dataset_schema,
)


def _make_fake_dataset(base: Path, name: str, valid: bool = True) -> Path:
    """Crea un fake dataset dir con struttura minima."""
    ds_dir = base / name
    ds_dir.mkdir(parents=True)

    if valid:
        # File obbligatori
        X = np.zeros((10, 5), dtype=np.float32)
        np.savez(ds_dir / "X_aparc_plv_theta.npz", X=X)
        np.save(ds_dir / "y.npy", np.array([0, 1] * 5))
        np.save(ds_dir / "groups.npy", np.arange(10))

    return ds_dir


@pytest.fixture()
def fake_features_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Crea data/features/ mock con 2 dataset validi + 1 invalido."""
    feat_base = tmp_path / "data" / "features"
    feat_base.mkdir(parents=True)

    _make_fake_dataset(feat_base, "fake_ds1", valid=True)
    _make_fake_dataset(feat_base, "fake_ds2", valid=True)
    _make_fake_dataset(feat_base, "fake_ds_empty", valid=False)  # dir vuota

    # Patch _ROOT in data_loader per puntare a tmp_path
    import dashboard.utils.data_loader as dl

    monkeypatch.setattr(dl, "_ROOT", tmp_path)

    return feat_base


def test_list_available_datasets_finds_valid(fake_features_dir: Path) -> None:
    """list_available_datasets deve trovare i 2 dataset validi, ignorare quello vuoto."""
    datasets = list_available_datasets()
    assert "fake_ds1" in datasets
    assert "fake_ds2" in datasets
    assert "fake_ds_empty" not in datasets, "Dataset senza X_*.npz non deve essere listato"


def test_validate_schema_valid_dataset(fake_features_dir: Path) -> None:
    """validate_dataset_schema su dataset valido deve restituire valid=True."""
    result = validate_dataset_schema("fake_ds1")
    assert result["valid"] is True
    assert len(result["errors"]) == 0
    assert result["n_combos"] == 1


def test_validate_schema_missing_y(fake_features_dir: Path) -> None:
    """validate_dataset_schema su dataset senza y.npy deve restituire valid=False."""
    ds_dir = fake_features_dir / "fake_ds1"
    (ds_dir / "y.npy").unlink()  # Rimuovi y.npy

    result = validate_dataset_schema("fake_ds1")
    assert result["valid"] is False
    assert any("y.npy" in e for e in result["errors"])


def test_validate_schema_warns_no_metadata(fake_features_dir: Path) -> None:
    """validate_dataset_schema deve segnalare warning se metadata.json è assente."""
    result = validate_dataset_schema("fake_ds2")
    assert any("metadata.json" in w for w in result["warnings"])
