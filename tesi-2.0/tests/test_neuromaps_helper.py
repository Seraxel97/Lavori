"""Tests STEP 3b — neuromaps_helper: parcellazione annotazioni brain maps.

I test verificano:
- parcellazione numpy-based (_parcellate_annotation) con label sintetiche MNE
- fetch_annotation_for_atlas ritorna dict con array shape corretta (skip se neuromaps non disponibile)
- graceful fallback quando neuromaps non disponibile
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

mne = pytest.importorskip("mne")


# ── helpers sintetici ──────────────────────────────────────────────────────────


def _make_synthetic_labels(n_verts_per_hemi: int = 200, n_rois: int = 4) -> list:
    """Crea label MNE sintetiche con vertici casuali distribuiti tra lh e rh."""
    labels = []
    verts_per_roi = n_verts_per_hemi // n_rois
    for i in range(n_rois):
        hemi = "lh" if i < n_rois // 2 else "rh"
        start = (i % (n_rois // 2)) * verts_per_roi
        verts = np.arange(start, start + verts_per_roi, dtype=np.int32)
        lbl = mne.Label(verts, hemi=hemi, name=f"roi-{i:02d}")
        labels.append(lbl)
    return labels


def _make_annotation_tuple(n_verts: int = 200) -> tuple:
    """Crea tuple (lh_arr, rh_arr) di dati sintetici compatibili con _parcellate_annotation."""
    rng = np.random.default_rng(2)
    lh = rng.standard_normal(n_verts).astype(np.float32)
    rh = rng.standard_normal(n_verts).astype(np.float32)
    return lh, rh


# ── tests _parcellate_annotation ─────────────────────────────────────────────


def test_parcellate_annotation_shape(subjects_dir) -> None:
    """_parcellate_annotation ritorna array shape (n_rois,)."""
    from parcellation.neuromaps_helper import _parcellate_annotation

    labels = _make_synthetic_labels(200, 4)
    img = _make_annotation_tuple(200)
    result = _parcellate_annotation(img, labels)
    assert result.shape == (len(labels),), f"Expected ({len(labels)},), got {result.shape}"


def test_parcellate_annotation_finite(subjects_dir) -> None:
    """Nessun NaN per label con vertici validi."""
    from parcellation.neuromaps_helper import _parcellate_annotation

    labels = _make_synthetic_labels(200, 4)
    img = _make_annotation_tuple(200)
    result = _parcellate_annotation(img, labels)
    # Label con vertici nel range → no NaN
    valid = [lbl for lbl in labels if len(lbl.vertices) > 0]
    for i, lbl in enumerate(labels):
        if lbl in valid and np.max(lbl.vertices) < 200:
            assert np.isfinite(result[i]), f"ROI {lbl.name}: valore non finito {result[i]}"


def test_parcellate_annotation_values(subjects_dir) -> None:
    """Valore per ROI = media dei vertici della label."""
    from parcellation.neuromaps_helper import _parcellate_annotation

    n_verts = 100
    rng = np.random.default_rng(3)
    lh_arr = rng.standard_normal(n_verts)

    # Una sola label lh con vertici [0, 1, 2, 3, 4]
    verts = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    lbl = mne.Label(verts, hemi="lh", name="test-roi")
    img = (lh_arr, np.zeros(n_verts))

    result = _parcellate_annotation(img, [lbl])
    expected = float(np.mean(lh_arr[verts]))
    assert np.isclose(result[0], expected), f"Expected {expected:.4f}, got {result[0]:.4f}"


# ── tests fetch_annotation_for_atlas ─────────────────────────────────────────


def test_fetch_annotation_no_neuromaps_returns_empty() -> None:
    """Senza neuromaps, fetch_annotation_for_atlas ritorna dict vuoto con warning."""
    import parcellation.neuromaps_helper as nm_mod

    original = nm_mod._NEUROMAPS_AVAILABLE
    nm_mod._NEUROMAPS_AVAILABLE = False
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = nm_mod.fetch_annotation_for_atlas("aparc")
        assert result == {}
        assert any("neuromaps" in str(w.message).lower() for w in caught)
    finally:
        nm_mod._NEUROMAPS_AVAILABLE = original


@pytest.mark.skipif(
    not __import__(
        "parcellation.neuromaps_helper", fromlist=["_NEUROMAPS_AVAILABLE"]
    )._NEUROMAPS_AVAILABLE,
    reason="neuromaps non installato",
)
def test_fetch_annotation_for_atlas_aparc_structure(subjects_dir) -> None:
    """fetch_annotation_for_atlas ritorna dict con array float (se neuromaps disponibile)."""
    from parcellation.neuromaps_helper import fetch_annotation_for_atlas

    try:
        result = fetch_annotation_for_atlas("aparc")
    except Exception:
        pytest.skip("Download annotazione neuromaps fallito (rete non disponibile)")

    assert isinstance(result, dict)
    for key, arr in result.items():
        assert isinstance(arr, np.ndarray), f"{key}: atteso ndarray, got {type(arr)}"
        assert arr.ndim == 1, f"{key}: atteso 1D, got shape {arr.shape}"
        assert len(arr) > 0, f"{key}: array vuoto"
