"""BAND-5-9-T04: edge case tests parcellation — 4 atlas × 4 mode × scenari degeneri.

Scenari coperti:
  - 4 atlas × 4 mode (sweep normale su dati reali, skip se file assenti)
  - n_times=1 STC → shape corretta, nessun NaN
  - zero data STC → output tutto zero, finito
  - pca_flip × n_times=1 × 4 atlas → PASS (MNE gestisce senza SVD)
  - empty label (allow_empty=True) → riga zero + RuntimeWarning
  - wrong subject → OSError informativo
  - get_labels atlas invalido → KeyError
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from parcellation.extract_label_tc import (  # noqa: E402
    extract_tc,
    extract_tc_from_files,
    get_labels,
)

_FOUR_ATLASES = ["aparc", "destrieux", "schaefer100", "schaefer200"]
_FOUR_MODES = ["mean", "mean_flip", "pca_flip", "max"]
_EXPECTED_NROI = {"aparc": 68, "destrieux": 148, "schaefer100": 100, "schaefer200": 200}


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def src_and_stc(fwd_path, stc_path):
    """Forward src + real STC sub-05 (skip se file assenti)."""
    fwd = mne.read_forward_solution(str(fwd_path), verbose=False)
    stc = mne.read_source_estimate(str(stc_path))
    return fwd["src"], stc


# ── get_labels edge cases (no STC needed) ─────────────────────────────────────


def test_get_labels_invalid_atlas() -> None:
    """get_labels con atlas inesistente solleva KeyError."""
    with pytest.raises(KeyError):
        get_labels("nonexistent_atlas")  # type: ignore[arg-type]


def test_get_labels_wrong_subject() -> None:
    """get_labels con subject non esistente solleva OSError (directory assente)."""
    with pytest.raises(OSError):
        get_labels("aparc", subject="sub-NONEXISTENT-999")


@pytest.mark.parametrize("atlas", _FOUR_ATLASES)
def test_get_labels_excludes_medial_wall(atlas: str) -> None:
    """Nessuna label medial_wall/unknown/background post-filtro per i 4 atlas."""
    labels = get_labels(atlas)  # type: ignore[arg-type]
    assert len(labels) == _EXPECTED_NROI[atlas]
    bad = [lbl.name for lbl in labels if any(kw in lbl.name.lower()
           for kw in ("unknown", "medial_wall", "background"))]
    assert bad == [], f"{atlas}: label escluse ancora presenti: {bad}"


# ── extract_tc_from_files input validation ────────────────────────────────────


def test_extract_tc_no_src_raises(stc_path) -> None:
    """Chiamata senza fwd_path e src_path → ValueError."""
    with pytest.raises(ValueError, match="esattamente uno"):
        extract_tc_from_files(stc_path, "aparc")


def test_extract_tc_both_src_raises(stc_path, fwd_path) -> None:
    """Chiamata con entrambi fwd_path e src_path → ValueError."""
    with pytest.raises(ValueError, match="mutuamente esclusivi"):
        extract_tc_from_files(stc_path, "aparc", fwd_path=fwd_path, src_path=fwd_path)


# ── sweep 4 atlas × 4 mode su STC reale ─────────────────────────────────────


@pytest.mark.parametrize("mode", _FOUR_MODES)
@pytest.mark.parametrize("atlas", _FOUR_ATLASES)
def test_extract_tc_4atlas_4mode(atlas: str, mode: str, src_and_stc) -> None:
    """4 atlas × 4 mode su STC reale sub-05: shape corretta, finito, no NaN."""
    src, stc = src_and_stc
    tc, names = extract_tc(stc, atlas, src, mode=mode)  # type: ignore[arg-type]
    assert tc.shape == (_EXPECTED_NROI[atlas], stc.data.shape[1]), (
        f"{atlas}/{mode}: shape {tc.shape} inatteso"
    )
    assert len(names) == _EXPECTED_NROI[atlas]
    assert np.all(np.isfinite(tc)), f"{atlas}/{mode}: NaN/Inf trovati"


# ── edge: n_times=1 ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", _FOUR_MODES)
@pytest.mark.parametrize("atlas", _FOUR_ATLASES)
def test_n_times_1_all_modes(atlas: str, mode: str, src_and_stc) -> None:
    """STC con n_times=1: shape (n_rois, 1) finita per tutti i mode × atlas.

    Verifica che pca_flip non crashi con 1 time point.
    """
    src, stc = src_and_stc
    stc_1t = copy.deepcopy(stc)
    stc_1t.data = stc.data[:, :1]
    tc, _ = extract_tc(stc_1t, atlas, src, mode=mode)  # type: ignore[arg-type]
    assert tc.shape == (_EXPECTED_NROI[atlas], 1), (
        f"{atlas}/{mode}: shape {tc.shape}, atteso ({_EXPECTED_NROI[atlas]}, 1)"
    )
    assert np.all(np.isfinite(tc)), f"{atlas}/{mode}: NaN con n_times=1"


# ── edge: zero data ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", _FOUR_MODES)
def test_zero_data_stc_output_finite(mode: str, src_and_stc) -> None:
    """STC con dati tutti zero: output finito (no NaN da 0/0)."""
    src, stc = src_and_stc
    stc_zero = copy.deepcopy(stc)
    stc_zero.data = np.zeros_like(stc.data)
    tc, _ = extract_tc(stc_zero, "aparc", src, mode=mode)
    assert np.all(np.isfinite(tc)), f"mode={mode}: NaN con dati zero"
    assert np.all(tc == 0.0), f"mode={mode}: output non-zero con STC zero"


# ── edge: label vuota (allow_empty=True) ─────────────────────────────────────


def test_empty_label_produces_zero_row(src_and_stc) -> None:
    """Label con 0 vertici (allow_empty=True): riga di output tutta zero + warning."""
    src, stc = src_and_stc
    labels = get_labels("aparc")
    empty_lbl = mne.Label(np.array([], dtype=int), hemi="lh", name="empty-test-edge")
    labels_with_empty = labels + [empty_lbl]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tc = mne.extract_label_time_course(
            stc, labels_with_empty, src, mode="mean", allow_empty=True, verbose=False
        )

    assert tc.shape[0] == len(labels_with_empty), "n_rows != n_labels"
    assert np.all(tc[-1] == 0.0), "Label vuota: riga non-zero"
    assert np.all(np.isfinite(tc)), "NaN trovati con label vuota"
    runtime_warns = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warns, "Label vuota: RuntimeWarning atteso da MNE assente"
