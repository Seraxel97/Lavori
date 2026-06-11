"""Tests STEP 2b — apply_inverse_epochs_rs: N=10 epoch sintetici.

Strategia: carica inv_op e epochs.info reali (skip se mancanti),
crea dati sintetici via EpochsArray compatibile con inv_op.
"""

from __future__ import annotations

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from source_reconstruction.apply_inverse_epochs_rs import (  # noqa: E402
    apply_inverse_epochs_rs,
    save_stcs,
)

_INV_PATH = pytest.importorskip  # unused sentinel


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def inv_op_and_info(inv_path, epochs_path):
    """Inverse operator + info reale sub-05, per costruire epoch sintetici."""
    inv = mne.minimum_norm.read_inverse_operator(str(inv_path), verbose=False)
    epochs_real = mne.read_epochs(str(epochs_path), preload=False, verbose=False)
    return inv, epochs_real.info, epochs_real.tmin, len(epochs_real.times)


@pytest.fixture(scope="module")
def fake_epochs(inv_op_and_info):
    """10 epoch sintetici (dati random) con info compatibile con inv_op."""
    inv, info, tmin, n_times = inv_op_and_info
    rng = np.random.default_rng(0)
    n_ep = 10
    data = rng.standard_normal((n_ep, info["nchan"], n_times)) * 1e-6
    events = np.column_stack(
        [
            np.arange(n_ep) * (n_times + 10),
            np.zeros(n_ep, dtype=int),
            np.ones(n_ep, dtype=int),
        ]
    )
    return mne.EpochsArray(data, info, events=events, tmin=tmin, verbose=False)


@pytest.fixture(scope="module")
def inv_op(inv_op_and_info):
    return inv_op_and_info[0]


# ── tests ─────────────────────────────────────────────────────────────────────


def test_apply_inverse_epochs_rs_returns_list(fake_epochs, inv_op):
    """apply_inverse_epochs_rs ritorna una lista di SourceEstimate."""
    stcs = apply_inverse_epochs_rs(fake_epochs, inv_op)
    assert isinstance(stcs, list)
    assert len(stcs) == len(fake_epochs)


def test_apply_inverse_epochs_rs_stc_shape(fake_epochs, inv_op):
    """Ogni STC ha shape (n_sources, n_times) coerente tra epoch."""
    stcs = apply_inverse_epochs_rs(fake_epochs, inv_op)
    shapes = [s.data.shape for s in stcs]
    assert all(sh == shapes[0] for sh in shapes), f"Shapes inconsistenti: {shapes[:3]}"
    n_sources, n_times = shapes[0]
    assert n_sources > 0
    assert n_times == len(fake_epochs.times)


def test_apply_inverse_epochs_rs_finite(fake_epochs, inv_op):
    """Nessun NaN/Inf nel risultato STC."""
    stcs = apply_inverse_epochs_rs(fake_epochs, inv_op)
    for idx, stc in enumerate(stcs):
        assert np.all(np.isfinite(stc.data)), f"Epoch {idx}: NaN/Inf trovati"


def test_apply_inverse_epochs_rs_magnitude(fake_epochs, inv_op):
    """pick_ori=None ritorna magnitudine (valori non negativi sul dato medio)."""
    stcs = apply_inverse_epochs_rs(fake_epochs, inv_op, pick_ori=None)
    for stc in stcs:
        # dSPM magnitude: valori >= 0
        assert np.all(stc.data >= 0), "Magnitudine ha valori negativi"


def test_save_stcs_roundtrip(fake_epochs, inv_op, tmp_path):
    """save_stcs crea file leggibili da MNE con shape corretta."""
    stcs = apply_inverse_epochs_rs(fake_epochs[:3], inv_op)
    stems = save_stcs(stcs, tmp_path, prefix="test_sub-05")
    assert len(stems) == 3
    for stem in stems:
        lh = stem.parent / (stem.name + "-lh.stc")
        assert lh.exists(), f"File non trovato: {lh}"
        stc_loaded = mne.read_source_estimate(str(stem))
        assert stc_loaded.data.shape == stcs[0].data.shape
