"""Test STEP 3 parcellation — 4 atlanti su STC reale sub-05."""

from pathlib import Path

import numpy as np
import pytest

from parcellation.extract_label_tc import (
    extract_tc,
    extract_tc_batch,
    extract_tc_from_files,
    get_labels,
)

# Numero atteso di ROI per atlante (post-filter medial_wall / unknown)
_EXPECTED_NROI: dict[str, int] = {
    "aparc": 68,
    "destrieux": 148,
    "schaefer100": 100,
    "schaefer200": 200,
}

_FOUR_ATLASES = list(_EXPECTED_NROI.keys())


# ---------------------------------------------------------------------------
# get_labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("atlas", _FOUR_ATLASES)
def test_get_labels_count(atlas: str) -> None:
    """get_labels ritorna il numero corretto di ROI per ciascun atlante."""
    labels = get_labels(atlas)  # type: ignore[arg-type]
    assert len(labels) == _EXPECTED_NROI[atlas], (
        f"{atlas}: attese {_EXPECTED_NROI[atlas]} label, trovate {len(labels)}"
    )


@pytest.mark.parametrize("atlas", _FOUR_ATLASES)
def test_get_labels_no_excluded(atlas: str) -> None:
    """Nessuna label esclusa (unknown, medial_wall, background) presente."""
    labels = get_labels(atlas)  # type: ignore[arg-type]
    bad = [
        lbl.name for lbl in labels
        if any(kw in lbl.name.lower() for kw in ("unknown", "medial_wall", "background"))
    ]
    assert bad == [], f"{atlas}: label escluse ancora presenti: {bad}"


# ---------------------------------------------------------------------------
# extract_tc_from_files — test 4 atlanti su STC reale sub-05
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("atlas", _FOUR_ATLASES)
def test_extract_tc_shape(stc_path: Path, fwd_path: Path, atlas: str) -> None:
    """Shape output (n_roi, n_times) coerente con atlante e STC sub-05."""
    tc, names = extract_tc_from_files(
        stc_path,
        atlas,  # type: ignore[arg-type]
        fwd_path=fwd_path,
    )
    n_roi = _EXPECTED_NROI[atlas]
    assert tc.ndim == 2, f"tc deve essere 2D, trovato {tc.ndim}D"
    assert tc.shape[0] == n_roi, (
        f"{atlas}: attese {n_roi} righe, trovato {tc.shape[0]}"
    )
    assert tc.shape[1] > 0, "n_times deve essere > 0"
    assert len(names) == n_roi, "len(names) != n_roi"


@pytest.mark.parametrize("atlas", _FOUR_ATLASES)
def test_extract_tc_finite(stc_path: Path, fwd_path: Path, atlas: str) -> None:
    """Nessun NaN/Inf nel time course estratto."""
    tc, _ = extract_tc_from_files(
        stc_path,
        atlas,  # type: ignore[arg-type]
        fwd_path=fwd_path,
    )
    assert np.all(np.isfinite(tc)), f"{atlas}: NaN o Inf trovati nel TC"


@pytest.mark.parametrize("atlas", _FOUR_ATLASES)
def test_extract_tc_names_are_strings(stc_path: Path, fwd_path: Path, atlas: str) -> None:
    """I nomi label sono stringhe non vuote."""
    _, names = extract_tc_from_files(
        stc_path,
        atlas,  # type: ignore[arg-type]
        fwd_path=fwd_path,
    )
    assert all(isinstance(n, str) and len(n) > 0 for n in names), (
        f"{atlas}: nomi label non validi"
    )


# ---------------------------------------------------------------------------
# extract_tc_from_files — validazione signature
# ---------------------------------------------------------------------------


def test_extract_tc_from_files_requires_one_source(stc_path: Path) -> None:
    """Deve sollevare ValueError se nessun src/fwd fornito."""
    with pytest.raises(ValueError, match="fwd_path"):
        extract_tc_from_files(stc_path, "aparc")


def test_extract_tc_from_files_exclusive_sources(stc_path: Path, fwd_path: Path) -> None:
    """Deve sollevare ValueError se entrambi src e fwd forniti."""
    with pytest.raises(ValueError, match="mutuamente esclusivi"):
        extract_tc_from_files(
            stc_path,
            "aparc",
            fwd_path=fwd_path,
            src_path=fwd_path,  # volutamente sbagliato — solo per il test
        )


# ---------------------------------------------------------------------------
# extract_tc_batch — test batch vs per-epoch consistency
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_stcs_and_src(inv_path, epochs_path):
    """3 STC sintetici (con dati reali inv+info) + SourceSpaces per test batch."""
    import mne
    import mne.minimum_norm
    import numpy as np

    inv = mne.minimum_norm.read_inverse_operator(str(inv_path), verbose=False)
    epochs_real = mne.read_epochs(str(epochs_path), preload=False, verbose=False)
    info = epochs_real.info
    tmin = epochs_real.tmin
    n_times = len(epochs_real.times)
    rng = np.random.default_rng(99)
    n_ep = 3
    data = rng.standard_normal((n_ep, info["nchan"], n_times)) * 1e-6
    events = np.column_stack([
        np.arange(n_ep) * (n_times + 10),
        np.zeros(n_ep, int),
        np.ones(n_ep, int),
    ])
    fake_epochs = mne.EpochsArray(data, info, events=events, tmin=tmin, verbose=False)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stcs = list(mne.minimum_norm.apply_inverse_epochs(
            fake_epochs, inv, lambda2=1.0 / 9.0, method="dSPM",
            pick_ori="normal", return_generator=False, verbose=False,
        ))
    return stcs, inv["src"]


def test_extract_tc_batch_shape(fake_stcs_and_src):
    """extract_tc_batch ritorna (n_stcs, n_labels, n_times) per aparc."""
    stcs, src = fake_stcs_and_src
    tc, names = extract_tc_batch(stcs, "aparc", src)
    assert tc.ndim == 3, f"atteso 3D, trovato {tc.ndim}D"
    assert tc.shape[0] == len(stcs), "prima dim deve essere n_stcs"
    assert tc.shape[1] == 68, f"aparc: attese 68 label, trovate {tc.shape[1]}"
    assert tc.shape[2] > 0, "n_times deve essere > 0"
    assert len(names) == 68


def test_extract_tc_batch_finite(fake_stcs_and_src):
    """Nessun NaN/Inf nel risultato batch."""
    stcs, src = fake_stcs_and_src
    tc, _ = extract_tc_batch(stcs, "aparc", src)
    assert np.all(np.isfinite(tc)), "extract_tc_batch: NaN/Inf trovati"


def test_extract_tc_batch_consistent_with_per_epoch(fake_stcs_and_src):
    """extract_tc_batch[i] == extract_tc(stcs[i]) per ogni epoch."""
    stcs, src = fake_stcs_and_src
    tc_batch, names_batch = extract_tc_batch(stcs, "aparc", src)
    for i, stc in enumerate(stcs):
        tc_single, names_single = extract_tc(stc, "aparc", src)
        np.testing.assert_allclose(
            tc_batch[i], tc_single, rtol=1e-5,
            err_msg=f"Epoch {i}: batch != per-epoch",
        )
    assert names_batch == names_single
