"""
Pipeline ds004504 adapter — step 2.1 smoke test su sub-001.

Adattamento di run_pipeline_n30.py per ds004504 (.set EEGLAB, no ses-*, no EyesOpen).

Usage:
    python scripts/run_pipeline_ds004504.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import mne
import mne.minimum_norm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

# ── Costanti ──────────────────────────────────────────────────────────────────

BIDS_ROOT = Path("data/raw/ds004504")
DERIV = Path("data/derivatives/mne-bids-pipeline-ds004504")
LABEL_TS_DIR = Path("data/label_ts/ds004504")
CONN_DIR = Path("data/connectivity/ds004504")
FEAT_DIR = Path("data/features/ds004504")
RESULTS_DIR = Path("data/results/ds004504")
HEARTBEAT_PATH = Path(".planning/heartbeats_tesi/haiku-tesi-lemon-2.json")
SUBJECTS_DIR = Path("/home/seraxel/mne_data/MNE-fsaverage-data")

SESSION = None
CONDITIONS = {"EC": "eyesclosed"}

SFREQ_TARGET = 250.0
FILTER_LO = 1.0
FILTER_HI = 40.0
EPOCH_DURATION = 2.0
EPOCH_OVERLAP = 0.5
LAMBDA2 = 1.0 / 9.0
METHOD = "dSPM"
LOOSE = 0.2
DEPTH = 0.8
MINDIST = 5.0

ATLASES = ["aparc"]
ALL_METRICS = ["wpli", "coh", "plv", "imcoh"]
ALL_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

SUBJECTS_DS004504 = [f"sub-{i:03d}" for i in range(1, 51)]

for d in [LABEL_TS_DIR, CONN_DIR, FEAT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Heartbeat ─────────────────────────────────────────────────────────────────

def write_hb(step: str, progress: str, extra: dict | None = None) -> None:
    data: dict = {
        "worker": "haiku-tesi-lemon-2",
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "in_progress",
        "current_sprint": "STEP_2.1_preprocessing_adapter",
        "current_step": step,
        "progress": progress,
    }
    if extra:
        data.update(extra)
    HEARTBEAT_PATH.write_text(json.dumps(data, indent=2))


# ── STEP 2 — Forward + inverse ────────────────────────────────────────────────

def _set_path(sub: str, cond_task: str) -> Path:
    return BIDS_ROOT / sub / "eeg" / f"{sub}_task-{cond_task}_eeg.set"


def _out_dir(sub: str) -> Path:
    d = DERIV / sub / "eeg"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_raw(set_path: Path) -> mne.io.Raw:
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    raw.pick("eeg", verbose=False)
    raw.set_montage("standard_1020", match_case=False, verbose=False)
    raw.filter(FILTER_LO, FILTER_HI, method="fir", verbose=False)
    raw.resample(SFREQ_TARGET, verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    return raw


def step2_fwd_inv(sub: str) -> tuple[mne.Forward, mne.minimum_norm.InverseOperator]:
    out = _out_dir(sub)
    fwd_p = out / f"{sub}_fwd.fif"
    inv_p = out / f"{sub}_inv.fif"

    if fwd_p.exists() and inv_p.exists():
        fwd = mne.read_forward_solution(str(fwd_p), verbose=False)
        inv = mne.minimum_norm.read_inverse_operator(str(inv_p), verbose=False)
        print(f"  [step2] {sub}: fwd+inv cached")
        return fwd, inv

    first_set = _set_path(sub, "eyesclosed")
    if not first_set.exists():
        raise FileNotFoundError(f"File .set non trovato: {first_set}")
    raw = _load_raw(first_set)
    info = raw.info
    del raw

    if not fwd_p.exists():
        print(f"  [step2] {sub}: computing forward…")
        src = mne.read_source_spaces(
            str(SUBJECTS_DIR / "fsaverage" / "bem" / "fsaverage-oct6-src.fif"), verbose=False)
        bem = mne.read_bem_solution(
            str(SUBJECTS_DIR / "fsaverage" / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"), verbose=False)
        trans = str(SUBJECTS_DIR / "fsaverage" / "bem" / "fsaverage-trans.fif")
        fwd = mne.make_forward_solution(
            info, trans=trans, src=src, bem=bem,
            eeg=True, meg=False, mindist=MINDIST, verbose=False)
        mne.write_forward_solution(str(fwd_p), fwd, overwrite=True, verbose=False)
        print(f"    → {fwd_p.name}")
    else:
        fwd = mne.read_forward_solution(str(fwd_p), verbose=False)

    if not inv_p.exists():
        noise_cov = mne.make_ad_hoc_cov(info, verbose=False)
        inv = mne.minimum_norm.make_inverse_operator(
            info, fwd, noise_cov, loose=LOOSE, depth=DEPTH, verbose=False)
        mne.minimum_norm.write_inverse_operator(str(inv_p), inv, overwrite=True, verbose=False)
        print(f"    → {inv_p.name}")
    else:
        inv = mne.minimum_norm.read_inverse_operator(str(inv_p), verbose=False)

    return fwd, inv


# ── STEP 2b — Per-epoch STCs ──────────────────────────────────────────────────

def _stcs_npz(sub: str, cond: str) -> Path:
    return DERIV / sub / "eeg" / f"{sub}_task-RestingState_cond-{cond}_inv-{METHOD}-stcs.npz"


def step2b_stcs(sub: str, inv_op: mne.minimum_norm.InverseOperator) -> dict:
    out: dict = {}
    cond = "EC"
    task = CONDITIONS[cond]
    npz = _stcs_npz(sub, cond)
    if npz.exists():
        print(f"  [step2b] {sub}: stcs.npz cached")
        d = np.load(npz, allow_pickle=True)
        out[cond] = {"n_epochs": int(d["n_epochs"]), "file_mb": npz.stat().st_size / 1e6}
        return out

    set_p = _set_path(sub, task)
    if not set_p.exists():
        print(f"  [step2b] {sub}: .set missing → skip")
        out[cond] = {"n_epochs": 0, "file_mb": 0, "status": "missing_set"}
        return out

    raw = _load_raw(set_p)
    events = mne.make_fixed_length_events(raw, duration=EPOCH_DURATION, overlap=EPOCH_OVERLAP)
    epochs = mne.Epochs(raw, events, tmin=0.0, tmax=EPOCH_DURATION,
                        baseline=None, preload=True, verbose=False)
    del raw

    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs, inv_op, lambda2=LAMBDA2, method=METHOD,
        return_generator=False, verbose=False)
    del epochs

    data = np.stack([s.data for s in stcs]).astype(np.float32)
    vl, vr = stcs[0].vertices[0], stcs[0].vertices[1]
    np.savez_compressed(npz, data=data, vertices_lh=vl, vertices_rh=vr,
                        tmin=stcs[0].tmin, tstep=stcs[0].tstep,
                        n_epochs=len(stcs), sfreq=SFREQ_TARGET)
    del stcs, data

    mb = npz.stat().st_size / 1e6
    print(f"  [step2b] {sub}: {mb:.0f} MB")
    out[cond] = {"n_epochs": int(np.load(npz, allow_pickle=True)["n_epochs"]), "file_mb": mb}
    return out


# ── STEP 3b — Parcellazione ───────────────────────────────────────────────────

def _label_ts_npz(sub: str, cond: str) -> Path:
    return LABEL_TS_DIR / f"{sub}_cond-{cond}_aparc_label_ts.npz"


def step3b_parc(sub: str, fwd: mne.Forward) -> dict:
    from parcellation.extract_label_tc import get_labels
    src = fwd["src"]
    out: dict = {}

    atlas = "aparc"
    labels = get_labels(atlas, subject="fsaverage")
    label_names = [lbl.name for lbl in labels]

    cond = "EC"
    out_p = _label_ts_npz(sub, cond)
    if out_p.exists():
        print(f"  [step3b] {sub}: cached")
        out[cond] = {"status": "cached"}
        return out

    stcs_p = _stcs_npz(sub, cond)
    if not stcs_p.exists():
        print(f"  [step3b] {sub}: stcs.npz missing → skip")
        out[cond] = {"status": "missing_stcs"}
        return out

    d = np.load(stcs_p, allow_pickle=True)
    data = d["data"]
    stcs_list = [
        mne.SourceEstimate(
            data=data[i].astype(np.float64),
            vertices=[d["vertices_lh"], d["vertices_rh"]],
            tmin=float(d["tmin"]), tstep=float(d["tstep"]),
            subject="fsaverage")
        for i in range(data.shape[0])]
    del data

    tcs = mne.extract_label_time_course(
        stcs_list, labels, src, mode="mean_flip",
        allow_empty=True, verbose=False)
    del stcs_list

    label_tc = np.stack(tcs).astype(np.float32)
    del tcs

    np.savez_compressed(out_p, label_tc=label_tc,
                        label_names=np.array(label_names),
                        subject=sub, condition=cond, atlas=atlas,
                        mode="mean_flip", sfreq=SFREQ_TARGET)
    del label_tc

    print(f"  [step3b] {sub}: OK")
    out[cond] = {"status": "done"}
    return out


# ── STEP 4b — FC per-epoch ───────────────────────────────────────────────────

def _conn_npz(sub: str, cond: str, metric: str, band: str) -> Path:
    return CONN_DIR / f"{sub}_atlas-aparc_cond-{cond}_metric-{metric}_band-{band}_per-epoch.npz"


def step4b_fc(subjects: list[str]) -> int:
    from connectivity.fc_dispatcher import compute_fc
    n_done = 0
    for sub in subjects:
        cond = "EC"
        label_ts_p = _label_ts_npz(sub, cond)
        if not label_ts_p.exists():
            print(f"  [step4b] {sub}: label_ts missing → skip")
            continue

        d_lt = np.load(label_ts_p, allow_pickle=True)
        label_tc = d_lt["label_tc"].astype(np.float64)

        for metric in ALL_METRICS:
            need_bands = {
                b: hz for b, hz in ALL_BANDS.items()
                if not _conn_npz(sub, cond, metric, b).exists()
            }
            for b in ALL_BANDS:
                if b not in need_bands:
                    n_done += 1

            if not need_bands:
                continue

            fc_dict = compute_fc(label_tc, SFREQ_TARGET, metric,
                                 bands=need_bands, mode="multitaper")

            for b, mat in fc_dict.items():
                out_p = _conn_npz(sub, cond, metric, b)
                np.savez_compressed(out_p, fc_matrix=mat,
                                    subject=sub, condition=cond, atlas="aparc",
                                    metric=metric, band=b, sfreq=SFREQ_TARGET)
                n_done += 1
                print(f"  [step4b] {sub} {metric} {b}: OK")

        del label_tc
    return n_done


# ── STEP 5 — Feature extraction ───────────────────────────────────────────────

def step5_features(subjects: list[str], force: bool = False) -> list[dict]:
    cond = "EC"
    y, groups = [], []
    for i, sub in enumerate(subjects):
        y.append(0)  # EC only → single label
        groups.append(i)
    np.save(FEAT_DIR / "y.npy", np.array(y, dtype=np.int32))
    np.save(FEAT_DIR / "groups.npy", np.array(groups, dtype=np.int32))

    results = []
    for metric in ALL_METRICS:
        for band in ALL_BANDS:
            out_p = FEAT_DIR / f"X_aparc_{metric}_{band}.npz"
            if not force and out_p.exists():
                d = np.load(out_p, allow_pickle=True)
                if d["X"].shape[0] == len(subjects):
                    print(f"  [step5] {metric} {band}: SKIP (N={len(subjects)} ok)")
                    results.append({"atlas": "aparc", "metric": metric, "band": band,
                                    "shape": list(d["X"].shape), "skipped": True})
                    continue

            rows = []
            ok = True
            for sub in subjects:
                fc_p = _conn_npz(sub, cond, metric, band)
                if not fc_p.exists():
                    print(f"  [step5] MISSING: {fc_p.name}")
                    ok = False
                    break
                d = np.load(fc_p, allow_pickle=True)
                mat = d["fc_matrix"].astype(np.float64)
                n = mat.shape[0]
                idx = np.triu_indices(n, k=1)
                rows.append(mat[idx])

            if not ok:
                results.append({"atlas": "aparc", "metric": metric, "band": band,
                                "error": "missing_fc"})
                continue

            X = np.stack(rows)
            np.savez_compressed(out_p, X=X)
            print(f"  [step5] {metric} {band}: shape={X.shape}")
            results.append({"atlas": "aparc", "metric": metric, "band": band,
                            "shape": list(X.shape)})

    meta = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "subjects": subjects, "conditions": ["EC"],
        "atlases": ["aparc"], "metrics": ALL_METRICS, "bands": list(ALL_BANDS.keys()),
        "n_subjects": len(subjects), "n_samples": len(subjects),
    }
    (FEAT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    return results


def run_fc_features() -> None:
    """Step 4b + 5: FC + feature aggregation per ds004504."""
    subjects_done = [
        sub for sub in SUBJECTS_DS004504
        if (LABEL_TS_DIR / f"{sub}_cond-EC_aparc_label_ts.npz").exists()
    ]
    print(f"[ds004504 step4b] {len(subjects_done)} sub con label_ts disponibili")
    write_hb("step4b_start", f"{len(subjects_done)} subs")
    step4b_fc(subjects_done)
    write_hb("step4b_done", "FC computed")
    step5_features(subjects_done)
    write_hb("step5_done", f"Features extracted for {len(subjects_done)} subs")


# ── Main — Step 2.2 full pipeline (fwd+inv+stcs+parc) ───────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-to-fc", action="store_true")
    _args = parser.parse_args()

    subjects = SUBJECTS_DS004504
    print(f"[ds004504] Processing {len(subjects)} subjects — step2+2b+3b")

    for i, sub in enumerate(subjects):
        print(f"\n[{i+1}/{len(subjects)}] {sub}")
        set_p = _set_path(sub, "eyesclosed")
        if not set_p.exists():
            print(f"  SKIP: {set_p} not found")
            continue
        try:
            fwd, inv = step2_fwd_inv(sub)
            step2b_stcs(sub, inv)
            step3b_parc(sub, fwd)
        except Exception as e:
            print(f"  ERROR {sub}: {e}")
            continue
        stcs_npz = _stcs_npz(sub, "EC")
        if stcs_npz.exists():
            stcs_npz.unlink()
            print("  [cleanup] stcs.npz rimosso — disco preservato")

    print("\n✅ Pipeline step 2.2 complete")


if __name__ == "__main__":
    main()
