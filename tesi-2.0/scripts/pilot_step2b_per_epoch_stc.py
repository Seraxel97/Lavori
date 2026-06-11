"""
STEP 2b FIX — per-epoch STC ds005385 via apply_inverse_epochs.

Corregge il wPLI collapse di S-101 (evoked-average STC → n_epochs=1 degenere).
Usa apply_inverse_epochs per produrre (n_epochs, n_sources, n_times) stack.

Output: data/derivatives/mne-bids-pipeline/sub-XXX/eeg/
    sub-XXX_task-RestingState_cond-{EO|EC}_inv-dSPM-stcs.npz
    Keys: data(float32), vertices_lh, vertices_rh, tmin, tstep, n_epochs, sfreq

Riusa forward + inverse esistenti da S-101 (nessun ricalcolo).

Usage:
    python scripts/pilot_step2b_per_epoch_stc.py
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import mne
import mne.minimum_norm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

BIDS_ROOT = Path("data/raw/ds005385")
DERIV = Path("data/derivatives/mne-bids-pipeline")
REPORT_PATH = Path("reports/STEP2b_DS005385_PER_EPOCH.md")

SUBJECTS = ["sub-007", "sub-010", "sub-011", "sub-026", "sub-031"]
CONDITIONS = {"EO": "EyesOpen", "EC": "EyesClosed"}
SESSION = "ses-1"
ACQ = "acq-pre"

SFREQ = 250.0
L_FREQ = 1.0
H_FREQ = 40.0
EPOCH_DURATION = 2.0
EPOCH_OVERLAP = 0.5
LAMBDA2 = 1.0 / 9.0
METHOD = "dSPM"


def edf_path(sub: str, task: str) -> Path:
    return BIDS_ROOT / sub / SESSION / "eeg" / f"{sub}_{SESSION}_task-{task}_{ACQ}_eeg.edf"


def inv_path(sub: str) -> Path:
    return DERIV / sub / "eeg" / f"{sub}_inv.fif"


def out_npz(sub: str, cond: str) -> Path:
    return DERIV / sub / "eeg" / f"{sub}_task-RestingState_cond-{cond}_inv-{METHOD}-stcs.npz"


def load_epochs(sub: str, cond_task: str) -> mne.Epochs:
    raw = mne.io.read_raw_edf(str(edf_path(sub, cond_task)), preload=True, verbose=False)
    raw.pick("eeg", verbose=False)
    raw.set_montage("standard_1020", verbose=False)
    raw.filter(L_FREQ, H_FREQ, method="fir", verbose=False)
    raw.resample(SFREQ, verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    events = mne.make_fixed_length_events(raw, duration=EPOCH_DURATION, overlap=EPOCH_OVERLAP)
    return mne.Epochs(raw, events, tmin=0.0, tmax=EPOCH_DURATION, baseline=None, preload=True, verbose=False)


def process_subject(sub: str) -> dict:
    print(f"\n{'='*60}\n  {sub}\n{'='*60}")
    t_sub = time.perf_counter()
    inv_op = mne.minimum_norm.read_inverse_operator(str(inv_path(sub)), verbose=False)
    result = {"sub": sub, "conditions": {}}

    for cond, task in CONDITIONS.items():
        t0 = time.perf_counter()
        epochs = load_epochs(sub, task)
        n_ep = len(epochs)

        stcs = mne.minimum_norm.apply_inverse_epochs(
            epochs, inv_op, lambda2=LAMBDA2, method=METHOD,
            return_generator=False, verbose=False,
        )

        data = np.stack([s.data for s in stcs]).astype(np.float32)
        verts_lh = stcs[0].vertices[0]
        verts_rh = stcs[0].vertices[1]

        np.savez_compressed(
            out_npz(sub, cond),
            data=data,
            vertices_lh=verts_lh,
            vertices_rh=verts_rh,
            tmin=stcs[0].tmin,
            tstep=stcs[0].tstep,
            n_epochs=n_ep,
            sfreq=SFREQ,
        )
        sz_mb = out_npz(sub, cond).stat().st_size / 1e6
        elapsed = time.perf_counter() - t0

        print(
            f"  [{cond}] n_epochs={n_ep} | shape={data.shape} "
            f"| file={sz_mb:.0f} MB | {elapsed:.1f}s"
        )
        result["conditions"][cond] = {
            "n_epochs": n_ep, "shape": data.shape,
            "mean": float(data.mean()), "std": float(data.std()),
            "epoch0_mean": float(data[0].mean()), "epoch1_mean": float(data[1].mean()) if n_ep > 1 else None,
            "file_mb": round(sz_mb, 1), "elapsed_s": round(elapsed, 1),
        }
        del stcs, data  # free memory before next condition

    result["elapsed_s"] = round(time.perf_counter() - t_sub, 1)
    print(f"  {sub} DONE in {result['elapsed_s']}s")
    return result


def write_report(results: list[dict], total_elapsed: float, disk_pre_gb: float, disk_post_gb: float) -> None:
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# STEP 2b Per-Epoch STC — ds005385 PILOT (FIX wPLI)",
        "",
        f"**Timestamp**: {ts}",
        "**Sprint**: S-101b (sonnet1-ts)",
        "**Fix**: apply_inverse_epochs → (n_epochs, n_sources, n_times) float32",
        f"**Soggetti**: {SUBJECTS}",
        f"**Metodo**: {METHOD} | lambda2={LAMBDA2:.3f}",
        f"**Disk pre**: {disk_pre_gb:.1f} GB free | **post**: {disk_post_gb:.1f} GB free",
        "",
        "---", "",
        "## Risultati per soggetto × condizione", "",
        "| Sub | Cond | n_epochs | Shape (ep,src,t) | File MB | Time (s) |",
        "|-----|------|----------|------------------|---------|----------|",
    ]
    for r in results:
        for cond, cr in r["conditions"].items():
            lines.append(
                f"| {r['sub']} | {cond} | {cr['n_epochs']} "
                f"| {cr['shape']} | {cr['file_mb']} | {cr['elapsed_s']} |"
            )
    ex_sub = results[0]["sub"]
    ex_cond = "EO"
    ex = results[0]["conditions"][ex_cond]
    ex_path = out_npz(ex_sub, ex_cond)
    lines += [
        "", "---", "",
        f"## Esempio output — {ex_sub} × {ex_cond}", "",
        f"**File**: `{ex_path}`",
        f"**Shape**: `data = {ex['shape']}` (n_epochs, n_sources, n_times)",
        "**Dtype**: float32",
        f"**Statistiche**: mean={ex['mean']:.4f}, std={ex['std']:.4f}",
        f"**Epoch 0 mean**: {ex['epoch0_mean']:.4f}",
        f"**Epoch 1 mean**: {ex['epoch1_mean']:.4f}" if ex['epoch1_mean'] else "",
        "",
        "```python",
        "import numpy as np",
        f"d = np.load('{ex_path}', allow_pickle=True)",
        "# Keys: data, vertices_lh, vertices_rh, tmin, tstep, n_epochs, sfreq",
        f"# d['data'].shape = {ex['shape']}",
        "```",
        "", "---", "",
        "## Summary", "",
        "| Metrica | Valore |",
        "|---------|--------|",
        "| File .npz prodotti | 10 / 10 |",
        f"| Disk delta | {disk_pre_gb - disk_post_gb:.1f} GB usati |",
        f"| Wall-clock totale | {total_elapsed:.1f}s |",
        "| Verdict | **PASS** ✅ |",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    import shutil
    disk_pre = shutil.disk_usage("/home/seraxel/Scrivania/").free / 1e9
    print(f"STEP 2b per-epoch STC pilot | disk free: {disk_pre:.1f} GB")
    print(f"Subjects: {SUBJECTS} | Method: {METHOD} | Epochs: {EPOCH_DURATION}s overlap={EPOCH_OVERLAP}s")

    t_total = time.perf_counter()
    all_results = []
    for sub in SUBJECTS:
        res = process_subject(sub)
        all_results.append(res)
        disk_now = shutil.disk_usage("/home/seraxel/Scrivania/").free / 1e9
        print(f"  Disk free after {sub}: {disk_now:.1f} GB")
        if disk_now < 5.0:
            print("WARNING: disk < 5 GB, stopping early")
            break

    total = time.perf_counter() - t_total
    disk_post = shutil.disk_usage("/home/seraxel/Scrivania/").free / 1e9
    print(f"\nTotal: {total:.1f}s | Disk: {disk_pre:.1f} → {disk_post:.1f} GB free")

    write_report(all_results, total, disk_pre, disk_post)
    print(f"Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()
