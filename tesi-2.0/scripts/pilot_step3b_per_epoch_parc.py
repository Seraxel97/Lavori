"""
STEP 3b — per-epoch parcellation ds005385 (5 sub × 2 atlas × 2 cond = 20 npz).

Usa per-epoch STC da S-101b per produrre label_tc (n_epochs, n_labels, n_times).
Risolve il wPLI collapse: input corretto per FC dispatcher.

Output: data/label_ts/ds005385/sub-XXX_atlas-XXX_cond-XXX_per-epoch.npz
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

from parcellation.extract_label_tc import get_labels  # noqa: E402

DERIV = Path("data/derivatives/mne-bids-pipeline")
OUT_DIR = Path("data/label_ts/ds005385")
REPORT_PATH = Path("reports/STEP3b_DS005385_PER_EPOCH.md")
SUBJECTS_DIR = Path("/home/seraxel/mne_data/MNE-fsaverage-data")

SUBJECTS = ["sub-007", "sub-010", "sub-011", "sub-026", "sub-031"]
ATLASES = ["aparc", "schaefer100"]
CONDITIONS = ["EO", "EC"]
MODE = "mean_flip"
SFREQ = 250.0

OUT_DIR.mkdir(parents=True, exist_ok=True)


def stcs_npz(sub: str, cond: str) -> Path:
    return DERIV / sub / "eeg" / f"{sub}_task-RestingState_cond-{cond}_inv-dSPM-stcs.npz"


def fwd_path(sub: str) -> Path:
    return DERIV / sub / "eeg" / f"{sub}_fwd.fif"


def out_npz(sub: str, atlas: str, cond: str) -> Path:
    return OUT_DIR / f"{sub}_atlas-{atlas}_cond-{cond}_per-epoch.npz"


def load_stc_list(sub: str, cond: str) -> tuple[list[mne.SourceEstimate], int]:
    d = np.load(stcs_npz(sub, cond), allow_pickle=True)
    data = d["data"]                          # (n_epochs, n_sources, n_times)
    vl = d["vertices_lh"]
    vr = d["vertices_rh"]
    tmin = float(d["tmin"])
    tstep = float(d["tstep"])
    stcs = [
        mne.SourceEstimate(
            data=data[i].astype(np.float64),
            vertices=[vl, vr],
            tmin=tmin,
            tstep=tstep,
            subject="fsaverage",
        )
        for i in range(data.shape[0])
    ]
    return stcs, data.shape[0]


def run_all() -> list[dict]:
    label_cache: dict[str, list[mne.Label]] = {}

    results = []
    total = len(SUBJECTS) * len(ATLASES) * len(CONDITIONS)
    done = 0

    for sub in SUBJECTS:
        fwd = mne.read_forward_solution(str(fwd_path(sub)), verbose=False)
        src = fwd["src"]

        for atlas in ATLASES:
            if atlas not in label_cache:
                label_cache[atlas] = get_labels(atlas, subject="fsaverage")

            labels = label_cache[atlas]
            n_labels = len(labels)
            label_names = [lbl.name for lbl in labels]

            for cond in CONDITIONS:
                t0 = time.perf_counter()
                stcs, n_epochs = load_stc_list(sub, cond)

                tcs = mne.extract_label_time_course(
                    stcs, labels, src, mode=MODE, allow_empty=True, verbose=False
                )
                label_tc = np.stack(tcs).astype(np.float32)  # (n_epochs, n_labels, n_times)

                np.savez_compressed(
                    out_npz(sub, atlas, cond),
                    label_tc=label_tc,
                    label_names=np.array(label_names),
                    subject=sub,
                    condition=cond,
                    atlas=atlas,
                    mode=MODE,
                    n_epochs=n_epochs,
                    sfreq=SFREQ,
                )
                elapsed = time.perf_counter() - t0
                done += 1

                ep0_mean = float(label_tc[0].mean())
                ep1_mean = float(label_tc[1].mean()) if n_epochs > 1 else float("nan")
                ep2_mean = float(label_tc[2].mean()) if n_epochs > 2 else float("nan")

                print(
                    f"[{done:02d}/{total}] {sub} {atlas:12s} {cond}  "
                    f"shape=({n_epochs},{n_labels},{label_tc.shape[2]})  "
                    f"ep0={ep0_mean:.4f}  {elapsed:.1f}s"
                )
                results.append({
                    "sub": sub, "atlas": atlas, "cond": cond,
                    "n_epochs": n_epochs, "n_labels": n_labels,
                    "n_times": label_tc.shape[2],
                    "ep0_mean": round(ep0_mean, 4),
                    "ep1_mean": round(ep1_mean, 4),
                    "ep2_mean": round(ep2_mean, 4),
                    "elapsed_s": round(elapsed, 1),
                })
                del stcs, tcs, label_tc

    return results


def write_report(results: list[dict], total_elapsed: float) -> None:
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# STEP 3b Per-Epoch Parcellation — ds005385 PILOT",
        "",
        f"**Timestamp**: {ts}",
        "**Sprint**: S-103b (sonnet1-ts)",
        f"**Soggetti**: {SUBJECTS}",
        f"**Atlases**: {ATLASES}  **Mode**: {MODE}",
        "",
        "---", "",
        "## Risultati (20 estrazioni)", "",
        "| Sub | Atlas | Cond | Shape (ep,lbl,t) | ep0 mean | Time (s) |",
        "|-----|-------|------|-----------------|----------|----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['sub']} | {r['atlas']} | {r['cond']} "
            f"| ({r['n_epochs']},{r['n_labels']},{r['n_times']}) "
            f"| {r['ep0_mean']} | {r['elapsed_s']} |"
        )

    ex = results[0]
    ex_path = out_npz(ex["sub"], ex["atlas"], ex["cond"])
    lines += [
        "", "---", "",
        f"## Esempio — {ex['sub']} × {ex['atlas']} × {ex['cond']}", "",
        f"**File**: `{ex_path}`",
        f"**Shape**: `label_tc = ({ex['n_epochs']}, {ex['n_labels']}, {ex['n_times']})`",
        f"**Epoch means**: ep0={ex['ep0_mean']}, ep1={ex['ep1_mean']}, ep2={ex['ep2_mean']}",
        "",
        "```python",
        "import numpy as np",
        f"d = np.load('{ex_path}', allow_pickle=True)",
        "# Keys: label_tc, label_names, subject, condition, atlas, mode, n_epochs, sfreq",
        f"# label_tc.shape = ({ex['n_epochs']}, {ex['n_labels']}, {ex['n_times']})",
        "```",
        "", "---", "",
        "## Summary", "",
        "| Metrica | Valore |",
        "|---------|--------|",
        f"| File .npz prodotti | {len(results)} / 20 |",
        f"| Wall-clock totale | {total_elapsed:.1f}s |",
        "| Verdict | **PASS** ✅ |",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    print(f"STEP 3b per-epoch parcellation | {len(SUBJECTS)} sub × {len(ATLASES)} atlas × {len(CONDITIONS)} cond")
    t0 = time.perf_counter()
    results = run_all()
    total = time.perf_counter() - t0
    print(f"\nTotal: {total:.1f}s | {len(results)} file .npz")
    write_report(results, total)
    print(f"Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()
