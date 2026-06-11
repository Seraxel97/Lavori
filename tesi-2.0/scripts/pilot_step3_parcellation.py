"""
STEP 3 pilot — parcellation ds005385 (5 sub × 2 atlas × 2 cond = 20 .npz).

Usa extract_tc_from_files su STC prodotti da S-101.
Output: data/label_ts/ds005385/sub-XXX_atlas-XXX_cond-XXX.npz

Usage:
    python scripts/pilot_step3_parcellation.py
"""

from __future__ import annotations

import sys
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

warnings.filterwarnings("ignore")

DERIV = Path("data/derivatives/mne-bids-pipeline")
OUT_DIR = Path("data/label_ts/ds005385")
REPORT_PATH = Path("reports/STEP3_DS005385_PILOT.md")

PILOT_SUBJECTS = ["sub-007", "sub-010", "sub-011", "sub-026", "sub-031"]
ATLASES = ["aparc", "schaefer100"]
CONDITIONS = ["EO", "EC"]
TASK = "RestingState"
METHOD = "dSPM"
MODE = "mean_flip"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def stc_stem(sub: str, cond: str) -> Path:
    return DERIV / sub / "eeg" / f"{sub}_task-{TASK}_cond-{cond}_inv-{METHOD}"


def fwd_path(sub: str) -> Path:
    return DERIV / sub / "eeg" / f"{sub}_fwd.fif"


def npz_path(sub: str, atlas: str, cond: str) -> Path:
    return OUT_DIR / f"{sub}_atlas-{atlas}_cond-{cond}.npz"


def run_extraction() -> list[dict]:
    from parcellation.extract_label_tc import extract_tc_from_files

    results = []
    total = len(PILOT_SUBJECTS) * len(ATLASES) * len(CONDITIONS)
    done = 0

    for sub in PILOT_SUBJECTS:
        fwd = fwd_path(sub)
        for atlas in ATLASES:
            for cond in CONDITIONS:
                stem = stc_stem(sub, cond)
                out = npz_path(sub, atlas, cond)
                t0 = time.perf_counter()

                tc, names = extract_tc_from_files(
                    stem, atlas, fwd_path=fwd, mode=MODE
                )
                np.savez_compressed(
                    out,
                    label_tc=tc,
                    label_names=np.array(names),
                    atlas=atlas,
                    subject=sub,
                    condition=cond,
                    task=TASK,
                    mode=MODE,
                )
                elapsed = time.perf_counter() - t0
                done += 1
                print(
                    f"[{done:02d}/{total}] {sub} {atlas:12s} {cond}  "
                    f"shape=({tc.shape[0]},{tc.shape[1]})  {elapsed:.2f}s"
                )
                results.append({
                    "sub": sub, "atlas": atlas, "cond": cond,
                    "n_labels": tc.shape[0], "n_times": tc.shape[1],
                    "label_names_5": names[:5], "elapsed_s": round(elapsed, 3),
                    "out": str(out),
                })

    return results


def write_report(results: list[dict], total_elapsed: float) -> None:
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# STEP 3 Parcellation — ds005385 PILOT",
        "",
        f"**Timestamp**: {ts}",
        "**Sprint**: S-103 (sonnet1-ts)",
        f"**Soggetti**: {PILOT_SUBJECTS}",
        f"**Atlases**: {ATLASES}",
        f"**Condizioni**: {CONDITIONS}",
        f"**Mode**: `{MODE}`",
        "",
        "---",
        "",
        "## Risultati (sub × atlas × cond)",
        "",
        "| Sub | Atlas | Cond | n_labels | n_times | Wall (s) |",
        "|-----|-------|------|----------|---------|----------|",
    ]
    all_ok = True
    for r in results:
        lines.append(
            f"| {r['sub']} | {r['atlas']} | {r['cond']} "
            f"| {r['n_labels']} | {r['n_times']} | {r['elapsed_s']} |"
        )
        if r["n_labels"] == 0:
            all_ok = False

    # Example: first result
    ex = results[0]
    lines += [
        "",
        "---",
        "",
        f"## Esempio output — {ex['sub']} × {ex['atlas']} × {ex['cond']}",
        "",
        f"**File**: `{ex['out']}`",
        "",
        "```python",
        "import numpy as np",
        f"d = np.load('{ex['out']}', allow_pickle=True)",
        "# Keys: label_tc, label_names, atlas, subject, condition, task, mode",
        f"# label_tc.shape = ({ex['n_labels']}, {ex['n_times']})",
        f"# label_names[:5] = {ex['label_names_5']}",
        "```",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metrica | Valore |",
        "|---------|--------|",
        f"| File .npz prodotti | {len(results)} / 20 |",
        f"| Atlases | {', '.join(ATLASES)} |",
        f"| Wall-clock totale | {total_elapsed:.1f}s |",
        f"| Verdict | {'**PASS** ✅' if all_ok else '**WARN** ⚠️'} |",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    print(f"STEP 3 Parcellation pilot — {len(PILOT_SUBJECTS)} sub × {len(ATLASES)} atlases × {len(CONDITIONS)} cond")
    print(f"Mode: {MODE}")
    t0 = time.perf_counter()
    results = run_extraction()
    total = time.perf_counter() - t0
    print(f"\nTotal: {total:.1f}s | {len(results)} file .npz")
    write_report(results, total)
    print(f"Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()
