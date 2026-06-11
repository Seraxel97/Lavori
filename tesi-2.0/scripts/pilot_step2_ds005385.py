"""
STEP 2 pilot — source reconstruction ds005385 (5 soggetti).

Per ogni soggetto e condizione (EyesOpen, EyesClosed):
  1. Carica EDF raw (ses-1, acq-pre)
  2. Filtra 1-40 Hz, montage standard_1020, re-reference average
  3. Decima 1000 Hz → 250 Hz
  4. Epoch fixed-length 2s
  5. Forward solution (fsaverage template, oct6)
  6. Noise cov ad-hoc + inverse operator (dSPM)
  7. Evoked medio → STC
  8. Salva: fwd.fif, inv.fif, cond-EO/EC_inv-dSPM-{lh,rh}.stc

Usage:
    python scripts/pilot_step2_ds005385.py [--subjects sub-007 ...]
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import mne
import mne.minimum_norm

warnings.filterwarnings("ignore")

# ── Costanti ──────────────────────────────────────────────────────────────────

BIDS_ROOT = Path("data/raw/ds005385")
DERIV = Path("data/derivatives/mne-bids-pipeline")
SUBJECTS_DIR = Path("/home/seraxel/mne_data/MNE-fsaverage-data")
FSAVERAGE = "fsaverage"

PILOT_SUBJECTS = ["sub-007", "sub-010", "sub-011", "sub-026", "sub-031"]
CONDITIONS = {
    "EO": "EyesOpen",
    "EC": "EyesClosed",
}
SESSION = "ses-1"
ACQ = "acq-pre"

SFREQ_TARGET = 250.0       # Hz post-decimate
FILTER_LO = 1.0            # Hz
FILTER_HI = 40.0           # Hz
EPOCH_DURATION = 2.0       # s
EPOCH_OVERLAP = 0.5        # s
LAMBDA2 = 1.0 / 9.0
METHOD = "dSPM"
LOOSE = 0.2
DEPTH = 0.8
MINDIST = 5.0              # mm, skip sources too close to inner skull

# ── Helpers ───────────────────────────────────────────────────────────────────


def _edf_path(sub: str, cond_task: str) -> Path:
    return BIDS_ROOT / sub / SESSION / "eeg" / (
        f"{sub}_{SESSION}_task-{cond_task}_{ACQ}_eeg.edf"
    )


def _out_dir(sub: str) -> Path:
    d = DERIV / sub / "eeg"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_and_preprocess(edf_path: Path) -> mne.io.Raw:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.pick("eeg", verbose=False)          # drop Status channel
    raw.set_montage("standard_1020", verbose=False)
    raw.filter(FILTER_LO, FILTER_HI, method="fir", verbose=False)
    decim = int(raw.info["sfreq"] / SFREQ_TARGET)
    if decim > 1:
        raw.resample(SFREQ_TARGET, verbose=False)
    raw.set_eeg_reference("average", projection=True, verbose=False)
    return raw


def make_epochs(raw: mne.io.Raw) -> mne.Epochs:
    overlap = EPOCH_OVERLAP
    events = mne.make_fixed_length_events(
        raw, duration=EPOCH_DURATION, overlap=overlap
    )
    epochs = mne.Epochs(
        raw, events, tmin=0.0, tmax=EPOCH_DURATION,
        baseline=None, preload=True, verbose=False
    )
    return epochs


def compute_forward(info: mne.Info, out_dir: Path, sub: str) -> mne.Forward:
    fwd_path = out_dir / f"{sub}_fwd.fif"
    if fwd_path.exists():
        print(f"  [fwd] loading cached {fwd_path.name}")
        return mne.read_forward_solution(str(fwd_path), verbose=False)

    print("  [fwd] computing forward solution (oct6, fsaverage template)…")
    src = mne.read_source_spaces(
        str(SUBJECTS_DIR / FSAVERAGE / "bem" / "fsaverage-oct6-src.fif"),
        verbose=False
    )
    bem = mne.read_bem_solution(
        str(SUBJECTS_DIR / FSAVERAGE / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"),
        verbose=False
    )
    trans = str(SUBJECTS_DIR / FSAVERAGE / "bem" / "fsaverage-trans.fif")
    fwd = mne.make_forward_solution(
        info, trans=trans, src=src, bem=bem,
        eeg=True, meg=False, mindist=MINDIST, verbose=False
    )
    mne.write_forward_solution(str(fwd_path), fwd, overwrite=True, verbose=False)
    print(f"    saved → {fwd_path.name}")
    return fwd


def compute_inverse(info: mne.Info, fwd: mne.Forward, out_dir: Path, sub: str) -> mne.minimum_norm.InverseOperator:
    inv_path = out_dir / f"{sub}_inv.fif"
    if inv_path.exists():
        print(f"  [inv] loading cached {inv_path.name}")
        return mne.minimum_norm.read_inverse_operator(str(inv_path), verbose=False)

    noise_cov = mne.make_ad_hoc_cov(info, verbose=False)
    inv_op = mne.minimum_norm.make_inverse_operator(
        info, fwd, noise_cov, loose=LOOSE, depth=DEPTH, verbose=False
    )
    mne.minimum_norm.write_inverse_operator(str(inv_path), inv_op, overwrite=True, verbose=False)
    print(f"  [inv] saved → {inv_path.name}")
    return inv_op


def process_condition(
    sub: str, cond_key: str, cond_task: str,
    inv_op: mne.minimum_norm.InverseOperator, out_dir: Path
) -> dict:
    edf = _edf_path(sub, cond_task)
    if not edf.exists():
        return {"cond": cond_key, "status": "missing_file", "file": str(edf)}

    t0 = time.perf_counter()
    raw = load_and_preprocess(edf)
    epochs = make_epochs(raw)
    evoked = epochs.average()

    stc = mne.minimum_norm.apply_inverse(
        evoked, inv_op, lambda2=LAMBDA2, method=METHOD, verbose=False
    )
    stem = out_dir / f"{sub}_task-RestingState_cond-{cond_key}_inv-{METHOD}"
    stc.save(str(stem), overwrite=True, verbose=False)
    elapsed = time.perf_counter() - t0

    info = {
        "cond": cond_key,
        "status": "ok",
        "n_epochs": len(epochs),
        "evoked_shape": evoked.data.shape,
        "stc_shape": stc.data.shape,
        "stc_vertices": (len(stc.vertices[0]), len(stc.vertices[1])),
        "stc_tstep": stc.tstep,
        "elapsed_s": round(elapsed, 2),
        "files": [f"{stem.name}-lh.stc", f"{stem.name}-rh.stc"],
    }
    print(
        f"  [{cond_key}] {len(epochs)} epochs | evoked={evoked.data.shape} "
        f"| STC={stc.data.shape} vertices=({len(stc.vertices[0])},{len(stc.vertices[1])}) "
        f"| {elapsed:.1f}s"
    )
    return info


def process_subject(sub: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  {sub}")
    print(f"{'='*60}")
    t_sub = time.perf_counter()
    out_dir = _out_dir(sub)
    results: dict = {"sub": sub, "conditions": {}, "warnings": []}

    # Load one raw to get info for forward/inverse computation
    first_edf = _edf_path(sub, "EyesOpen")
    if not first_edf.exists():
        first_edf = _edf_path(sub, "EyesClosed")
    raw_info = load_and_preprocess(first_edf)
    fwd = compute_forward(raw_info.info, out_dir, sub)
    inv_op = compute_inverse(raw_info.info, fwd, out_dir, sub)
    del raw_info

    for cond_key, cond_task in CONDITIONS.items():
        cres = process_condition(sub, cond_key, cond_task, inv_op, out_dir)
        results["conditions"][cond_key] = cres

    results["elapsed_s"] = round(time.perf_counter() - t_sub, 2)
    print(f"  {sub} DONE in {results['elapsed_s']:.1f}s")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="Pilot STEP 2 source recon ds005385")
    ap.add_argument("--subjects", nargs="+", default=PILOT_SUBJECTS)
    args = ap.parse_args()

    t_total = time.perf_counter()
    print("Pilot STEP 2 — ds005385 source reconstruction")
    print(f"Subjects: {args.subjects}")
    print(f"Method: {METHOD}, lambda2={LAMBDA2:.3f}, loose={LOOSE}, depth={DEPTH}")
    print(f"Sfreq target: {SFREQ_TARGET} Hz | Epochs: {EPOCH_DURATION}s | Filter: {FILTER_LO}-{FILTER_HI} Hz")

    all_results = []
    for sub in args.subjects:
        res = process_subject(sub)
        all_results.append(res)

    total_elapsed = time.perf_counter() - t_total
    print(f"\nTotal wall-clock: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Write report
    _write_report(all_results, total_elapsed)
    print("Report → reports/STEP2_DS005385_PILOT.md")


def _write_report(results: list[dict], total_elapsed: float) -> None:
    from datetime import UTC, datetime
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    lines = [
        "# STEP 2 Source Reconstruction — ds005385 PILOT",
        "",
        f"**Timestamp**: {ts}",
        "**Sprint**: S-101 (sonnet1-ts)",
        f"**Subjects**: {[r['sub'] for r in results]}",
        f"**Method**: {METHOD} | lambda2={LAMBDA2:.3f} | loose={LOOSE} | depth={DEPTH}",
        "**Template MRI**: fsaverage oct6",
        f"**Preprocessing**: {FILTER_LO}-{FILTER_HI} Hz, {SFREQ_TARGET} Hz, epochs {EPOCH_DURATION}s",
        "",
        "---",
        "",
        "## Risultati per soggetto",
        "",
    ]

    all_ok = True
    for r in results:
        sub = r["sub"]
        lines += [f"### {sub}", ""]
        lines.append(f"**Wall-clock**: {r.get('elapsed_s', '?')}s")
        for cond, cres in r.get("conditions", {}).items():
            status = cres.get("status", "?")
            if status != "ok":
                all_ok = False
                lines.append(f"- `{cond}`: **{status}** ⚠️")
            else:
                lines.append(
                    f"- `{cond}`: {cres['n_epochs']} epochs | "
                    f"evoked={cres['evoked_shape']} | "
                    f"STC={cres['stc_shape']} "
                    f"vtx=({cres['stc_vertices'][0]},{cres['stc_vertices'][1]}) | "
                    f"{cres['elapsed_s']}s"
                )
        if r.get("warnings"):
            for w in r["warnings"]:
                lines.append(f"- ⚠️ {w}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Summary",
        "",
        "| Metrica | Valore |",
        "|---------|--------|",
        f"| Soggetti | {len(results)} |",
        "| Condizioni | EO, EC (task-EyesOpen, task-EyesClosed) |",
        f"| Wall-clock totale | {total_elapsed:.1f}s ({total_elapsed/60:.1f} min) |",
        f"| Verdict | {'**PASS** ✅' if all_ok else '**WARN** ⚠️'} |",
        "",
    ]

    out = Path("reports/STEP2_DS005385_PILOT.md")
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
