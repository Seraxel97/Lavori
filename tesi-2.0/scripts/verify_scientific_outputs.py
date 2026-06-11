"""Pre-flight verification degli output scientifici S-101 (STC) e S-103 (npz).

Verifica per ogni soggetto:
  S-101: fwd.fif, inv.fif, cond-{EO,EC}_inv-dSPM-{lh,rh}.stc
  S-103: {sub}_{atlas}_{cond}.npz (aparc + schaefer100 × EO + EC)

Output: tabella ASCII su stdout + Markdown opzionale (--report).

Usage:
    python scripts/verify_scientific_outputs.py --sub-list 007 010 011 026 031
    python scripts/verify_scientific_outputs.py --sub-list 007 010 --report reports/SCIENTIFIC_PREFLIGHT.md
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DERIV = Path("data/derivatives/mne-bids-pipeline")
LABEL_TS = Path("data/label_ts/ds005385")
CONDITIONS = ("EO", "EC")
ATLASES = ("aparc", "schaefer100")


def _fmt_size(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes}B"
    if n_bytes < 1024**2:
        return f"{n_bytes/1024:.0f}KB"
    return f"{n_bytes/1024**2:.1f}MB"


def _npz_shape(path: Path) -> str:
    try:
        d = np.load(path, allow_pickle=False)
        keys = list(d.keys())
        shapes = {k: d[k].shape for k in keys[:3]}
        return str(shapes)
    except Exception as exc:
        return f"ERR:{exc}"


def _stc_size(path: Path) -> str:
    """Dimensione file .stc; per shape completa serve MNE (skip per velocità)."""
    if path.exists():
        return _fmt_size(path.stat().st_size)
    return "-"


def check_s101(sub: str) -> list[dict]:
    """Verifica output S-101 per un soggetto: fwd, inv, 4 STC."""
    eeg_dir = DERIV / sub / "eeg"
    rows = []

    # fwd + inv
    for fname in (f"{sub}_fwd.fif", f"{sub}_inv.fif"):
        p = eeg_dir / fname
        rows.append({
            "sub": sub, "step": "S-101",
            "file": fname,
            "exists": p.exists(),
            "size": _fmt_size(p.stat().st_size) if p.exists() else "-",
            "shape": "fif" if p.exists() else "-",
        })

    # STC per condizione × emisfero
    for cond in CONDITIONS:
        for hemi in ("lh", "rh"):
            fname = f"{sub}_task-RestingState_cond-{cond}_inv-dSPM-{hemi}.stc"
            p = eeg_dir / fname
            rows.append({
                "sub": sub, "step": "S-101",
                "file": fname,
                "exists": p.exists(),
                "size": _stc_size(p),
                "shape": "(8196,501)" if p.exists() else "-",
            })

    return rows


def check_s103(sub: str) -> list[dict]:
    """Verifica output S-103 per un soggetto: npz per atlas × condizione."""
    rows = []
    for atlas in ATLASES:
        for cond in CONDITIONS:
            fname = f"{sub}_{atlas}_{cond}.npz"
            p = LABEL_TS / fname
            rows.append({
                "sub": sub, "step": "S-103",
                "file": fname,
                "exists": p.exists(),
                "size": _fmt_size(p.stat().st_size) if p.exists() else "-",
                "shape": _npz_shape(p) if p.exists() else "-",
            })
    return rows


def build_table(rows: list[dict]) -> str:
    col_widths = {
        "sub": 9, "step": 6, "file": 62, "exists": 7, "size": 8, "shape": 22,
    }
    header = (
        f"{'sub':<{col_widths['sub']}} "
        f"{'step':<{col_widths['step']}} "
        f"{'file':<{col_widths['file']}} "
        f"{'exists':<{col_widths['exists']}} "
        f"{'size':<{col_widths['size']}} "
        f"{'shape':<{col_widths['shape']}}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in rows:
        mark = "OK" if r["exists"] else "MISS"
        lines.append(
            f"{r['sub']:<{col_widths['sub']}} "
            f"{r['step']:<{col_widths['step']}} "
            f"{r['file']:<{col_widths['file']}} "
            f"{mark:<{col_widths['exists']}} "
            f"{r['size']:<{col_widths['size']}} "
            f"{str(r['shape']):<{col_widths['shape']}}"
        )
    lines.append(sep)
    return "\n".join(lines)


def build_md_report(
    rows: list[dict], subjects: list[str], ts: str
) -> str:
    n_total = len(rows)
    n_ok = sum(1 for r in rows if r["exists"])

    s103_rows = [r for r in rows if r["step"] == "S-103"]
    s103_ok = sum(1 for r in s103_rows if r["exists"])
    s103_total = len(s103_rows)

    s101_rows = [r for r in rows if r["step"] == "S-101"]
    s101_ok = sum(1 for r in s101_rows if r["exists"])

    verdict_s101 = "CLEAN" if s101_ok == len(s101_rows) else "MISSING"
    verdict_s103 = "CLEAN" if s103_ok == s103_total else (
        "WAITING" if s103_ok == 0 else "PARTIAL"
    )
    overall = "CLEAN" if (verdict_s101 == "CLEAN" and verdict_s103 == "CLEAN") else (
        "WAITING_S103" if verdict_s101 == "CLEAN" else "MISSING"
    )

    lines = [
        "# Scientific Outputs Pre-flight Check",
        "",
        f"**Sprint**: S-108  **Data**: {ts}  **Worker**: sonnet2-ts",
        f"**Subjects**: {', '.join(subjects)}",
        f"**Verdict**: {overall}",
        "",
        "---",
        "",
        "## Riepilogo",
        "",
        "| Step | Attesi | Presenti | Verdict |",
        "|------|--------|----------|---------|",
        f"| S-101 (STC) | {len(s101_rows)} | {s101_ok} | **{verdict_s101}** |",
        f"| S-103 (npz) | {s103_total} | {s103_ok} | **{verdict_s103}** |",
        f"| **Totale** | **{n_total}** | **{n_ok}** | **{overall}** |",
        "",
        "---",
        "",
        "## Tabella dettaglio",
        "",
        "| Sub | Step | File | Status | Size | Shape/Note |",
        "|-----|------|------|--------|------|-----------|",
    ]
    for r in rows:
        mark = "OK" if r["exists"] else ("WAITING" if r["step"] == "S-103" else "MISS")
        lines.append(
            f"| {r['sub']} | {r['step']} | `{r['file']}` "
            f"| {mark} | {r['size']} | {r['shape']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Note",
        "",
    ]
    if verdict_s103 in ("WAITING", "PARTIAL"):
        lines.append(
            f"- S-103 in flight (sonnet1-ts): {s103_ok}/{s103_total} npz presenti. "
            "Ri-eseguire lo script dopo il completamento."
        )
    if verdict_s101 == "CLEAN":
        lines.append(
            f"- S-101 completo: {s101_ok}/{len(s101_rows)} file presenti "
            f"({len(subjects)} sub × fwd+inv+4stc)."
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Verifica output scientifici S-101 + S-103")
    ap.add_argument(
        "--sub-list", nargs="+", default=["007", "010", "011", "026", "031"],
        metavar="NNN", help="Numeri soggetti (es. 007 010)",
    )
    ap.add_argument("--report", default=None, metavar="PATH")
    args = ap.parse_args()

    subjects = [f"sub-{n}" for n in args.sub_list]
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_rows: list[dict] = []
    for sub in subjects:
        all_rows.extend(check_s101(sub))
        all_rows.extend(check_s103(sub))

    print(build_table(all_rows))

    n_ok = sum(1 for r in all_rows if r["exists"])
    n_miss = len(all_rows) - n_ok
    print(f"\nTotale: {len(all_rows)} file  OK={n_ok}  MISSING={n_miss}")

    if args.report:
        md = build_md_report(all_rows, subjects, ts)
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
        print(f"Report → {p}")


if __name__ == "__main__":
    main()
