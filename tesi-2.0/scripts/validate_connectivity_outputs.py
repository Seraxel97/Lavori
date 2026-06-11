"""Validazione output FC S-104 (connectivity npz).

Verifica per ogni file .npz:
  1. File esiste (altrimenti WAITING).
  2. Caricabile (no CorruptedFile).
  3. Chiave fc_{band} presente.
  4. Shape NxN simmetrica (N = n ROI atlas).
  5. No NaN / no Inf.
  6. Range valori in [-1,1] per metriche signed; [0,1] per metriche positive.
  7. Simmetria numerica: |M - M.T|.max() < tol.

Range attesi per metrica:
  wpli, imcoh, ciplv, pli  → [-1, 1]   (signed, phase-based)
  wpli2_debiased, plv, coh → [0, 1]    (non-negative)

N ROI attesi per atlas:
  aparc       → 68
  schaefer100 → 100
  schaefer200 → 200
  destrieux   → 148

Usage:
    python scripts/validate_connectivity_outputs.py \\
        --sub 007 --atlas aparc --cond EO --metric wpli --band alpha
    python scripts/validate_connectivity_outputs.py \\
        --sub-list 007 010 011 026 031 --report reports/CONN_PREFLIGHT.md
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

FC_DIR = Path("data/connectivity/ds005385")

# Range per metrica: (min_ok, max_ok)
METRIC_RANGES: dict[str, tuple[float, float]] = {
    "wpli": (-1.0, 1.0),
    "wpli2_debiased": (0.0, 1.0),
    "plv": (0.0, 1.0),
    "coh": (0.0, 1.0),
    "imcoh": (-1.0, 1.0),
    "ciplv": (-1.0, 1.0),
    "pli": (-1.0, 1.0),
}

N_ROI: dict[str, int] = {
    "aparc": 68,
    "schaefer100": 100,
    "schaefer200": 200,
    "destrieux": 148,
}

SYMMETRY_TOL = 1e-5


def _fc_path(sub: str, atlas: str, cond: str, metric: str) -> Path:
    return FC_DIR / f"{sub}_atlas-{atlas}_cond-{cond}_metric-{metric}.npz"


def validate_one(
    sub: str, atlas: str, cond: str, metric: str, band: str
) -> dict:
    path = _fc_path(sub, atlas, cond, metric)
    result: dict = {
        "sub": sub, "atlas": atlas, "cond": cond, "metric": metric, "band": band,
        "file": path.name, "exists": path.exists(),
        "checks": {}, "verdict": "WAITING", "errors": [],
    }

    if not path.exists():
        return result

    # 1. Carica
    try:
        d = np.load(path, allow_pickle=False)
    except Exception as exc:
        result["verdict"] = "FAIL"
        result["errors"].append(f"load error: {exc}")
        return result

    fc_key = f"fc_{band}"

    # 2. Chiave presente
    if fc_key not in d:
        result["verdict"] = "FAIL"
        result["errors"].append(f"key '{fc_key}' not in npz (keys: {list(d.keys())})")
        return result

    m = d[fc_key]
    n_expected = N_ROI.get(atlas, -1)
    shape_ok = m.ndim == 2 and m.shape[0] == m.shape[1]
    n_ok = m.shape == (n_expected, n_expected) if n_expected > 0 else shape_ok
    result["checks"]["shape"] = f"{m.shape}" + (" OK" if n_ok else " WARN")

    # 3. NaN / Inf
    nan_count = int(np.isnan(m).sum())
    inf_count = int(np.isinf(m).sum())
    result["checks"]["nan"] = f"{nan_count}" + (" OK" if nan_count == 0 else " FAIL")
    result["checks"]["inf"] = f"{inf_count}" + (" OK" if inf_count == 0 else " FAIL")

    # 4. Range
    vmin, vmax = float(np.nanmin(m)), float(np.nanmax(m))
    rmin, rmax = METRIC_RANGES.get(metric, (-1.0, 1.0))
    range_ok = vmin >= rmin - 1e-6 and vmax <= rmax + 1e-6
    result["checks"]["range"] = (
        f"[{vmin:.3f},{vmax:.3f}] ⊆ [{rmin},{rmax}]" + (" OK" if range_ok else " FAIL")
    )

    # 5. Simmetria
    sym_err = float(np.abs(m - m.T).max())
    sym_ok = sym_err < SYMMETRY_TOL
    result["checks"]["symmetry"] = f"max_err={sym_err:.2e}" + (" OK" if sym_ok else " FAIL")

    # Verdict
    fails = [k for k, v in result["checks"].items() if "FAIL" in str(v)]
    if nan_count > 0:
        result["errors"].append(f"NaN count={nan_count}")
    if inf_count > 0:
        result["errors"].append(f"Inf count={inf_count}")
    if not range_ok:
        result["errors"].append(f"range out of [{rmin},{rmax}]: [{vmin:.4f},{vmax:.4f}]")
    if not sym_ok:
        result["errors"].append(f"not symmetric: max_err={sym_err:.2e}")

    result["verdict"] = "FAIL" if fails else "CLEAN"
    return result


def print_table(results: list[dict]) -> None:
    w = {"sub": 9, "atlas": 12, "cond": 5, "metric": 16, "band": 7,
         "verdict": 8, "checks": 55}
    hdr = (f"{'sub':<{w['sub']}} {'atlas':<{w['atlas']}} {'cond':<{w['cond']}} "
           f"{'metric':<{w['metric']}} {'band':<{w['band']}} {'verdict':<{w['verdict']}} checks")
    sep = "-" * (sum(w.values()) + 10)
    print(sep)
    print(hdr)
    print(sep)
    for r in results:
        checks_str = "  ".join(f"{k}={v}" for k, v in r["checks"].items()) or "-"
        print(f"{r['sub']:<{w['sub']}} {r['atlas']:<{w['atlas']}} {r['cond']:<{w['cond']}} "
              f"{r['metric']:<{w['metric']}} {r['band']:<{w['band']}} "
              f"{r['verdict']:<{w['verdict']}} {checks_str}")
    print(sep)


def build_md(results: list[dict], ts: str) -> str:
    n_clean = sum(1 for r in results if r["verdict"] == "CLEAN")
    n_fail = sum(1 for r in results if r["verdict"] == "FAIL")
    n_wait = sum(1 for r in results if r["verdict"] == "WAITING")
    overall = "WAITING" if n_wait == len(results) else (
        "FAIL" if n_fail > 0 else ("PARTIAL" if n_wait > 0 else "CLEAN")
    )
    lines = [
        "# Connectivity Outputs Pre-flight Check",
        "",
        f"**Sprint**: S-111  **Data**: {ts}  **Worker**: sonnet2-ts",
        f"**Verdict**: {overall}  CLEAN={n_clean}  FAIL={n_fail}  WAITING={n_wait}",
        "",
        "---",
        "",
        "## Tabella verifica",
        "",
        "| Sub | Atlas | Cond | Metric | Band | Verdict | Shape | NaN | Range | Symmetry |",
        "|-----|-------|------|--------|------|---------|-------|-----|-------|----------|",
    ]
    for r in results:
        c = r["checks"]
        lines.append(
            f"| {r['sub']} | {r['atlas']} | {r['cond']} | {r['metric']} | {r['band']} "
            f"| **{r['verdict']}** "
            f"| {c.get('shape','-')} | {c.get('nan','-')} "
            f"| {c.get('range','-')} | {c.get('symmetry','-')} |"
        )

    errors = [(r["sub"], r["metric"], r["errors"]) for r in results if r["errors"]]
    if errors:
        lines += ["", "## Errori", ""]
        for sub, metric, errs in errors:
            for e in errs:
                lines.append(f"- `{sub}` `{metric}`: {e}")

    lines += [
        "",
        "---",
        "",
        "## Note lifecycle FC",
        "",
        f"- **FC_DIR**: `{FC_DIR}/`",
        "- **Pattern file**: `{sub}_atlas-{atlas}_cond-{cond}_metric-{metric}.npz`",
        "- **Chiave npz**: `fc_{band}` (es. `fc_alpha`, `fc_beta`)",
        "- **Metriche signed** [-1,1]: wpli, imcoh, ciplv, pli",
        "- **Metriche positive** [0,1]: wpli2_debiased, plv, coh",
        "- **Simmetria**: |M - M.T|.max() < 1e-5",
        f"- **S-104 status**: {'DONE' if n_wait == 0 else 'IN FLIGHT (waiting)'}",
    ]
    if overall == "WAITING":
        lines.append(
            "\nS-104 in flight (sonnet1-ts). "
            "Ri-eseguire dopo completamento per aggiornare il report."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Valida output FC S-104")
    ap.add_argument("--sub", default=None, help="Singolo soggetto (es. 007)")
    ap.add_argument("--sub-list", nargs="+", default=None, metavar="NNN")
    ap.add_argument("--atlas", default="aparc",
                    choices=list(N_ROI.keys()))
    ap.add_argument("--cond", default="EO", choices=["EO", "EC"])
    ap.add_argument("--metric", default="wpli",
                    choices=list(METRIC_RANGES.keys()))
    ap.add_argument("--band", default="alpha")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    # Costruisci lista (sub, atlas, cond, metric, band)
    subs: list[str] = []
    if args.sub_list:
        subs = [f"sub-{n}" for n in args.sub_list]
    elif args.sub:
        subs = [f"sub-{args.sub}"]
    else:
        subs = ["sub-007", "sub-010", "sub-011", "sub-026", "sub-031"]

    # Se lista sub → itera su atlas/cond/metric/band default
    atlases = [args.atlas]
    conds = [args.cond]
    metrics = [args.metric]
    bands = [args.band]

    results = []
    for sub in subs:
        for atlas in atlases:
            for cond in conds:
                for metric in metrics:
                    for band in bands:
                        results.append(validate_one(sub, atlas, cond, metric, band))

    print_table(results)
    n_clean = sum(1 for r in results if r["verdict"] == "CLEAN")
    n_fail = sum(1 for r in results if r["verdict"] == "FAIL")
    n_wait = sum(1 for r in results if r["verdict"] == "WAITING")
    print(f"\nTotale={len(results)}  CLEAN={n_clean}  FAIL={n_fail}  WAITING={n_wait}")

    if args.report:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        md = build_md(results, ts)
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
        print(f"Report → {p}")


if __name__ == "__main__":
    main()
