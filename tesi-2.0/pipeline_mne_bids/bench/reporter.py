"""Bench reporter: dump JSON incrementale e summary Markdown."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def dump_json(path: Path, meta: dict, runs: list[dict]) -> None:
    """Scrive BENCH_MATRIX_RESULTS.json (sovrascrittura atomica)."""
    data = {"meta": meta, "runs": runs}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _marginal(runs: list[dict], key: str) -> dict[str, float]:
    sums: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        if r["balanced_accuracy"] is not None:
            sums[r[key]].append(r["balanced_accuracy"])
    return {k: round(sum(v) / len(v), 4) for k, v in sums.items()}


def write_summary(path: Path, meta: dict, runs: list[dict]) -> None:
    """Scrive BENCH_MATRIX_SUMMARY.md con Top-10 e marginal BA per asse."""
    valid = [r for r in runs if r["balanced_accuracy"] is not None]
    failed = [r for r in runs if r["error"] is not None]

    lines = [
        "# Bench Matrix Summary — matchingpennies",
        "",
        f"**Subject**: {meta['subject']} | **N run**: {meta['n_runs']}  ",
        f"**git**: {meta['git_sha']} | **config**: {meta['config_hash']}  ",
        f"**Start**: {meta['ts_start']} → **End**: {meta['ts_end']}  ",
        "",
        "## Top-10 by Balanced Accuracy",
        "",
        "| BA | Atlas | Band | Metric | Algorithm | N feat |",
        "|----|-------|------|--------|-----------|--------|",
    ]

    top10 = sorted(valid, key=lambda r: r["balanced_accuracy"], reverse=True)[:10]
    for r in top10:
        lines.append(
            f"| {r['balanced_accuracy']:.4f} | {r['atlas']} | {r['band']} "
            f"| {r['metric']} | {r['algorithm']} | {r['n_features']} |"
        )

    for axis_key, axis_label in [
        ("metric", "FC Metric"),
        ("atlas", "Atlas"),
        ("algorithm", "Algorithm"),
        ("band", "Band"),
    ]:
        marginals = _marginal(valid, axis_key)
        lines += ["", f"### {axis_label} (marginal BA)"]
        lines += [f"- {k}: {v}" for k, v in sorted(marginals.items(), key=lambda x: -x[1])]

    if failed:
        lines += ["", f"## Failed runs ({len(failed)})", ""]
        for r in failed[:10]:
            lines.append(
                f"- run {r['run_id']} {r['atlas']}/{r['band']}/{r['metric']}: {r['error']}"
            )

    wall = sum(r["fit_time_s"] or 0 for r in runs)
    lines += ["", "## Wall-clock budget", f"- Totale fit: {wall:.1f}s"]

    path.write_text("\n".join(lines))
