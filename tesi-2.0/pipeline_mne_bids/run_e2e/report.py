"""Funzioni di reporting E2E — markdown + json summary."""

from __future__ import annotations

import json
from pathlib import Path


def write_report(summary: dict, path: Path) -> None:
    """Scrive il report MD dell'E2E run.

    Parameters
    ----------
    summary:
        Dizionario risultati prodotto da exec.run_e2e.
    path:
        Path file MD di output.
    """
    lines = [
        "# E2E Smoke Test — matchingpennies",
        "",
        f"**Subject**: {summary['subject']}  ",
        f"**Atlas**: {summary['atlas']}  ",
        f"**FC metric**: {summary['metric']}  ",
        f"**Bande**: {', '.join(summary['bands'])}  ",
        f"**N epoch**: {summary['n_epochs']} | sfreq={summary['sfreq']:.0f} Hz  ",
        f"**N labels**: {summary['n_labels']} | N features: {summary['n_features']}  ",
        f"**Elapsed**: {summary['elapsed_s']}s  ",
        "",
        "## Risultati ML (StratifiedKFold)",
        "",
        "| Algoritmo | BA mean | BA std | Fold BAs |",
        "|-----------|---------|--------|----------|",
    ]
    for algo, r in summary["results"].items():
        fold_str = " | ".join(f"{b:.3f}" for b in r["ba_per_fold"])
        lines.append(f"| {algo} | {r['ba_mean']:.3f} | {r['ba_std']:.3f} | {fold_str} |")

    lines += [
        "",
        "> **Nota**: N=1 soggetto → risultati non significativi scientificamente.",
        "> Pipeline smoke test: verifica che tutti i moduli si interfaccino correttamente.",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines))
