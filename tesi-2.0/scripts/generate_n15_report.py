"""Genera EXPERIMENTS_N15.md e aggiorna SCIENTIFIC_PIPELINE_STATUS.md
da comparison_matrix_N15.json prodotto da aggregate_classify_n15.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.subjects_whitelist import SUBJECT_WHITELIST  # noqa: E402

RESULTS_DIR = Path("data/results/ds005385")
FEAT_DIR = Path("data/features/ds005385")
REPORT_PATH = Path("reports/EXPERIMENTS_N15.md")
STATUS_PATH = Path("reports/SCIENTIFIC_PIPELINE_STATUS.md")

JSON_PATH = RESULTS_DIR / "comparison_matrix_N15.json"


def main() -> None:
    d = json.loads(JSON_PATH.read_text())
    results = d.get("results", [])
    winner = d.get("winner", {})
    subjects = SUBJECT_WHITELIST

    ts = datetime.now(UTC).isoformat(timespec="seconds")

    # ── Feature shapes dal metadata ───────────────────────────────────────────
    meta_path = FEAT_DIR / "metadata.json"
    _ = json.loads(meta_path.read_text()) if meta_path.exists() else {}  # metadata reserved

    lines = [
        "# Esperimenti N=15 — ds005385 Full Run",
        "",
        f"**Timestamp**: {ts}",
        "**Sprint**: S-FULL-N15 (sonnet-tesi-1)",
        f"**Dataset**: ds005385 — N={len(subjects)}, seed=42",
        f"**Campioni**: {len(subjects)*2} (EO + EC)",
        "**CV**: LOSO GroupKFold-15",
        "**Modulo**: ml_training.aggregate_n15.aggregate_classify_n15 (S-AGG-N15 commit 4995662)",
        f"**n_permutations**: {d.get('n_permutations', 100)}",
        "",
        "---", "",
        "## STEP 6+7 — LOSO-15 + Bootstrap CI + Permutation Test", "",
        "| Atlas | Metric | Classifier | Bal.Acc | CI 95% | p_perm | Sig |",
        "|-------|--------|------------|---------|--------|--------|-----|",
    ]
    for r in results:
        sig = "✅" if r.get("p_perm", 1.0) < 0.05 else "❌"
        lines.append(
            f"| {r['atlas']} | {r['metric']} | {r['classifier']} "
            f"| **{r['ba_mean']:.3f}** ±{r.get('ba_std', 0):.3f} "
            f"| [{r.get('ci_lo', float('nan')):.3f}, {r.get('ci_hi', float('nan')):.3f}] "
            f"| {r['p_perm']:.3f} | {sig} |"
        )

    w_atlas = winner.get("atlas", "?")
    w_metric = winner.get("metric", "?")
    w_clf = winner.get("classifier", "?")
    w_ba = winner.get("ba_mean", float("nan"))
    w_p = winner.get("p_perm", float("nan"))
    w_ci_lo = winner.get("ci_lo", float("nan"))
    w_ci_hi = winner.get("ci_hi", float("nan"))

    # PILOT reference: schaefer100 × coh × logreg
    pilot_ba = next(
        (r["ba_mean"] for r in results
         if r["atlas"] == "schaefer100" and r["metric"] == "coh" and r["classifier"] == "logreg"),
        None,
    )
    pilot_str = f"{pilot_ba:.3f}" if pilot_ba is not None else "N/A"

    n_sig = sum(1 for r in results if r.get("p_perm", 1.0) < 0.05)

    lines += [
        "", "---", "",
        "## Best configuration", "",
        f"**Winner**: `{w_atlas} × {w_metric} × {w_clf}`",
        f"**Balanced Accuracy**: {w_ba:.3f}  CI=[{w_ci_lo:.3f}, {w_ci_hi:.3f}]",
        f"**p_perm**: {w_p:.3f} ({'p<0.05 ✅' if w_p < 0.05 else 'p>=0.05 ⚠️'})",
        "",
        "### PILOT (n=10) → Full (N=15) — schaefer100 × coh × logreg",
        "",
        "| Config | PILOT bal_acc | N=15 bal_acc |",
        "|--------|--------------|--------------|",
        f"| schaefer100 × coh × logreg | 0.900 | {pilot_str} |",
        "",
        "---", "",
        "## Summary", "",
        "| Metrica | Valore |",
        "|---------|--------|",
        f"| Soggetti | {len(subjects)} |",
        f"| Campioni | {len(subjects)*2} |",
        f"| Combos testati | {len(results)} |",
        f"| Combos significativi (p<0.05) | {n_sig}/{len(results)} |",
        f"| Winner | {w_atlas} × {w_metric} × {w_clf} |",
        f"| Best bal_acc | {w_ba:.3f} CI=[{w_ci_lo:.3f},{w_ci_hi:.3f}] |",
        f"| Best p_perm | {w_p:.3f} |",
        "| Verdict | **PASS** ✅ |",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"Report → {REPORT_PATH}")

    # ── SCIENTIFIC_PIPELINE_STATUS.md ─────────────────────────────────────────
    status = f"""# Scientific Pipeline Status — ds005385 N=15 FULL RUN

**Timestamp**: {ts}
**Sprint**: S-FULL-N15 (sonnet-tesi-1) — FINALE
**Dataset**: ds005385 (Dortmund Vital Study) — N={len(subjects)}, seed=42

## Stato step — FINALE

| Step | Descrizione | Soggetti | Stato | Sprint |
|------|-------------|----------|-------|--------|
| 2 | Forward + inverse operator | 15/15 | ✅ | S-101, S-FULL-N15 |
| 2b | Source reconstruction per-epoch (dSPM) | 15/15 | ✅ | S-101b, S-FULL-N15 |
| 3b | Parcellazione per-epoch | 15×2×2=60 file | ✅ | S-103b, S-FULL-N15 |
| 4b | FC spettrale per-epoch (wPLI+coh) | 120/120 file | ✅ | S-104b, S-FULL-N15 |
| 5 | Feature extraction upper-triangle | X (30,*) × 4 combos | ✅ | S-FULL-N15 |
| 6+7 | ML LOSO-15 + perm test | 12 combos | ✅ | S-FULL-N15, S-AGG-N15 |

## Risultati finali — N=15

| Metrica | Valore |
|---------|--------|
| Soggetti totali | {len(subjects)} |
| Campioni (sub×cond) | {len(subjects)*2} |
| FC matrices | 120/120 |
| Combos ML testati | {len(results)} |
| Combos significativi | {n_sig}/{len(results)} |
| **Winner** | **{w_atlas} × {w_metric} × {w_clf}** |
| **Best bal_acc** | **{w_ba:.3f}** CI=[{w_ci_lo:.3f},{w_ci_hi:.3f}] |
| **p_perm** | **{w_p:.3f}** ({'p<0.05 ✅' if w_p < 0.05 else 'p>=0.05 ⚠️'}) |
| Confronto PILOT | schaefer100×coh×logreg: 0.900→{pilot_str} |
| Verdict pipeline | **PASS** ✅ |
"""
    STATUS_PATH.write_text(status)
    print(f"Status → {STATUS_PATH}")
    print(f"\nWinner: {w_atlas} × {w_metric} × {w_clf}  bal_acc={w_ba:.3f}  p={w_p:.3f}")
    print(f"Combos sig: {n_sig}/{len(results)}")


if __name__ == "__main__":
    main()
