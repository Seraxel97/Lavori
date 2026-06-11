"""
Results comparison helper — cross-run table, marginal BA per asse, FDR batch.

Carica BENCH_MATRIX_RESULTS.json e produce tabelle aggregate per confronto
configurazioni (metric x atlas x algorithm x band).

Usage:
    python analysis/compare_runs.py \\
        --bench reports/BENCH_MATRIX_RESULTS.json \\
        --output reports/COMPARE_RUNS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_AXES = ("metric", "atlas", "algorithm", "band")


def load_bench(path: str | Path) -> pd.DataFrame:
    """Carica BENCH_MATRIX_RESULTS.json come DataFrame.

    Parameters
    ----------
    path:
        Path al file JSON prodotto da run_bench_matrix.py.

    Returns
    -------
    pd.DataFrame
        Colonne: metric, atlas, algorithm, band, ba, + eventuali extra.
        DataFrame vuoto se il file e' un placeholder (runs=[]).
    """
    data = json.loads(Path(path).read_text())
    runs = data.get("runs", [])
    if not runs:
        return pd.DataFrame(columns=list(_AXES) + ["ba"])
    return pd.DataFrame(runs)


def marginal_by_axis(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Media BA per un asse (metric, atlas, algorithm, band).

    Parameters
    ----------
    df:
        DataFrame da load_bench.
    axis:
        Colonna su cui aggregare.

    Returns
    -------
    pd.DataFrame
        Colonne: [axis, 'ba_mean', 'ba_std', 'n_runs']. Ordinato per ba_mean desc.

    Raises
    ------
    ValueError
        Se axis non in ('metric', 'atlas', 'algorithm', 'band').
    """
    if axis not in _AXES:
        raise ValueError(f"axis deve essere in {_AXES}, got {axis!r}")
    if df.empty or axis not in df.columns or "ba" not in df.columns:
        return pd.DataFrame(columns=[axis, "ba_mean", "ba_std", "n_runs"])

    agg = (
        df.groupby(axis)["ba"]
        .agg(ba_mean="mean", ba_std="std", n_runs="count")
        .reset_index()
        .sort_values("ba_mean", ascending=False)
    )
    return agg.reset_index(drop=True)


def top_n(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-N configurazioni per BA decrescente.

    Parameters
    ----------
    df:
        DataFrame da load_bench.
    n:
        Numero di righe da restituire.

    Returns
    -------
    pd.DataFrame
        Top-N righe ordinate per ba desc.
    """
    if df.empty or "ba" not in df.columns:
        return df.head(n)
    return df.nlargest(n, "ba").reset_index(drop=True)


def _df_to_md(df: pd.DataFrame) -> str:
    """Converte DataFrame in tabella markdown."""
    if df.empty:
        return "_Nessun dato disponibile._\n"
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare benchmark runs — tabelle BA per asse")
    ap.add_argument("--bench", default="reports/BENCH_MATRIX_RESULTS.json")
    ap.add_argument("--output", default="reports/COMPARE_RUNS.md")
    ap.add_argument("--n-top", type=int, default=10)
    args = ap.parse_args()

    df = load_bench(args.bench)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Compare Runs — Benchmark Results\n",
        f"**Source**: `{args.bench}`  \n",
        f"**N runs totali**: {len(df)}\n\n",
    ]

    if df.empty:
        lines.append(
            "> **Nota**: `BENCH_MATRIX_RESULTS.json` e' un placeholder. "
            "Eseguire `run_bench_matrix.py` per risultati reali.\n"
        )
    else:
        lines.append(f"## Top {args.n_top} configurazioni\n\n")
        lines.append(_df_to_md(top_n(df, args.n_top)))
        lines.append("\n")
        for axis in _AXES:
            agg = marginal_by_axis(df, axis)
            lines.append(f"## Marginal BA per `{axis}`\n\n")
            lines.append(_df_to_md(agg))
            lines.append("\n")

    out.write_text("".join(lines))
    print(f"Report salvato: {out}")
    if not df.empty:
        print(f"Top 3:\n{top_n(df, 3).to_string(index=False)}")
