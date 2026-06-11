"""Analisi risultati benchmark: compare runs, FDR correction, profiling."""

from analysis.compare_runs import load_bench, marginal_by_axis, top_n
from analysis.fdr_helper import apply_fdr
from analysis.profile_bench import profile_and_report

__all__ = [
    "load_bench",
    "marginal_by_axis",
    "top_n",
    "apply_fdr",
    "profile_and_report",
]
