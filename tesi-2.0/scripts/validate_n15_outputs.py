"""S-VALIDATE-N15 — Data integrity validation for N=15 ds005385 outputs.

Validates label_tc per-epoch, FC matrices, and feature matrices produced by
the S-FULL-N15 pipeline run. Read-only — no pipeline execution, no data modification.

Usage
-----
    python scripts/validate_n15_outputs.py
    python scripts/validate_n15_outputs.py --data-root /path/to/Tesi_2.0 --output reports/VALIDATE_N15.md

Exit codes: 0=PASS, 1=WARN (anomalies present but not critical), 2=FAIL
"""

from __future__ import annotations

import argparse
import datetime
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUBJECTS = [
    "sub-007", "sub-010", "sub-011", "sub-026", "sub-031",
    "sub-033", "sub-041", "sub-066", "sub-071", "sub-080",
    "sub-125", "sub-157", "sub-169", "sub-185", "sub-195",
]
ATLASES = ["aparc", "schaefer100"]
CONDITIONS = ["EO", "EC"]
METRICS = ["wpli", "coh"]

_ATLAS_N_LABELS = {"aparc": 68, "schaefer100": 100}
_N_TIMES_EXPECTED = 501
_N_EPOCHS_MIN, _N_EPOCHS_MAX = 80, 200
_N_SUBJECTS = 15
_N_SAMPLES_EXPECTED = _N_SUBJECTS * 2  # 30
_FEAT_N_LABELS = {"aparc": 2278, "schaefer100": 4950}

_WPLI_OFFDIAG_MED_MIN = 0.0   # wPLI can be 0 on diagonal; off-diag median > 0
_WPLI_OFFDIAG_MAX_ABS = 1.0
_COH_MIN, _COH_MAX = 0.0, 1.0
_LABEL_TC_MAX_ABS = 1e6


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    category: str
    file: str
    field: str
    value: str


@dataclass
class SubjectStats:
    subject: str
    wpli_eo_median: float = float("nan")
    wpli_ec_median: float = float("nan")
    wpli_eo_max: float = float("nan")
    wpli_ec_max: float = float("nan")
    coh_eo_median: float = float("nan")
    coh_ec_median: float = float("nan")


@dataclass
class ValidationReport:
    timestamp: str
    lt_expected: int = 60
    lt_found: int = 0
    lt_valid: int = 0
    lt_anomalies: int = 0
    fc_expected: int = 120
    fc_found: int = 0
    fc_valid: int = 0
    fc_anomalies: int = 0
    feat_expected: int = 4
    feat_found: int = 0
    feat_valid: int = 0
    feat_anomalies: int = 0
    anomalies: list[Anomaly] = field(default_factory=list)
    subject_stats: list[SubjectStats] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        total_anom = self.lt_anomalies + self.fc_anomalies + self.feat_anomalies
        missing = (
            (self.lt_expected - self.lt_found)
            + (self.fc_expected - self.fc_found)
            + (self.feat_expected - self.feat_found)
        )
        if missing > 0 or self.feat_anomalies > 0:
            return "FAIL"
        if total_anom > 0:
            return "WARN"
        return "PASS"


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_label_tc(lt_dir: Path, report: ValidationReport) -> None:
    """Validate all label_tc per-epoch files."""
    for sub in SUBJECTS:
        for atlas in ATLASES:
            for cond in CONDITIONS:
                fname = f"{sub}_atlas-{atlas}_cond-{cond}_per-epoch.npz"
                fpath = lt_dir / fname
                if not fpath.exists():
                    report.lt_anomalies += 1
                    report.anomalies.append(Anomaly("label_tc", fname, "exists", "MISSING"))
                    continue
                report.lt_found += 1
                try:
                    d = np.load(fpath, allow_pickle=False)
                    lt = d["label_tc"]
                except Exception as e:
                    report.lt_anomalies += 1
                    report.anomalies.append(Anomaly("label_tc", fname, "load", str(e)[:80]))
                    continue

                ok = True
                n_epochs, n_labels, n_times = lt.shape

                if n_labels != _ATLAS_N_LABELS[atlas]:
                    report.anomalies.append(Anomaly(
                        "label_tc", fname, "n_labels",
                        f"{n_labels} != {_ATLAS_N_LABELS[atlas]}"
                    ))
                    ok = False

                if n_times != _N_TIMES_EXPECTED:
                    report.anomalies.append(Anomaly(
                        "label_tc", fname, "n_times",
                        f"{n_times} != {_N_TIMES_EXPECTED}"
                    ))
                    ok = False

                if not (_N_EPOCHS_MIN <= n_epochs <= _N_EPOCHS_MAX):
                    report.anomalies.append(Anomaly(
                        "label_tc", fname, "n_epochs",
                        f"{n_epochs} out of [{_N_EPOCHS_MIN}, {_N_EPOCHS_MAX}]"
                    ))
                    ok = False

                if not np.isfinite(lt).all():
                    n_bad = int(np.sum(~np.isfinite(lt)))
                    report.anomalies.append(Anomaly(
                        "label_tc", fname, "finite", f"{n_bad} NaN/Inf"
                    ))
                    ok = False

                max_abs = float(np.abs(lt).max())
                if max_abs > _LABEL_TC_MAX_ABS:
                    report.anomalies.append(Anomaly(
                        "label_tc", fname, "max_abs", f"{max_abs:.3e} > {_LABEL_TC_MAX_ABS:.0e}"
                    ))
                    ok = False

                if ok:
                    report.lt_valid += 1
                else:
                    report.lt_anomalies += 1


def _validate_fc(fc_dir: Path, report: ValidationReport) -> None:
    """Validate FC per-epoch matrices (120 expected: 15×2×2×2)."""
    stats_map: dict[str, SubjectStats] = {s: SubjectStats(subject=s) for s in SUBJECTS}

    for sub in SUBJECTS:
        for atlas in ATLASES:
            n_labels = _ATLAS_N_LABELS[atlas]
            for cond in CONDITIONS:
                for metric in METRICS:
                    fname = (f"{sub}_atlas-{atlas}_cond-{cond}"
                             f"_metric-{metric}_band-alpha_per-epoch.npz")
                    fpath = fc_dir / fname
                    if not fpath.exists():
                        report.fc_anomalies += 1
                        report.anomalies.append(Anomaly("FC", fname, "exists", "MISSING"))
                        continue
                    report.fc_found += 1
                    try:
                        d = np.load(fpath, allow_pickle=False)
                        mat = d["fc_matrix"]
                    except Exception as e:
                        report.fc_anomalies += 1
                        report.anomalies.append(Anomaly("FC", fname, "load", str(e)[:80]))
                        continue

                    ok = True

                    if mat.shape != (n_labels, n_labels):
                        report.anomalies.append(Anomaly(
                            "FC", fname, "shape",
                            f"{mat.shape} != ({n_labels},{n_labels})"
                        ))
                        ok = False

                    if not np.isfinite(mat).all():
                        report.anomalies.append(Anomaly(
                            "FC", fname, "finite",
                            f"{int(np.sum(~np.isfinite(mat)))} NaN/Inf"
                        ))
                        ok = False
                    else:
                        # Symmetry check
                        sym_err = float(np.abs(mat - mat.T).max())
                        if sym_err > 1e-6:
                            report.anomalies.append(Anomaly(
                                "FC", fname, "symmetry", f"max_err={sym_err:.2e}"
                            ))
                            ok = False

                        # Range checks
                        mask_offdiag = ~np.eye(n_labels, dtype=bool)
                        off = mat[mask_offdiag]
                        med = float(np.median(off))
                        mx = float(np.abs(off).max())

                        if metric == "wpli":
                            if not (0.0 <= mx <= _WPLI_OFFDIAG_MAX_ABS):
                                report.anomalies.append(Anomaly(
                                    "FC", fname, "wpli_range",
                                    f"max_abs={mx:.4f} out of [0,1]"
                                ))
                                ok = False
                            # Store per-sub stats
                            st = stats_map[sub]
                            if cond == "EO":
                                st.wpli_eo_median = med
                                st.wpli_eo_max = mx
                            else:
                                st.wpli_ec_median = med
                                st.wpli_ec_max = mx

                        elif metric == "coh":
                            if not (_COH_MIN <= med <= _COH_MAX) or not (mx <= _COH_MAX + 1e-6):
                                report.anomalies.append(Anomaly(
                                    "FC", fname, "coh_range",
                                    f"median={med:.4f}, max={mx:.4f}"
                                ))
                                ok = False
                            st = stats_map[sub]
                            if cond == "EO":
                                st.coh_eo_median = med
                            else:
                                st.coh_ec_median = med

                    if ok:
                        report.fc_valid += 1
                    else:
                        report.fc_anomalies += 1

    report.subject_stats = list(stats_map.values())


def _validate_features(xs_dir: Path, report: ValidationReport) -> None:
    """Validate feature matrices X_*.npz + y.npy + groups.npy."""
    y_path = xs_dir / "y.npy"
    groups_path = xs_dir / "groups.npy"

    # Validate y and groups first
    y = groups = None
    if not y_path.exists():
        report.feat_anomalies += 1
        report.anomalies.append(Anomaly("features", "y.npy", "exists", "MISSING"))
    else:
        y = np.load(y_path)
        if y.shape != (_N_SAMPLES_EXPECTED,):
            report.feat_anomalies += 1
            report.anomalies.append(Anomaly(
                "features", "y.npy", "shape", f"{y.shape} != ({_N_SAMPLES_EXPECTED},)"
            ))
        unique_y = np.unique(y)
        if not np.array_equal(unique_y, [0, 1]):
            report.feat_anomalies += 1
            report.anomalies.append(Anomaly(
                "features", "y.npy", "values", f"unique={unique_y}"
            ))
        n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
        if n0 != _N_SUBJECTS or n1 != _N_SUBJECTS:
            report.feat_anomalies += 1
            report.anomalies.append(Anomaly(
                "features", "y.npy", "balance", f"class0={n0}, class1={n1} (expect {_N_SUBJECTS} each)"
            ))

    if not groups_path.exists():
        report.feat_anomalies += 1
        report.anomalies.append(Anomaly("features", "groups.npy", "exists", "MISSING"))
    else:
        groups = np.load(groups_path)
        if groups.shape != (_N_SAMPLES_EXPECTED,):
            report.feat_anomalies += 1
            report.anomalies.append(Anomaly(
                "features", "groups.npy", "shape", f"{groups.shape} != ({_N_SAMPLES_EXPECTED},)"
            ))
        if len(np.unique(groups)) != _N_SUBJECTS:
            report.feat_anomalies += 1
            report.anomalies.append(Anomaly(
                "features", "groups.npy", "unique_groups",
                f"{len(np.unique(groups))} != {_N_SUBJECTS}"
            ))

    for atlas in ATLASES:
        for metric in METRICS:
            fname = f"X_{atlas}_{metric}_alpha.npz"
            fpath = xs_dir / fname
            if not fpath.exists():
                report.feat_anomalies += 1
                report.anomalies.append(Anomaly("features", fname, "exists", "MISSING"))
                continue
            report.feat_found += 1
            try:
                X = np.load(fpath, allow_pickle=False)["X"]
            except Exception as e:
                report.feat_anomalies += 1
                report.anomalies.append(Anomaly("features", fname, "load", str(e)[:80]))
                continue

            ok = True
            expected_nfeat = _FEAT_N_LABELS[atlas]

            if X.shape[0] != _N_SAMPLES_EXPECTED:
                report.anomalies.append(Anomaly(
                    "features", fname, "n_samples",
                    f"{X.shape[0]} != {_N_SAMPLES_EXPECTED}"
                ))
                ok = False

            if X.shape[1] != expected_nfeat:
                report.anomalies.append(Anomaly(
                    "features", fname, "n_features",
                    f"{X.shape[1]} != {expected_nfeat}"
                ))
                ok = False

            if not np.isfinite(X).all():
                n_bad = int(np.sum(~np.isfinite(X)))
                report.anomalies.append(Anomaly(
                    "features", fname, "finite", f"{n_bad} NaN/Inf"
                ))
                ok = False

            if ok:
                report.feat_valid += 1
            else:
                report.feat_anomalies += 1


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _render_report(report: ValidationReport) -> str:
    ts = report.timestamp
    verdict = report.verdict
    verdict_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[verdict]

    lines = [
        "# N=15 Outputs — Validation Report",
        "",
        f"**Timestamp**: {ts}  ",
        "**Sprint**: S-VALIDATE-N15  ",
        "**Data root**: `data/{label_ts,connectivity,features}/ds005385/`  ",
        f"**Subjects**: {len(SUBJECTS)} ({', '.join(SUBJECTS[:5])}, ...)",
        "",
        "## Summary",
        "",
        "| Categoria | File attesi | Trovati | Validi | Anomalie |",
        "|-----------|-------------|---------|--------|----------|",
        f"| label_tc per-epoch | {report.lt_expected} | {report.lt_found} | {report.lt_valid} | {report.lt_anomalies} |",
        f"| FC matrices (per-epoch) | {report.fc_expected} | {report.fc_found} | {report.fc_valid} | {report.fc_anomalies} |",
        f"| Feature matrices X | {report.feat_expected} | {report.feat_found} | {report.feat_valid} | {report.feat_anomalies} |",
        "",
        f"**Verdict**: {verdict_icon} **{verdict}**",
        "",
    ]

    if report.anomalies:
        lines += [
            "## Anomalie dettagliate",
            "",
            "| Categoria | File | Campo | Valore |",
            "|-----------|------|-------|--------|",
        ]
        for a in report.anomalies:
            lines.append(f"| {a.category} | `{a.file}` | {a.field} | {a.value} |")
        lines.append("")
    else:
        lines += ["## Anomalie dettagliate", "", "Nessuna anomalia rilevata.", ""]

    # wPLI table
    lines += [
        "## Range fisiologico — wPLI off-diagonal (per sub, atlas=aparc)",
        "",
        "| Sub | EO median | EC median | EO max | EC max |",
        "|-----|-----------|-----------|--------|--------|",
    ]
    for st in report.subject_stats:
        lines.append(
            f"| {st.subject} "
            f"| {st.wpli_eo_median:.4f} "
            f"| {st.wpli_ec_median:.4f} "
            f"| {st.wpli_eo_max:.4f} "
            f"| {st.wpli_ec_max:.4f} |"
        )

    lines += [
        "",
        "## Range — coh off-diagonal (per sub, atlas=aparc)",
        "",
        "| Sub | EO median | EC median |",
        "|-----|-----------|-----------|",
    ]
    for st in report.subject_stats:
        lines.append(
            f"| {st.subject} "
            f"| {st.coh_eo_median:.4f} "
            f"| {st.coh_ec_median:.4f} |"
        )

    lines += [
        "",
        "## Note",
        "",
        "- Validazione read-only — nessuna modifica ai dati.",
        "- label_tc: chiave `label_tc`, shape `(n_epochs, n_labels, n_times)`.",
        "- FC: chiave `fc_matrix`, simmetria verificata (`max|M - M^T| < 1e-6`).",
        "- Features: `X` da npz, `y`/`groups` da `.npy` separati.",
        "- wPLI atteso: off-diagonal in [0, 1], range fisiologico ~[0.05, 0.50].",
        "- coh atteso: off-diagonal in [0, 1].",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(data_root: Path, output: Path) -> int:
    """Run full validation. Returns exit code (0=PASS, 1=WARN, 2=FAIL)."""
    lt_dir = data_root / "data/label_ts/ds005385"
    fc_dir = data_root / "data/connectivity/ds005385"
    xs_dir = data_root / "data/features/ds005385"

    report = ValidationReport(
        timestamp=datetime.datetime.now().astimezone().isoformat()
    )

    _validate_label_tc(lt_dir, report)
    _validate_fc(fc_dir, report)
    _validate_features(xs_dir, report)

    md = _render_report(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(md)

    verdict = report.verdict
    print(f"Validation complete — Verdict: {verdict}")
    print(f"  label_tc : {report.lt_valid}/{report.lt_found} valid ({report.lt_anomalies} anomalies)")
    print(f"  FC       : {report.fc_valid}/{report.fc_found} valid ({report.fc_anomalies} anomalies)")
    print(f"  features : {report.feat_valid}/{report.feat_found} valid ({report.feat_anomalies} anomalies)")
    print(f"  Report   → {output}")

    return {"PASS": 0, "WARN": 1, "FAIL": 2}[verdict]


def main() -> int:
    ap = argparse.ArgumentParser(description="S-VALIDATE-N15 — data integrity check")
    ap.add_argument("--data-root", default=".", type=Path,
                    help="Tesi_2.0 root (default: cwd)")
    ap.add_argument("--output", default="reports/VALIDATE_N15.md", type=Path,
                    help="Output markdown report")
    args = ap.parse_args()
    return validate(args.data_root, args.output)


if __name__ == "__main__":
    sys.exit(main())
