"""S-FIG-N15 — Smoke test for generate_figures.py.

Verifies that the full figure pipeline runs without errors on a minimal
synthetic comparison matrix (both PILOT and N=15 schemas).
"""

from __future__ import annotations

import json

import numpy as np
import pytest


@pytest.fixture()
def pilot_comparison(tmp_path):
    """Minimal PILOT-schema comparison_matrix.json."""
    data = {
        "description": "test n=5",
        "note": "synthetic",
        "matrix": {
            "aparc": {
                "wpli": {"logreg": {"bal_acc": 0.7, "acc": 0.7, "auc": 0.8, "f1": 0.65}},
                "coh": {"logreg": {"bal_acc": 0.8, "acc": 0.8, "auc": 0.9, "f1": 0.75}},
            }
        },
    }
    p = tmp_path / "comparison_matrix.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def n15_comparison(tmp_path):
    """Minimal N=15-schema comparison_matrix.json."""
    data = {
        "_hashes": {},
        "_meta": {"band": "alpha", "n_permutations": 100, "n_subjects": 5},
        "results": [
            {
                "atlas": "aparc", "metric": "wpli", "band": "alpha",
                "classifier": "logreg", "ba_mean": 0.75, "ba_std": 0.05,
                "ci_lo": 0.65, "ci_hi": 0.85, "p_perm": 0.02,
                "n_subjects": 5, "n_features": 100,
            },
            {
                "atlas": "aparc", "metric": "coh", "band": "alpha",
                "classifier": "lda", "ba_mean": 0.80, "ba_std": 0.04,
                "ci_lo": 0.70, "ci_hi": 0.90, "p_perm": 0.01,
                "n_subjects": 5, "n_features": 100,
            },
        ],
        "winner": {
            "atlas": "aparc", "metric": "coh", "band": "alpha",
            "classifier": "lda", "ba_mean": 0.80, "ba_std": 0.04,
            "ci_lo": 0.70, "ci_hi": 0.90, "p_perm": 0.01,
            "n_subjects": 5, "n_features": 100,
        },
    }
    p = tmp_path / "comparison_matrix_N15.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def synthetic_features(tmp_path):
    """Minimal feature dir: X_aparc_coh_alpha.npz + y.npy + groups.npy."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 50))
    y = np.array([0, 1] * 5)
    groups = np.repeat(np.arange(5), 2)
    np.savez_compressed(tmp_path / "X_aparc_coh_alpha.npz", X=X)
    np.save(tmp_path / "y.npy", y)
    np.save(tmp_path / "groups.npy", groups)
    # Also create schaefer100 for completeness
    X2 = rng.standard_normal((10, 80))
    np.savez_compressed(tmp_path / "X_schaefer100_coh_alpha.npz", X=X2)
    return tmp_path


def test_smoke_pilot_schema(pilot_comparison, synthetic_features, tmp_path):
    """Full pipeline on PILOT schema: 6 PNG files produced."""
    from scripts.generate_figures import run

    out_dir = tmp_path / "figs"
    ret = run(
        comparison=pilot_comparison,
        features_dir=synthetic_features,
        fc_dir=tmp_path,  # no FC files → fig_fc_avg handles gracefully
        out_dir=out_dir,
        force=True,
    )
    assert ret == 0
    # At least bar, heatmap, pperm, summary table must exist
    for fname in ["fig_balacc_bar.png", "fig_balacc_heatmap.png",
                  "fig_pperm_bar.png", "fig_summary_table.png"]:
        assert (out_dir / fname).exists(), f"Missing: {fname}"


def test_smoke_n15_schema(n15_comparison, synthetic_features, tmp_path):
    """Full pipeline on N=15 schema: CI bars and p_perm rendered."""
    from scripts.generate_figures import run

    out_dir = tmp_path / "figs_n15"
    ret = run(
        comparison=n15_comparison,
        features_dir=synthetic_features,
        fc_dir=tmp_path,
        out_dir=out_dir,
        force=True,
    )
    assert ret == 0
    assert (out_dir / "fig_pperm_bar.png").exists()
    assert (out_dir / "fig_balacc_bar.png").exists()


def test_idempotent(pilot_comparison, synthetic_features, tmp_path):
    """Second run without --force skips all figures."""
    from scripts.generate_figures import run

    out_dir = tmp_path / "figs_idem"
    run(comparison=pilot_comparison, features_dir=synthetic_features,
        fc_dir=tmp_path, out_dir=out_dir, force=True)
    mtimes_before = {f.name: f.stat().st_mtime for f in out_dir.glob("*.png")}

    run(comparison=pilot_comparison, features_dir=synthetic_features,
        fc_dir=tmp_path, out_dir=out_dir, force=False)
    for fname, mtime in mtimes_before.items():
        assert (out_dir / fname).stat().st_mtime == mtime, f"{fname} was modified on second run"


def test_load_comparison_pilot(pilot_comparison):
    """Parser returns ConfigResult list with bal_acc populated."""
    from scripts.generate_figures import load_comparison

    cd = load_comparison(pilot_comparison)
    assert cd.schema == "pilot"
    assert len(cd.results) >= 1
    assert all(0.0 <= r.bal_acc <= 1.0 for r in cd.results)


def test_load_comparison_n15(n15_comparison):
    """Parser returns results with CI and p_perm."""
    from scripts.generate_figures import load_comparison

    cd = load_comparison(n15_comparison)
    assert cd.schema == "n15"
    assert len(cd.results) == 2
    assert all(not np.isnan(r.p_perm) for r in cd.results)
    assert cd.winner is not None
    assert cd.winner.bal_acc == 0.80
