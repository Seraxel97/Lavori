"""S-44: test run output schema standardization."""

from __future__ import annotations

from common.run_schema import build_run_output, validate_run_schema

_MINIMAL_MANIFEST = {
    "run_id": "test_run",
    "timestamp": "2026-05-01T05:00:00+00:00",
    "python_version": "3.13.12",
}

_FULL_VALID = {
    "run_id": "e2e_sub-05_aparc_wpli",
    "timestamp": "2026-05-01T05:00:00+00:00",
    "manifest": {
        **_MINIMAL_MANIFEST,
        "mne_version": "1.11.0",
        "numpy_version": "2.4.4",
        "sklearn_version": "1.8.0",
        "git_sha": "abc123",
        "config_hash": None,
        "seeds": {"random_state": 42, "numpy_seed": 42},
        "env_pkgs": {"mne": "1.11.0"},
    },
    "steps": [
        {"step_name": "load_epochs", "status": "ok", "elapsed_s": 1.2},
        {"step_name": "compute_fc", "status": "ok", "elapsed_s": 0.5},
        {"step_name": "ml_training", "status": "ok", "elapsed_s": 3.1},
    ],
    "errors": [],
    "summary": {"ba_mean_logreg": 0.62, "n_features": 340},
}


def test_valid_schema() -> None:
    """Pipeline output valido → validate_run_schema ritorna lista vuota."""
    errors = validate_run_schema(_FULL_VALID)
    assert errors == [], "Run valido rigettato:\n" + "\n".join(errors)


def test_invalid_schema_missing_field() -> None:
    """run_id mancante → errore di schema rilevato."""
    bad = {k: v for k, v in _FULL_VALID.items() if k != "run_id"}
    errors = validate_run_schema(bad)
    assert errors, "Attesi errori con run_id mancante"
    joined = " ".join(errors).lower()
    assert "run_id" in joined, f"Errore non menziona 'run_id': {errors}"


def test_partial_run_allowed() -> None:
    """Run con errors non vuoto e steps parziali → ancora valido (no all-or-nothing)."""
    partial = {
        "run_id": "partial_run_001",
        "timestamp": "2026-05-01T06:00:00+00:00",
        "manifest": _MINIMAL_MANIFEST,
        "steps": [{"step_name": "load_epochs", "status": "ok"}],
        "errors": ["compute_fc: banda delta ignorata (cicli < 5)"],
        "summary": {},
    }
    errors = validate_run_schema(partial)
    assert errors == [], (
        "Run parziale (errors non vuoto) rigettato — schema troppo strict:\n"
        + "\n".join(errors)
    )


def test_build_run_output_valid() -> None:
    """build_run_output produce un dict conforme allo schema."""
    from common.reproducibility import build_manifest
    manifest = build_manifest("schema_test_run")
    run = build_run_output(
        "schema_test_run", manifest,
        steps=[{"step_name": "smoke", "status": "ok"}],
        summary={"ba_mean": 0.5},
    )
    errors = validate_run_schema(run)
    assert errors == [], "build_run_output non conforme:\n" + "\n".join(errors)
