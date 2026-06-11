"""Test step 1.7: selettore target nella dashboard — smoke test import."""

from __future__ import annotations

import sys
from pathlib import Path


def test_dashboard_importable():
    """Dashboard app deve essere importabile senza crash (no Streamlit runtime needed)."""
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Streamlit non può girare in pytest, testiamo solo la struttura del file
    app_path = root / "dashboard" / "app.py"
    assert app_path.exists(), "dashboard/app.py non trovato"
    source = app_path.read_text()
    assert "sel_target" in source, "selettore target non trovato in app.py"
    assert "EO/EC (positive-control)" in source
    assert "Sesso" in source
    assert "Età" in source
    assert "ml_sex_results.json" in source
    assert "ml_age_results.json" in source


def test_ml_sex_module_importable():
    from ml_training.ml_sex import load_data, run_cv  # noqa: F401

    assert callable(load_data)
    assert callable(run_cv)


def test_ml_age_module_importable():
    from ml_training.ml_age import load_data, run_cv  # noqa: F401

    assert callable(load_data)
    assert callable(run_cv)
