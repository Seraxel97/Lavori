"""Test di regressione: features/dispatcher.py NON deve applicare normalizzazione.

Verifica che l'output di build_X abbia media e std raw (non (0,1)),
prevenendo future double-normalization accidentali.
"""

from __future__ import annotations

import numpy as np

from features.dispatcher import build_X


def _make_mock_label_tc(n_epochs: int = 5, n_labels: int = 4, n_times: int = 200) -> np.ndarray:
    """Genera label_tc sintetico con valori non-normalizzati (media≠0, std≠1)."""
    rng = np.random.default_rng(42)
    # offset=10 + scale=5 → media≠0, std≠1 garantiti
    return rng.standard_normal((n_epochs, n_labels, n_times)) * 5 + 10


def test_build_x_output_not_normalized() -> None:
    """build_X NON deve normalizzare: mean raw output non deve essere ≈ 0."""
    label_tc = _make_mock_label_tc()
    X, _ = build_X(label_tc, sfreq=250.0, include_fc=False, include_univariate=True)

    col_means = np.mean(X, axis=0)
    # Almeno alcune colonne devono avere |mean| > 0.1 (non normalizzate)
    assert np.any(np.abs(col_means) > 0.1), (
        "Tutte le colonne hanno mean ≈ 0 — sospetta normalizzazione in features/dispatcher.py"
    )


def test_build_x_std_not_unit() -> None:
    """build_X NON deve standardizzare: std raw output non deve essere ≈ 1 ovunque."""
    label_tc = _make_mock_label_tc()
    X, _ = build_X(label_tc, sfreq=250.0, include_fc=False, include_univariate=True)

    col_stds = np.std(X, axis=0)
    # Almeno alcune colonne devono avere std > 2 o < 0.5 (non standardizzate)
    non_unit = np.sum((col_stds > 2.0) | (col_stds < 0.5))
    assert non_unit > 0, (
        f"Tutte le {len(col_stds)} colonne hanno std ≈ 1 — sospetta normalizzazione in dispatcher"
    )
