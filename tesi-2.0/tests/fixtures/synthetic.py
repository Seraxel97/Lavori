"""Shared synthetic fixtures for all module tests.

Fornisce dati numpy deterministici (seed=42) senza dipendere da dati reali.
Usata da test che non richiedono dataset matchingpennies.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Generatore deterministico condiviso (seed=42, session-scoped)."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_label_tc(rng: np.random.Generator) -> np.ndarray:
    """Label time courses sintetici: shape (n_epochs=10, n_labels=20, n_times=1000)."""
    return rng.standard_normal((10, 20, 1000))


@pytest.fixture
def synthetic_fc(rng: np.random.Generator) -> np.ndarray:
    """Matrice FC 20x20 simmetrica sintetica."""
    m = rng.standard_normal((20, 20))
    return (m + m.T) / 2


@pytest.fixture
def synthetic_X_y_groups(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dataset ML sintetico: X (50, 100), y binaria (50,), groups per 10 soggetti (50,)."""
    X = rng.standard_normal((50, 100))
    y = rng.integers(0, 2, 50)
    groups = np.repeat(np.arange(10), 5)
    return X, y, groups
