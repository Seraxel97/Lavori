"""Percorsi di progetto centralizzati — Tesi_2.0.

Tutti i path sono configurabili via env var con default relativi al repo.
Variabili d'ambiente:
  TESI_BIDS_ROOT     : root dataset BIDS (default: <repo>/data/eeg_matchingpennies)
  TESI_DERIV         : root derivatives mne-bids-pipeline (default: <repo>/data/derivatives/...)
  TESI_SUBJECTS_DIR  : directory fsaverage (default: ~/mne_data/MNE-fsaverage-data)
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent


def _validate_env_path(env_var: str, default: Path, must_exist: bool = True) -> Path:
    """Validate environment variable path for security.

    Parameters
    ----------
    env_var : str
        Environment variable name.
    default : Path
        Default path if env_var not set.
    must_exist : bool, optional
        If True, path must exist (default: True).

    Returns
    -------
    Path
        Validated absolute path.

    Raises
    ------
    ValueError
        If path is not absolute.
    FileNotFoundError
        If must_exist=True and path does not exist.
    """
    raw = os.environ.get(env_var, str(default))
    p = Path(raw).expanduser()

    if not p.is_absolute():
        raise ValueError(f"{env_var} must be absolute path, got: {p}")

    p = p.resolve()
    if must_exist and not p.exists():
        raise FileNotFoundError(f"{env_var} not found: {p}")

    return p


BIDS_ROOT = _validate_env_path(
    "TESI_BIDS_ROOT",
    _REPO_ROOT / "data" / "eeg_matchingpennies",
    must_exist=True,
)

DERIV = _validate_env_path(
    "TESI_DERIV",
    _REPO_ROOT / "data" / "derivatives" / "mne-bids-pipeline",
    must_exist=True,
)

SUBJECTS_DIR = _validate_env_path(
    "TESI_SUBJECTS_DIR",
    Path.home() / "mne_data" / "MNE-fsaverage-data",
    must_exist=True,
)
