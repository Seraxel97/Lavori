"""Wrapper subprocess su mne_bids_pipeline con timeout configurabile.

Implementa DoS protection: limita il tempo di esecuzione della pipeline
per evitare hang indefiniti in ambienti di test/CI.

Exit code 124 (standard POSIX timeout code) se timeout scaduto.
"""

from __future__ import annotations

import subprocess
import sys

DEFAULT_TIMEOUT_SEC = 30 * 60  # 30 minuti


def run(
    config_path: str,
    steps: str = "preprocessing",
    timeout: int = DEFAULT_TIMEOUT_SEC,
) -> int:
    """Run mne_bids_pipeline with timeout.

    Parameters
    ----------
    config_path : str
        Path to mne_bids_pipeline config file.
    steps : str, optional
        Steps to run (default: "preprocessing").
    timeout : int, optional
        Timeout in seconds (default: 30 min = 1800 sec).

    Returns
    -------
    int
        Exit code: 0 on success, 124 on timeout, other codes from pipeline.
    """
    cmd = ["mne_bids_pipeline", "--config", config_path, "--steps", steps]

    try:
        result = subprocess.run(cmd, timeout=timeout, check=False)
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"Pipeline timeout after {timeout}s", file=sys.stderr)
        return 124  # POSIX timeout exit code
