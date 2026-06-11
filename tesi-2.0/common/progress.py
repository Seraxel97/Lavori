"""Progress bar wrapper con ETA per pipeline steps."""

from __future__ import annotations

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def progress(iterable, total=None, desc=""):
    """Wrap iterable with progress bar if tqdm available.

    Parameters
    ----------
    iterable
        Any iterable to wrap.
    total : int, optional
        Total number of iterations (for tqdm).
    desc : str, optional
        Description string for progress bar.

    Returns
    -------
    iterable
        tqdm-wrapped iterable if tqdm available, otherwise original iterable.
    """
    if _HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc, ncols=80)
    return iterable
