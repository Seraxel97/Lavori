"""Estrazione y_age e y_sex da participants.tsv (ds005385)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DS005385_TSV = Path("/home/seraxel/Scrivania/Tesi/data/ds005385/participants.tsv")

SEX_MAP = {"F": 0, "M": 1}


def load_phenotype(
    subjects: list[str],
    tsv_path: Path = DS005385_TSV,
) -> tuple[np.ndarray, np.ndarray]:
    """Estrai y_age e y_sex allineati a `subjects`.

    Parameters
    ----------
    subjects:
        Lista ordinata di subject ID (es. ["sub-001", "sub-002", ...]).
    tsv_path:
        Path a participants.tsv BIDS.

    Returns
    -------
    y_age : np.ndarray float64, shape (n,)
    y_sex : np.ndarray int64, shape (n,)  — F→0, M→1

    Raises
    ------
    ValueError
        Se soggetti mancanti nel TSV o valori non validi.
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    df = df.set_index("participant_id")

    missing = [s for s in subjects if s not in df.index]
    if missing:
        raise ValueError(f"Soggetti mancanti nel TSV: {missing}")

    sub_df = df.loc[subjects]

    y_age = sub_df["age"].astype(float).to_numpy(dtype=np.float64)
    y_sex = sub_df["sex"].map(SEX_MAP).to_numpy(dtype=np.int64)

    if np.any(np.isnan(y_age)):
        raise ValueError("NaN in y_age")
    if np.any(np.isnan(y_sex.astype(float))):
        raise ValueError("NaN in y_sex")

    return y_age, y_sex
