"""Estrazione y_age e y_sex da participants.tsv ds004504.

ds004504 usa colonne 'Gender' (F/M) e 'Age' (int) invece di 'sex'/'age'.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

DS004504_TSV = Path("data/raw/ds004504/participants.tsv")
SEX_MAP = {"F": 0, "M": 1}


def load_phenotype_ds004504(
    subjects: list[str],
    tsv_path: Path = DS004504_TSV,
) -> tuple[np.ndarray, np.ndarray]:
    """Ritorna (y_age float64, y_sex int64) allineati a subjects."""
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    df = df.set_index("participant_id")
    missing = [s for s in subjects if s not in df.index]
    if missing:
        raise ValueError(f"Soggetti mancanti: {missing}")
    sub_df = df.loc[subjects]
    y_age = sub_df["Age"].astype(float).to_numpy(dtype=np.float64)
    y_sex = sub_df["Gender"].map(SEX_MAP).to_numpy(dtype=np.int64)
    if np.any(np.isnan(y_age)):
        raise ValueError("NaN in y_age")
    if np.any(pd.isna(sub_df["Gender"])):
        raise ValueError("NaN in y_sex (Gender)")
    return y_age, y_sex
