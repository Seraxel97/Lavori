"""Test step 1.1: labels_phenotype - y_age e y_sex da participants.tsv."""

from __future__ import annotations

import numpy as np
import pytest

from config.labels_phenotype import load_phenotype
from config.subjects_whitelist_n50 import SUBJECT_WHITELIST_N50


@pytest.fixture(scope="module")
def phenotype():
    return load_phenotype(SUBJECT_WHITELIST_N50)


def test_len(phenotype):
    y_age, y_sex = phenotype
    assert len(y_age) == 50
    assert len(y_sex) == 50


def test_no_nan_age(phenotype):
    y_age, _ = phenotype
    assert not np.any(np.isnan(y_age))


def test_no_nan_sex(phenotype):
    _, y_sex = phenotype
    assert not np.any(np.isnan(y_sex.astype(float)))


def test_age_range(phenotype):
    y_age, _ = phenotype
    assert y_age.min() >= 18.0
    assert y_age.max() <= 100.0


def test_sex_binary(phenotype):
    _, y_sex = phenotype
    assert set(y_sex.tolist()).issubset({0, 1})


def test_dtypes(phenotype):
    y_age, y_sex = phenotype
    assert y_age.dtype == np.float64
    assert y_sex.dtype == np.int64
