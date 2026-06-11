"""Tests per config/subjects_whitelist_n100.py — H-SCALE-N100."""

from __future__ import annotations

import importlib

from config.subjects_whitelist_n50 import SUBJECT_WHITELIST_N50 as N50
from config.subjects_whitelist_n100 import (
    _NEW_50,
    N_SUBJECTS,
    RANDOM_SEED,
)
from config.subjects_whitelist_n100 import (
    SUBJECT_WHITELIST_N100 as N100,
)


def test_n100_size():
    assert len(N100) == 100
    assert N_SUBJECTS == 100
    assert RANDOM_SEED == 43


def test_n50_subset_n100():
    assert set(N50).issubset(set(N100)), "hard-rule violated: N50 ⊄ N100"


def test_n100_new_50_size():
    assert len(_NEW_50) == 50


def test_n100_no_overlap_with_n50():
    assert not set(_NEW_50) & set(N50), "new_50 overlaps N50"


def test_n100_deterministic():
    import config.subjects_whitelist_n100 as m

    a = list(m.SUBJECT_WHITELIST_N100)
    importlib.reload(m)
    b = list(m.SUBJECT_WHITELIST_N100)
    assert a == b


def test_n100_no_duplicates():
    assert len(N100) == len(set(N100))


def test_n100_sorted():
    assert N100 == sorted(N100)
