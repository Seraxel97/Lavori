"""Tests per config/subjects_whitelist_n50.py — H-SCALE-N50."""

from __future__ import annotations

import hashlib
import importlib

from config.subjects_whitelist_n30 import SUBJECT_WHITELIST_N30 as N30
from config.subjects_whitelist_n50 import (
    _NEW_20,
    N50_HASH,
    N_SUBJECTS,
    RANDOM_SEED,
)
from config.subjects_whitelist_n50 import (
    SUBJECT_WHITELIST_N50 as N50,
)


def test_n50_size():
    assert len(N50) == 50
    assert N_SUBJECTS == 50
    assert RANDOM_SEED == 42


def test_n30_subset_n50():
    assert set(N30).issubset(set(N50)), "hard-rule violated: N30 ⊄ N50"


def test_n50_new_20_size():
    assert len(_NEW_20) == 20


def test_n50_no_overlap_with_n30():
    assert not set(_NEW_20) & set(N30), "new_20 overlaps N30"


def test_n50_deterministic():
    import config.subjects_whitelist_n50 as m

    a = list(m.SUBJECT_WHITELIST_N50)
    importlib.reload(m)
    b = list(m.SUBJECT_WHITELIST_N50)
    assert a == b


def test_n50_no_duplicates():
    assert len(N50) == len(set(N50))


def test_n50_hash():
    computed = hashlib.sha256(",".join(N50).encode()).hexdigest()[:16]
    assert computed == N50_HASH, f"hash mismatch: {computed} != {N50_HASH}"


def test_n50_sorted():
    assert N50 == sorted(N50)
