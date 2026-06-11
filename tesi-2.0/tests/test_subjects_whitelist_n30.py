"""Tests per config/subjects_whitelist_n30.py — H-SCALE-01."""

from __future__ import annotations

import hashlib
import importlib

from config.subjects_whitelist import SUBJECT_WHITELIST as N15
from config.subjects_whitelist_n30 import (
    _NEW_15,
    N30_HASH,
    N_SUBJECTS,
    RANDOM_SEED,
)
from config.subjects_whitelist_n30 import (
    SUBJECT_WHITELIST_N30 as N30,
)


def test_n30_size():
    assert len(N30) == 30
    assert N_SUBJECTS == 30
    assert RANDOM_SEED == 42


def test_n15_subset_n30():
    assert set(N15).issubset(set(N30)), "hard-rule violated: N15 ⊄ N30"


def test_n30_deterministic():
    import config.subjects_whitelist_n30 as m

    a = list(m.SUBJECT_WHITELIST_N30)
    importlib.reload(m)
    b = list(m.SUBJECT_WHITELIST_N30)
    assert a == b


def test_n30_no_duplicates():
    assert len(N30) == len(set(N30))


def test_new_15_size():
    assert len(_NEW_15) == 15
    assert len(set(_NEW_15)) == 15


def test_n15_and_new15_disjoint():
    assert set(N15).isdisjoint(set(_NEW_15)), "overlap tra N15 e _NEW_15"


def test_n30_hash():
    h = hashlib.sha256(",".join(N30).encode()).hexdigest()[:16]
    assert h == N30_HASH, f"N30 hash mismatch: {h} != {N30_HASH}"


def test_n30_sorted():
    assert N30 == sorted(N30)


def test_n30_valid_format():
    for s in N30:
        assert s.startswith("sub-"), f"invalid subject id: {s}"
        num = s.split("-")[1]
        assert num.isdigit() and len(num) == 3, f"invalid sub id format: {s}"
