"""Subjects whitelist N=179 ds005385 — full local valid (hard-rule N=100 ⊂ N=179).

Sprint: FASE 3 step 3.1 (orch-tesi, 2026-05-29)
Policy: full_local_valid_late_ses1_zero
Pool: 200 scaricati localmente in data/raw/ds005385/, 179 con late_ses1==0
Criterio: tutti i sub scaricati con late_ses1 == 0 in participants.tsv

Vincolo operatore (hard-rule): N=100 ⊂ N=179 — tutti i sub N100 inclusi, intra-cohort riproducibile.
Derivazione: valid_179 = {sid : sid in os.listdir(data/raw/ds005385) and late_ses1==0}
            new_79 = sorted(valid_179 - N100)
"""

from __future__ import annotations

from config.subjects_whitelist_n100 import SUBJECT_WHITELIST_N100 as _N100

SUBSET_POLICY = "full_local_valid_late_ses1_zero"
N_SUBJECTS = 179

_NEW_79: list[str] = [
    "sub-001", "sub-002", "sub-004", "sub-005", "sub-006",
    "sub-023", "sub-024", "sub-029", "sub-036", "sub-047",
    "sub-054", "sub-056", "sub-057", "sub-058", "sub-060",
    "sub-061", "sub-062", "sub-063", "sub-064", "sub-065",
    "sub-068", "sub-069", "sub-070", "sub-072", "sub-073",
    "sub-074", "sub-077", "sub-079", "sub-081", "sub-083",
    "sub-084", "sub-089", "sub-099", "sub-100", "sub-101",
    "sub-102", "sub-105", "sub-107", "sub-111", "sub-112",
    "sub-113", "sub-119", "sub-120", "sub-121", "sub-122",
    "sub-124", "sub-126", "sub-129", "sub-132", "sub-134",
    "sub-137", "sub-139", "sub-140", "sub-143", "sub-144",
    "sub-146", "sub-147", "sub-150", "sub-151", "sub-152",
    "sub-153", "sub-158", "sub-159", "sub-161", "sub-162",
    "sub-163", "sub-164", "sub-165", "sub-166", "sub-167",
    "sub-175", "sub-177", "sub-181", "sub-186", "sub-188",
    "sub-192", "sub-194", "sub-196", "sub-199",
]

SUBJECT_WHITELIST_N179: list[str] = sorted(set(_N100) | set(_NEW_79))

assert len(SUBJECT_WHITELIST_N179) == 179, f"expected 179, got {len(SUBJECT_WHITELIST_N179)}"
assert set(_N100).issubset(set(SUBJECT_WHITELIST_N179)), "hard-rule violated: N100 ⊄ N179"
assert not set(_NEW_79) & set(_N100), "overlap tra N100 e NEW_79"
assert len(_NEW_79) == 79, f"expected 79 new, got {len(_NEW_79)}"

DEFAULT_SESSION = "ses-1"
DEFAULT_ACQ = "acq-pre"
POOL_SIZE = 179
POOL_EXCLUDED = 21
POOL_TOTAL = 200
