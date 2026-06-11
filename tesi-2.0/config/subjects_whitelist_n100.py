"""Subjects whitelist N=100 ds005385 — estensione di N=50 (seed=43, hard-rule N=50 ⊂ N=100).

Sprint: FASE 1 step 1.6 (orch-tesi, 2026-05-28)
Policy: random_seed_43_N50_subset_N100
Pool: 179 validi (200 totali - 21 late_ses1 > 0)
Criterio: N=50 (whitelist esistente) + 50 nuovi da random.sample(extra_pool_129, 50) con seed=43

Vincolo operatore (hard-rule): N=50 ⊂ N=100 — tutti i sub N50 inclusi, intra-cohort riproducibile.
Derivazione: extra_pool = sorted(valid_179 - N50); rng=random.Random(43); new_50=sorted(rng.sample(extra_pool_129, 50))
"""

from __future__ import annotations

from config.subjects_whitelist_n50 import SUBJECT_WHITELIST_N50 as _N50

SUBSET_POLICY = "random_seed_43_N50_subset_N100"
N_SUBJECTS = 100
RANDOM_SEED = 43

_NEW_50: list[str] = [
    "sub-003", "sub-019", "sub-020", "sub-021", "sub-022",
    "sub-025", "sub-028", "sub-030", "sub-035", "sub-039",
    "sub-044", "sub-045", "sub-051", "sub-053", "sub-067",
    "sub-092", "sub-094", "sub-096", "sub-098", "sub-103",
    "sub-109", "sub-110", "sub-116", "sub-117", "sub-118",
    "sub-123", "sub-127", "sub-128", "sub-130", "sub-131",
    "sub-133", "sub-135", "sub-141", "sub-142", "sub-145",
    "sub-154", "sub-155", "sub-156", "sub-160", "sub-168",
    "sub-172", "sub-173", "sub-178", "sub-180", "sub-182",
    "sub-184", "sub-190", "sub-191", "sub-197", "sub-198",
]

SUBJECT_WHITELIST_N100: list[str] = sorted(set(_N50) | set(_NEW_50))

assert len(SUBJECT_WHITELIST_N100) == 100, f"expected 100, got {len(SUBJECT_WHITELIST_N100)}"
assert set(_N50).issubset(set(SUBJECT_WHITELIST_N100)), "hard-rule violated: N50 ⊄ N100"
assert not set(_NEW_50) & set(_N50), "overlap tra N50 e NEW_50"

DEFAULT_SESSION = "ses-1"
DEFAULT_ACQ = "acq-pre"
POOL_SIZE = 179
POOL_EXCLUDED = 21
POOL_TOTAL = 200
