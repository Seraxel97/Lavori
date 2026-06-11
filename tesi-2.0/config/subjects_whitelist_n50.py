"""Subjects whitelist N=50 ds005385 — estensione di N=30 (seed=42, hard-rule N=30 ⊂ N=50).

Sprint: H-SCALE-N50 (sonnet-tesi-1, 2026-05-13)
Policy: random_seed_42_N30_subset_N50
Pool: 179 validi (200 totali - 21 late_ses1 > 0)
Criterio: N=30 (whitelist esistente) + 20 nuovi da random.sample(extra_pool_149, 20) con seed=42

Vincolo operatore (hard-rule): N=30 ⊂ N=50 — tutti i sub N30 inclusi, intra-cohort riproducibile.
Derivazione: extra_pool = sorted(valid_179 - N30); rng=random.Random(42); new_20=sorted(rng.sample(extra_pool, 20))
"""

from __future__ import annotations

from config.subjects_whitelist_n30 import SUBJECT_WHITELIST_N30 as _N30

SUBSET_POLICY = "random_seed_42_N30_subset_N50"
N_SUBJECTS = 50
RANDOM_SEED = 42

_NEW_20: list[str] = [
    "sub-016",
    "sub-017",
    "sub-018",
    "sub-042",
    "sub-043",
    "sub-046",
    "sub-050",
    "sub-059",
    "sub-078",
    "sub-086",
    "sub-087",
    "sub-088",
    "sub-090",
    "sub-095",
    "sub-104",
    "sub-148",
    "sub-149",
    "sub-174",
    "sub-189",
    "sub-193",
]

SUBJECT_WHITELIST_N50: list[str] = sorted(set(_N30) | set(_NEW_20))

assert len(SUBJECT_WHITELIST_N50) == 50, f"expected 50, got {len(SUBJECT_WHITELIST_N50)}"
assert set(_N30).issubset(set(SUBJECT_WHITELIST_N50)), "hard-rule violated: N30 ⊄ N50"

DEFAULT_SESSION = "ses-1"
DEFAULT_ACQ = "acq-pre"
POOL_SIZE = 179
POOL_EXCLUDED = 21
POOL_TOTAL = 200
N50_HASH = "aa0c35423aef86c0"  # sha256(sorted N50 joined by comma)[:16]
