"""
Common configuration for all STEP 1+ processing.

Extracted from config_step1/step2 to avoid duplication.
Contains baseline preprocessing, subject/task, filtering parameters.
Step-specific configs import from here and override as needed.
"""

from common.paths import BIDS_ROOT, DERIV

# ── Percorsi ──────────────────────────────────────────────────────────────
bids_root = BIDS_ROOT
deriv_root = DERIV

# ── Soggetti / task ───────────────────────────────────────────────────────
subjects = ["05"]
task = "matchingpennies"
ch_types = ["eeg"]

# ── Esecuzione ────────────────────────────────────────────────────────────
interactive = False

# ── Filtri / preprocessing base ───────────────────────────────────────────
l_freq = None
h_freq = 100
zapline_fline = 50
zapline_iter = False

# Crop runs ai primi 10 minuti (run originali ≈ 31 min) per velocità in test
crop_runs = (0, 600)

# ── Reject epochs ─────────────────────────────────────────────────────────
reject = {"eeg": 150e-6}

# ── Condizioni / contrasti ────────────────────────────────────────────────
conditions = ["raised-left", "raised-right"]
contrasts = [("raised-left", "raised-right")]

# ── Decoding (built-in mne-bids-pipeline, NON il nostro ML) ──────────────
decode = True

# ── Montage standard EEG 10-20 (per source reconstruction) ─────────────────
eeg_template_montage = "standard_1020"

# ── Group analysis ────────────────────────────────────────────────────────
interpolate_bads_grand_average = False
