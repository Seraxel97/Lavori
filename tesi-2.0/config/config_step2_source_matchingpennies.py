"""
STEP 2 — Matchingpennies EEG, source reconstruction su fsaverage template.

Eredita la config STEP 1 (preprocessing, già validato) e aggiunge i parametri
per BEM + forward + inverse. Niente parcellazione, niente connettività.
Smoke test della source reconstruction end-to-end.
"""

# ── Percorsi ──────────────────────────────────────────────────────────────
bids_root = "/home/seraxel/Scrivania/Tesi_2.0/data/eeg_matchingpennies"
deriv_root = "/home/seraxel/Scrivania/Tesi_2.0/data/derivatives/mne-bids-pipeline"
subjects_dir = "/home/seraxel/mne_data/MNE-fsaverage-data"

# ── Soggetti / task ───────────────────────────────────────────────────────
subjects = ["05"]
task = "matchingpennies"
ch_types = ["eeg"]

# ── Montage: i canali (FC5, FC1, C3, CP5, CP1, FC2, FC6, C4, CP2, CP6) sono
# nomi 10-20 standard ma il BIDS dataset non include posizioni → applichiamo il
# template standard_1020 prima del forward computation. (Necessario per source recon.)
eeg_template_montage = "standard_1020"

# ── Esecuzione ────────────────────────────────────────────────────────────
interactive = False

# ── Filtri / preprocessing base (eredita STEP 1) ──────────────────────────
l_freq = None
h_freq = 100
zapline_fline = 50
zapline_iter = False
crop_runs = (0, 600)

# ── Reject epochs ─────────────────────────────────────────────────────────
reject = {"eeg": 150e-6}

# ── Condizioni / contrasti ────────────────────────────────────────────────
conditions = ["raised-left", "raised-right"]
contrasts = [("raised-left", "raised-right")]

# ── Decoding sensor (built-in) ────────────────────────────────────────────
decode = True

# ── Group analysis ────────────────────────────────────────────────────────
interpolate_bads_grand_average = False

# ── Source reconstruction (NUOVI parametri STEP 2) ────────────────────────
run_source_estimation = True
use_template_mri = "fsaverage"

inverse_method = "dSPM"
spacing = "oct6"
loose = 0.2
depth = 0.8

noise_cov = "ad-hoc"

inverse_targets = ["evoked"]
