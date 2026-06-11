"""Label mapping ds005385 EO vs EC — Dortmund Vital Study.

Derivato da analisi BIDS struttura (sprint S-100, 2026-05-01).

Dataset: ds005385 (OpenNeuro, doi:10.18112/openneuro.ds005385.v1.0.3)
N soggetti: 200 (179 validi, 21 esclusi per late_ses1 > 0)
Formato: EDF, 64 ch EEG, 1000 Hz, ref FCz, ~183s per recording

EO/EC sono codificati nel task name BIDS — nessuna ambiguità:
  task-EyesOpen   → EO (label 0)
  task-EyesClosed → EC (label 1)
"""

from __future__ import annotations

LABEL_EO = "eyes-open"
LABEL_EC = "eyes-closed"

# BIDS task names → condizione
TASK_TO_CONDITION: dict[str, str] = {
    "EyesOpen": "EO",
    "EyesClosed": "EC",
}

# Condizione → label numerica per ML
CONDITION_TO_INT: dict[str, int] = {
    "EO": 0,
    "EC": 1,
}

# Mapping derivato da events.tsv analysis (sprint S-100)
# Chiave: stringa task+acq come appare nel filename BIDS
CONDITION_MAP: dict[str, list[str]] = {
    "EO": [
        "task-EyesOpen_acq-pre",   # baseline pre-cognitivo (PRIMARIO)
        "task-EyesOpen_acq-post",  # post-cognitivo (opzionale, ~202 file)
    ],
    "EC": [
        "task-EyesClosed_acq-pre",  # baseline pre-cognitivo (PRIMARIO)
        "task-EyesClosed_acq-post", # post-cognitivo (opzionale)
    ],
}

# Acquisizione raccomandata per pilot S-101
DEFAULT_ACQ = "acq-pre"

# Sessione raccomandata (massima copertura)
DEFAULT_SESSION = "ses-1"

# Soggetti con late triggers ses-1 — da escludere (dati probabilmente non continui)
# Fonte: participants.tsv colonna late_ses1 > 0
EXCLUDE_LATE_SES1: list[str] = [
    "sub-008", "sub-009", "sub-013", "sub-027", "sub-037",
    "sub-049", "sub-052", "sub-055", "sub-075", "sub-085",
    "sub-093", "sub-097", "sub-106", "sub-108", "sub-114",
    "sub-115", "sub-138", "sub-170", "sub-176", "sub-179",
    "sub-183",
]

# Soggetti validi ses-1 (200 - 21 late = 179)
N_VALID_SES1 = 179

# Parametri EEG
SFREQ_ORIGINAL = 1000      # Hz (BrainAmp DC)
SFREQ_TARGET = 250         # Hz (dopo decimazione 1000/4)
N_CHANNELS_EEG = 64        # EEG channels (10-20, EasyCap actiCAP 64)
REFERENCE = "FCz"
POWERLINE_FREQ = 50        # Hz (Europa)
RECORDING_DURATION_S = 183 # secondi (~3 minuti per recording)
