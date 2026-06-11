"""Configurazione dataset LEMON per Tesi_2.0.

LEMON (Leipzig Mind-Brain-Body Dataset): dataset pubblico EEG a riposo
con paradigma EC/EO, compatibile con pipeline ds005385.

Ref: Babayan et al. (2019), Scientific Data.
DOI: https://doi.org/10.1038/sdata.2018.308
Download: https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-Dataset/

STATO: SCAFFOLDING — dati non ancora integrati.
Prerequisiti operatore (gate bloccante):
    1. Download LEMON BIDS da URL sopra (~200GB)
    2. Eseguire STEP 1-5 pipeline Tesi_2.0 su LEMON
    3. Verificare sfreq_target=500Hz + parcellazione aparc/schaefer100
    4. Aggiornare LEMON_RAW_PATH qui sotto con il path reale
"""

from __future__ import annotations

from pathlib import Path

# ── Path operatore (da configurare) ──────────────────────────────────────────
# BLOCCO OPERATORE: impostare il path reale dopo download LEMON BIDS.
LEMON_RAW_PATH: Path | None = None  # es. Path("/home/seraxel/Scrivania/LEMON_BIDS")

# ── Configurazione pipeline ───────────────────────────────────────────────────
DATASET_ID = "lemon"
SFREQ_TARGET = 500  # Hz — stesso di ds005385
PARADIGM = "EC_EO"  # Eyes Closed / Eyes Open

# Pilot whitelist (15 subjects, seed=42) — da aggiornare con sub-ID LEMON reali
# NOTA: sub-ID LEMON seguono formato "sub-010001", "sub-010002", etc.
PILOT_N = 15
PILOT_SEED = 42

# Label mapping EC/EO per LEMON
# LEMON usa "EC" e "EO" nei task names BIDS
LABEL_MAP = {
    "EC": 0,  # Eyes Closed → classe 0
    "EO": 1,  # Eyes Open → classe 1
}

# Atlanti supportati (stessi di ds005385)
ATLASES = ["aparc", "schaefer100"]
