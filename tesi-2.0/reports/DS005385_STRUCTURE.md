# Dataset Analysis — ds005385 (Dortmund Vital Study)

**Timestamp**: 2026-05-01T11:35:00Z  
**Sprint**: S-100 (sonnet1-ts)  
**Source**: `data/raw/ds005385/` → symlink `~/Scrivania/Tesi/data/ds005385/`  
**DOI**: 10.18112/openneuro.ds005385.v1.0.3  
**License**: CC0

---

## 1. Dataset overview

| Parametro | Valore |
|-----------|--------|
| Nome | Dortmund Vital Study — Resting-state EEG |
| Autori | Wascher, Schneider, Gajewski, Getzmann (IfADo, Dortmund) |
| BIDS version | 1.9.0 |
| Soggetti scaricati | **200** (sub-001 → sub-200) |
| Soggetti nel TSV completo | 608 (studio longitudinale completo) |
| Sessioni | ses-1 (2017), ses-2 (2022, follow-up ~5 anni) |
| Formato EEG | EDF (BrainProducts BrainAmp DC, EasyCap actiCAP 64) |
| Task | `task-EyesOpen`, `task-EyesClosed` |
| Acquisizioni | `acq-pre` (pre-cognitivo), `acq-post` (post-cognitivo, ~2h) |

---

## 2. Struttura BIDS

```
ds005385/
├── participants.tsv          (608 righe, 200 presenti nel download)
├── sub-001/
│   ├── sub-001_sessions.tsv  (session_id, recording_year)
│   ├── ses-1/
│   │   └── eeg/
│   │       ├── sub-001_ses-1_task-EyesOpen_acq-pre_eeg.edf
│   │       ├── sub-001_ses-1_task-EyesOpen_acq-pre_eeg.json
│   │       ├── sub-001_ses-1_task-EyesOpen_acq-pre_events.tsv
│   │       ├── sub-001_ses-1_task-EyesOpen_acq-pre_channels.tsv
│   │       ├── sub-001_ses-1_task-EyesClosed_acq-pre_eeg.edf
│   │       └── ...
│   └── ses-2/               (solo 51 soggetti con follow-up)
│       └── eeg/ ...
└── sub-200/ ...
```

### Distribuzione sessioni

| Sessione | Soggetti | Anno |
|----------|----------|------|
| ses-1 only | **149** | 2017 |
| ses-1 + ses-2 | **51** | 2017 + 2022 |

---

## 3. Parametri EEG

| Parametro | Valore |
|-----------|--------|
| Campionamento (sfreq) | **1000 Hz** |
| Canali EEG | **64** (10-20 scheme, EasyCap actiCAP 64) |
| Reference | **FCz** |
| Ground | AFz |
| Power line | 50 Hz (Europa) |
| Durata per recording | ~183–184 s (~3 minuti) |
| Formato file | EDF (continuo) |
| EOG channels | 0 (non separati) |

### Canali (64 EEG, esempio)
`Fp1, Fp2, F7, F3, Fz, F4, F8, ...` — montaggio 10-20 standard.

---

## 4. Inventario file EDF

| Task + Acquisizione | N file EDF | Note |
|--------------------|-----------|------|
| `task-EyesOpen_acq-pre` | 251 | 200 ses-1 + 51 ses-2 |
| `task-EyesClosed_acq-pre` | 251 | 200 ses-1 + 51 ses-2 |
| `task-EyesOpen_acq-post` | 202 | solo ses-1+ses-2 con follow-up |
| `task-EyesClosed_acq-post` | 202 | solo ses-1+ses-2 con follow-up |
| **TOTALE** | **906** | |

> **Nota**: `acq-post` = resting-state registrato dopo ~2h di batteria neurocognitiva.
> Per la pipeline Tesi_2.0 si useranno inizialmente **solo `acq-pre`** (baseline resting-state).

---

## 5. EO/EC label mapping

Il dataset encode EO/EC **direttamente nel task name** BIDS — nessuna ambiguità:

| Task BIDS | Condizione | Label |
|-----------|------------|-------|
| `task-EyesOpen` | Eyes Open (EO) | `"EO"` |
| `task-EyesClosed` | Eyes Closed (EC) | `"EC"` |

### Events TSV
I file `events.tsv` contengono solo trigger `boundary` (onset=1, duration=n/a):
```
onset  duration  type
1      n/a       boundary
```
Le registrazioni sono **resting-state puro** — nessuna struttura a trial. La classificazione
EO vs EC si basa esclusivamente sul filename (task name), non sugli eventi.

### CONDITION_MAP per pipeline

```python
CONDITION_MAP = {
    "EO": [
        "task-EyesOpen_acq-pre",
        "task-EyesOpen_acq-post",   # opzionale, da gating S-101
    ],
    "EC": [
        "task-EyesClosed_acq-pre",
        "task-EyesClosed_acq-post", # opzionale
    ],
}
```

**Default consigliato per S-101 pilot**: `acq-pre` only (baseline, più presente).

---

## 6. Demografiche soggetti (200 presenti)

| Parametro | Valore |
|-----------|--------|
| N | 200 |
| Età media ± SD | 44.7 ± 14.8 anni |
| Età range | [20, 70] anni |
| Sesso F/M | 129 F / 71 M |
| Mancini | ~20% (Edinburgh Handedness Inventory) |

---

## 7. Quality flags

| Flag | N soggetti | Dettaglio |
|------|------------|-----------|
| Late triggers ses-1 | **21** | `late_ses1 > 0` in participants.tsv — dati probabilmente non continui |
| Late triggers ses-2 | **2** | `late_ses2 > 0` |
| Totale flaggati | **23** | Da escludere o esaminare in preprocessing |

**Soggetti con late triggers ses-1 (esempi)**: sub-008 (1901 late), sub-009 (2080), sub-013 (1364), sub-027 (15), sub-037 (445).

> **Raccomandazione**: escludere i 21 soggetti con `late_ses1 > 0` per il pilot S-101.
> Questo porta a **179 soggetti validi ses-1** con dati continui affidabili.

---

## 8. Decision per pipeline Tesi_2.0

| Aspetto | Decisione |
|---------|-----------|
| Task primario | `task-EyesOpen` + `task-EyesClosed` |
| Acquisizione | `acq-pre` (baseline, tutti 200 sub) |
| Sessione | `ses-1` (massima copertura) |
| Soggetti validi (no late trigger) | **179** (escl. 21 con late_ses1 > 0) |
| Formato input | EDF → `mne.io.read_raw_edf()` |
| Sfreq post-decimate target | 250 Hz (1000/4) |
| Classificazione target | EO (0) vs EC (1) — binaria bilanciata |
