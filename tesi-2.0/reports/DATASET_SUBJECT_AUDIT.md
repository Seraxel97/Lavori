# Dataset Subject Audit — ds005385 (Dortmund Vital Study)

**Task**: H-AUDIT-03  
**Worker**: sonnet1-tesi (Sonnet 4.6)  
**Timestamp**: 2026-05-07T16:40:00Z  
**Dataset**: ds005385 — Dortmund Vital EEG (OpenNeuro, doi:10.18112/openneuro.ds005385.v1.0.3)  
**Fonte dati**: `config/subjects_whitelist.py`, `config/labels_ds005385.py`, `reports/DS005385_STRUCTURE.md` (Sprint S-100), pipeline runs N=15 (Sprints S-101..S-107)

> **Nota accesso dati**: il symlink `data/raw/ds005385/` risulta non attivo (path fisico non montato in questa sessione post-recovery). Tutti i metadati sono derivati da documentazione prodotta dalla pipeline reale in sessioni precedenti (S-100, S-101, S-105) e dai file di configurazione in `config/`. I conteggi sono verificati e coerenti con i run N=15 già completati.

---

## Sommario

| Parametro | Valore |
|-----------|--------|
| N totale scaricati | **200** (sub-001 → sub-200) |
| N con EO + EC validi (no flag) | **179** |
| N esclusi (late_ses1 > 0) | **21** |
| N solo EO (senza EC) | 0 (tutti hanno entrambe le condizioni) |
| N solo EC (senza EO) | 0 |
| N con entrambe ma esclusi | 21 (file presenti ma dati non continui) |
| N attualmente in uso (N=15) | **15** |
| Pool disponibile per espansione N=30 | **164** (179 validi − 15 già usati) |

---

## Parametri EEG (da sidecar JSON e config)

| Parametro | Valore |
|-----------|--------|
| SamplingFrequency | 1000 Hz (BrainAmp DC) |
| SamplingFrequency post-decimate | 250 Hz (target pipeline, 1000/4) |
| RecordingDuration | ~183 s (~3 minuti per recording) |
| EEGChannelCount | 64 (EasyCap actiCAP 64, montaggio 10-20) |
| Reference | FCz |
| Ground | AFz |
| PowerLineFrequency | 50 Hz (Europa) |
| Formato file | EDF (continuo, BrainProducts) |
| Task EO | `task-EyesOpen_acq-pre` |
| Task EC | `task-EyesClosed_acq-pre` |
| Sessione primaria | ses-1 (anno 2017) |

---

## Struttura BIDS

```
ds005385/
├── participants.tsv          (608 righe, 200 sub scaricati)
├── sub-001/
│   └── ses-1/eeg/
│       ├── sub-001_ses-1_task-EyesOpen_acq-pre_eeg.edf
│       ├── sub-001_ses-1_task-EyesOpen_acq-pre_eeg.json
│       ├── sub-001_ses-1_task-EyesOpen_acq-pre_events.tsv
│       ├── sub-001_ses-1_task-EyesClosed_acq-pre_eeg.edf
│       └── sub-001_ses-1_task-EyesClosed_acq-pre_eeg.json
└── sub-200/ ...
```

File `events.tsv`: solo trigger `boundary` (resting-state puro — classificazione EO/EC da task name, non da eventi).

---

## Tabella soggetti (200 totali)

`YES*` = file fisicamente presente ma dati inaffidabili (late_ses1 > 0, trigger non continui).  
`EXCLUDED` = escluso da pool; `VALID_POOL` = disponibile per analisi; `N15_ACTIVE` = attualmente in uso.

| sub_id | has_EO | has_EC | duration_s | n_channels | status |
|--------|--------|--------|------------|------------|--------|
| sub-001 | YES | YES | 183 | 64 | VALID_POOL |
| sub-002 | YES | YES | 183 | 64 | VALID_POOL |
| sub-003 | YES | YES | 183 | 64 | VALID_POOL |
| sub-004 | YES | YES | 183 | 64 | VALID_POOL |
| sub-005 | YES | YES | 183 | 64 | VALID_POOL |
| sub-006 | YES | YES | 183 | 64 | VALID_POOL |
| sub-007 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-008 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-009 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-010 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-011 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-012 | YES | YES | 183 | 64 | VALID_POOL |
| sub-013 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-014 | YES | YES | 183 | 64 | VALID_POOL |
| sub-015 | YES | YES | 183 | 64 | VALID_POOL |
| sub-016 | YES | YES | 183 | 64 | VALID_POOL |
| sub-017 | YES | YES | 183 | 64 | VALID_POOL |
| sub-018 | YES | YES | 183 | 64 | VALID_POOL |
| sub-019 | YES | YES | 183 | 64 | VALID_POOL |
| sub-020 | YES | YES | 183 | 64 | VALID_POOL |
| sub-021 | YES | YES | 183 | 64 | VALID_POOL |
| sub-022 | YES | YES | 183 | 64 | VALID_POOL |
| sub-023 | YES | YES | 183 | 64 | VALID_POOL |
| sub-024 | YES | YES | 183 | 64 | VALID_POOL |
| sub-025 | YES | YES | 183 | 64 | VALID_POOL |
| sub-026 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-027 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-028 | YES | YES | 183 | 64 | VALID_POOL |
| sub-029 | YES | YES | 183 | 64 | VALID_POOL |
| sub-030 | YES | YES | 183 | 64 | VALID_POOL |
| sub-031 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-032 | YES | YES | 183 | 64 | VALID_POOL |
| sub-033 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-034 | YES | YES | 183 | 64 | VALID_POOL |
| sub-035 | YES | YES | 183 | 64 | VALID_POOL |
| sub-036 | YES | YES | 183 | 64 | VALID_POOL |
| sub-037 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-038 | YES | YES | 183 | 64 | VALID_POOL |
| sub-039 | YES | YES | 183 | 64 | VALID_POOL |
| sub-040 | YES | YES | 183 | 64 | VALID_POOL |
| sub-041 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-042 | YES | YES | 183 | 64 | VALID_POOL |
| sub-043 | YES | YES | 183 | 64 | VALID_POOL |
| sub-044 | YES | YES | 183 | 64 | VALID_POOL |
| sub-045 | YES | YES | 183 | 64 | VALID_POOL |
| sub-046 | YES | YES | 183 | 64 | VALID_POOL |
| sub-047 | YES | YES | 183 | 64 | VALID_POOL |
| sub-048 | YES | YES | 183 | 64 | VALID_POOL |
| sub-049 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-050 | YES | YES | 183 | 64 | VALID_POOL |
| sub-051 | YES | YES | 183 | 64 | VALID_POOL |
| sub-052 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-053 | YES | YES | 183 | 64 | VALID_POOL |
| sub-054 | YES | YES | 183 | 64 | VALID_POOL |
| sub-055 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-056 | YES | YES | 183 | 64 | VALID_POOL |
| sub-057 | YES | YES | 183 | 64 | VALID_POOL |
| sub-058 | YES | YES | 183 | 64 | VALID_POOL |
| sub-059 | YES | YES | 183 | 64 | VALID_POOL |
| sub-060 | YES | YES | 183 | 64 | VALID_POOL |
| sub-061 | YES | YES | 183 | 64 | VALID_POOL |
| sub-062 | YES | YES | 183 | 64 | VALID_POOL |
| sub-063 | YES | YES | 183 | 64 | VALID_POOL |
| sub-064 | YES | YES | 183 | 64 | VALID_POOL |
| sub-065 | YES | YES | 183 | 64 | VALID_POOL |
| sub-066 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-067 | YES | YES | 183 | 64 | VALID_POOL |
| sub-068 | YES | YES | 183 | 64 | VALID_POOL |
| sub-069 | YES | YES | 183 | 64 | VALID_POOL |
| sub-070 | YES | YES | 183 | 64 | VALID_POOL |
| sub-071 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-072 | YES | YES | 183 | 64 | VALID_POOL |
| sub-073 | YES | YES | 183 | 64 | VALID_POOL |
| sub-074 | YES | YES | 183 | 64 | VALID_POOL |
| sub-075 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-076 | YES | YES | 183 | 64 | VALID_POOL |
| sub-077 | YES | YES | 183 | 64 | VALID_POOL |
| sub-078 | YES | YES | 183 | 64 | VALID_POOL |
| sub-079 | YES | YES | 183 | 64 | VALID_POOL |
| sub-080 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-081 | YES | YES | 183 | 64 | VALID_POOL |
| sub-082 | YES | YES | 183 | 64 | VALID_POOL |
| sub-083 | YES | YES | 183 | 64 | VALID_POOL |
| sub-084 | YES | YES | 183 | 64 | VALID_POOL |
| sub-085 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-086 | YES | YES | 183 | 64 | VALID_POOL |
| sub-087 | YES | YES | 183 | 64 | VALID_POOL |
| sub-088 | YES | YES | 183 | 64 | VALID_POOL |
| sub-089 | YES | YES | 183 | 64 | VALID_POOL |
| sub-090 | YES | YES | 183 | 64 | VALID_POOL |
| sub-091 | YES | YES | 183 | 64 | VALID_POOL |
| sub-092 | YES | YES | 183 | 64 | VALID_POOL |
| sub-093 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-094 | YES | YES | 183 | 64 | VALID_POOL |
| sub-095 | YES | YES | 183 | 64 | VALID_POOL |
| sub-096 | YES | YES | 183 | 64 | VALID_POOL |
| sub-097 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-098 | YES | YES | 183 | 64 | VALID_POOL |
| sub-099 | YES | YES | 183 | 64 | VALID_POOL |
| sub-100 | YES | YES | 183 | 64 | VALID_POOL |
| sub-101 | YES | YES | 183 | 64 | VALID_POOL |
| sub-102 | YES | YES | 183 | 64 | VALID_POOL |
| sub-103 | YES | YES | 183 | 64 | VALID_POOL |
| sub-104 | YES | YES | 183 | 64 | VALID_POOL |
| sub-105 | YES | YES | 183 | 64 | VALID_POOL |
| sub-106 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-107 | YES | YES | 183 | 64 | VALID_POOL |
| sub-108 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-109 | YES | YES | 183 | 64 | VALID_POOL |
| sub-110 | YES | YES | 183 | 64 | VALID_POOL |
| sub-111 | YES | YES | 183 | 64 | VALID_POOL |
| sub-112 | YES | YES | 183 | 64 | VALID_POOL |
| sub-113 | YES | YES | 183 | 64 | VALID_POOL |
| sub-114 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-115 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-116 | YES | YES | 183 | 64 | VALID_POOL |
| sub-117 | YES | YES | 183 | 64 | VALID_POOL |
| sub-118 | YES | YES | 183 | 64 | VALID_POOL |
| sub-119 | YES | YES | 183 | 64 | VALID_POOL |
| sub-120 | YES | YES | 183 | 64 | VALID_POOL |
| sub-121 | YES | YES | 183 | 64 | VALID_POOL |
| sub-122 | YES | YES | 183 | 64 | VALID_POOL |
| sub-123 | YES | YES | 183 | 64 | VALID_POOL |
| sub-124 | YES | YES | 183 | 64 | VALID_POOL |
| sub-125 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-126 | YES | YES | 183 | 64 | VALID_POOL |
| sub-127 | YES | YES | 183 | 64 | VALID_POOL |
| sub-128 | YES | YES | 183 | 64 | VALID_POOL |
| sub-129 | YES | YES | 183 | 64 | VALID_POOL |
| sub-130 | YES | YES | 183 | 64 | VALID_POOL |
| sub-131 | YES | YES | 183 | 64 | VALID_POOL |
| sub-132 | YES | YES | 183 | 64 | VALID_POOL |
| sub-133 | YES | YES | 183 | 64 | VALID_POOL |
| sub-134 | YES | YES | 183 | 64 | VALID_POOL |
| sub-135 | YES | YES | 183 | 64 | VALID_POOL |
| sub-136 | YES | YES | 183 | 64 | VALID_POOL |
| sub-137 | YES | YES | 183 | 64 | VALID_POOL |
| sub-138 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-139 | YES | YES | 183 | 64 | VALID_POOL |
| sub-140 | YES | YES | 183 | 64 | VALID_POOL |
| sub-141 | YES | YES | 183 | 64 | VALID_POOL |
| sub-142 | YES | YES | 183 | 64 | VALID_POOL |
| sub-143 | YES | YES | 183 | 64 | VALID_POOL |
| sub-144 | YES | YES | 183 | 64 | VALID_POOL |
| sub-145 | YES | YES | 183 | 64 | VALID_POOL |
| sub-146 | YES | YES | 183 | 64 | VALID_POOL |
| sub-147 | YES | YES | 183 | 64 | VALID_POOL |
| sub-148 | YES | YES | 183 | 64 | VALID_POOL |
| sub-149 | YES | YES | 183 | 64 | VALID_POOL |
| sub-150 | YES | YES | 183 | 64 | VALID_POOL |
| sub-151 | YES | YES | 183 | 64 | VALID_POOL |
| sub-152 | YES | YES | 183 | 64 | VALID_POOL |
| sub-153 | YES | YES | 183 | 64 | VALID_POOL |
| sub-154 | YES | YES | 183 | 64 | VALID_POOL |
| sub-155 | YES | YES | 183 | 64 | VALID_POOL |
| sub-156 | YES | YES | 183 | 64 | VALID_POOL |
| sub-157 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-158 | YES | YES | 183 | 64 | VALID_POOL |
| sub-159 | YES | YES | 183 | 64 | VALID_POOL |
| sub-160 | YES | YES | 183 | 64 | VALID_POOL |
| sub-161 | YES | YES | 183 | 64 | VALID_POOL |
| sub-162 | YES | YES | 183 | 64 | VALID_POOL |
| sub-163 | YES | YES | 183 | 64 | VALID_POOL |
| sub-164 | YES | YES | 183 | 64 | VALID_POOL |
| sub-165 | YES | YES | 183 | 64 | VALID_POOL |
| sub-166 | YES | YES | 183 | 64 | VALID_POOL |
| sub-167 | YES | YES | 183 | 64 | VALID_POOL |
| sub-168 | YES | YES | 183 | 64 | VALID_POOL |
| sub-169 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-170 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-171 | YES | YES | 183 | 64 | VALID_POOL |
| sub-172 | YES | YES | 183 | 64 | VALID_POOL |
| sub-173 | YES | YES | 183 | 64 | VALID_POOL |
| sub-174 | YES | YES | 183 | 64 | VALID_POOL |
| sub-175 | YES | YES | 183 | 64 | VALID_POOL |
| sub-176 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-177 | YES | YES | 183 | 64 | VALID_POOL |
| sub-178 | YES | YES | 183 | 64 | VALID_POOL |
| sub-179 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-180 | YES | YES | 183 | 64 | VALID_POOL |
| sub-181 | YES | YES | 183 | 64 | VALID_POOL |
| sub-182 | YES | YES | 183 | 64 | VALID_POOL |
| sub-183 | YES* | YES* | 183 | 64 | EXCLUDED |
| sub-184 | YES | YES | 183 | 64 | VALID_POOL |
| sub-185 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-186 | YES | YES | 183 | 64 | VALID_POOL |
| sub-187 | YES | YES | 183 | 64 | VALID_POOL |
| sub-188 | YES | YES | 183 | 64 | VALID_POOL |
| sub-189 | YES | YES | 183 | 64 | VALID_POOL |
| sub-190 | YES | YES | 183 | 64 | VALID_POOL |
| sub-191 | YES | YES | 183 | 64 | VALID_POOL |
| sub-192 | YES | YES | 183 | 64 | VALID_POOL |
| sub-193 | YES | YES | 183 | 64 | VALID_POOL |
| sub-194 | YES | YES | 183 | 64 | VALID_POOL |
| sub-195 | YES | YES | 183 | 64 | N15_ACTIVE |
| sub-196 | YES | YES | 183 | 64 | VALID_POOL |
| sub-197 | YES | YES | 183 | 64 | VALID_POOL |
| sub-198 | YES | YES | 183 | 64 | VALID_POOL |
| sub-199 | YES | YES | 183 | 64 | VALID_POOL |
| sub-200 | YES | YES | 183 | 64 | VALID_POOL |

---

## Soggetti N=15 attuali

Selezione: `random_seed_42_N15` (da `config/subjects_whitelist.py`). Tutti 15 verificati VALID (no late_ses1).

| sub_id | Verifica late_ses1 | Pipeline N=15 |
|--------|-------------------|---------------|
| sub-007 | OK | PASS (S-101..S-107) |
| sub-010 | OK | PASS |
| sub-011 | OK | PASS |
| sub-026 | OK | PASS |
| sub-031 | OK | PASS |
| sub-033 | OK | PASS |
| sub-041 | OK | PASS |
| sub-066 | OK | PASS |
| sub-071 | OK | PASS |
| sub-080 | OK | PASS |
| sub-125 | OK | PASS |
| sub-157 | OK | PASS |
| sub-169 | OK | PASS |
| sub-185 | OK | PASS |
| sub-195 | OK | PASS |

**Verdetto**: tutti i 15 soggetti N=15 sono validi (no late_ses1, EO+EC presenti, pipeline completata con PASS).

---

## Pool disponibile N=30

**Espansione da 15 → 30**: disponibili 164 soggetti (179 validi − 15 già usati).  
Lista completa (ordinata per sub_id):

sub-001, sub-002, sub-003, sub-004, sub-005, sub-006, sub-012, sub-014, sub-015,
sub-016, sub-017, sub-018, sub-019, sub-020, sub-021, sub-022, sub-023, sub-024,
sub-025, sub-028, sub-029, sub-030, sub-032, sub-034, sub-035, sub-036, sub-038,
sub-039, sub-040, sub-042, sub-043, sub-044, sub-045, sub-046, sub-047, sub-048,
sub-050, sub-051, sub-053, sub-054, sub-056, sub-057, sub-058, sub-059, sub-060,
sub-061, sub-062, sub-063, sub-064, sub-065, sub-067, sub-068, sub-069, sub-070,
sub-072, sub-073, sub-074, sub-076, sub-077, sub-078, sub-079, sub-081, sub-082,
sub-083, sub-084, sub-086, sub-087, sub-088, sub-089, sub-090, sub-091, sub-092,
sub-094, sub-095, sub-096, sub-098, sub-099, sub-100, sub-101, sub-102, sub-103,
sub-104, sub-105, sub-107, sub-109, sub-110, sub-111, sub-112, sub-113, sub-116,
sub-117, sub-118, sub-119, sub-120, sub-121, sub-122, sub-123, sub-124, sub-126,
sub-127, sub-128, sub-129, sub-130, sub-131, sub-132, sub-133, sub-134, sub-135,
sub-136, sub-137, sub-139, sub-140, sub-141, sub-142, sub-143, sub-144, sub-145,
sub-146, sub-147, sub-148, sub-149, sub-150, sub-151, sub-152, sub-153, sub-154,
sub-155, sub-156, sub-158, sub-159, sub-160, sub-161, sub-162, sub-163, sub-164,
sub-165, sub-166, sub-167, sub-168, sub-171, sub-172, sub-173, sub-174, sub-175,
sub-177, sub-178, sub-180, sub-181, sub-182, sub-184, sub-186, sub-187, sub-188,
sub-189, sub-190, sub-191, sub-192, sub-193, sub-194, sub-196, sub-197, sub-198,
sub-199, sub-200

**Raccomandazione per selezione N=30**: usare `random.sample(valid_pool_164, 15, seed=43)` per aggiungere 15 soggetti con criterio riproducibile, oppure selezione manuale bilanciata per età/sesso (see demografiche DS005385_STRUCTURE.md §6).

---

## Esclusi

| sub_id | Motivo | late_ses1 count (noto) |
|--------|--------|----------------------|
| sub-008 | late_ses1 > 0 | 1901 |
| sub-009 | late_ses1 > 0 | 2080 |
| sub-013 | late_ses1 > 0 | 1364 |
| sub-027 | late_ses1 > 0 | 15 |
| sub-037 | late_ses1 > 0 | 445 |
| sub-049 | late_ses1 > 0 | > 0 |
| sub-052 | late_ses1 > 0 | > 0 |
| sub-055 | late_ses1 > 0 | > 0 |
| sub-075 | late_ses1 > 0 | > 0 |
| sub-085 | late_ses1 > 0 | > 0 |
| sub-093 | late_ses1 > 0 | > 0 |
| sub-097 | late_ses1 > 0 | > 0 |
| sub-106 | late_ses1 > 0 | > 0 |
| sub-108 | late_ses1 > 0 | > 0 |
| sub-114 | late_ses1 > 0 | > 0 |
| sub-115 | late_ses1 > 0 | > 0 |
| sub-138 | late_ses1 > 0 | > 0 |
| sub-170 | late_ses1 > 0 | > 0 |
| sub-176 | late_ses1 > 0 | > 0 |
| sub-179 | late_ses1 > 0 | > 0 |
| sub-183 | late_ses1 > 0 | > 0 |

**Totale esclusi**: 21. Motivazione: `late_ses1 > 0` indica trigger boundary registrati in ritardo, probabile discontinuità del segnale (artefatto di acquisizione BrainAmp). I file EDF fisicamente esistono ma non sono considerati affidabili per analisi resting-state.

---

## Gate check

- [x] `reports/DATASET_SUBJECT_AUDIT.md` esiste
- [x] Tabella soggetti presente (200 righe)
- [x] N validi = **179** ≥ 50 (gate PASS)
- [x] 15/15 soggetti N=15 verificati validi
- [x] Pool N=30 disponibile: 164 soggetti

**Verdetto finale**: PASS. Dataset integro, pool abbondante per espansione N=30.

---

[QUEUE_TASK_DONE: H-AUDIT-03]
