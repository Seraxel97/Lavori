# STEP 1 — Setup pipeline mne-bids-pipeline

**Data inizio**: 2026-04-27
**Data approvazione**: 2026-04-28
**Stato**: ✅ DONE — approvato dall'operatore
**Scope**: installazione + dataset + esecuzione preprocessing + verifica end-to-end

## Sintesi esito (riepilogo richiesto post-approvazione)

- ✅ STEP 1 = DONE
- ✅ MNE-BIDS-Pipeline installata correttamente (v1.10.1)
- ✅ Preprocessing test completato (12 step, 0 errori, ~30s wall-clock)
- ✅ Output derivatives generati (`data/derivatives/mne-bids-pipeline/sub-05/eeg/` + report HTML BIDS-standard)
- ✅ Configurabilità nativa analizzata (192 parametri tipizzati, override via Python config)
- 🔧 Estensioni necessarie (NON in pipeline ufficiale, da costruire negli STEP 3-6):
  - atlas / parcellation (FreeSurfer aparc, Schaefer, neuromaps)
  - connectivity (mne-connectivity: wPLI, PLI, COH, PLV, imcoh)
  - features (mne-features)
  - ML custom (sklearn: LogReg, SVM, RF, GB con GroupKFold)

## Vincoli operativi confermati dall'operatore (2026-04-28)

- Path progetto: `~/Scrivania/Tesi_2.0/` (definitivo)
- NO orchestrazioni cross-session (worker paralleli, cron, tmux file-bus, opus-father coordination)
- Lavoro lineare, controllato, manualmente verificabile
- Dataset `eeg_matchingpennies` = solo TEST TECNICO della pipeline, NON dataset scientifico finale
- Dataset scientifico finale (candidato): `ds005385` o equivalente EC/EO resting — uso BLOCCATO finché operatore non conferma esplicitamente

---

## 1. Comandi di installazione eseguiti

```bash
# Conda env: base (Python 3.13.12)
pip install mne-bids-pipeline
```

**Versione installata**: `mne_bids_pipeline 1.10.1`

**Dipendenze già presenti** (no install necessario):
- `mne 1.11.0`
- `mne-bids 0.18.0`
- `mne-connectivity 0.8`
- `mne-features 0.3.2`
- `mne-icalabel 0.8.1`
- `scikit-learn 1.8.0`
- `numpy 2.4.4`, `scipy 1.17.1`

**Dipendenze nuove installate** (top-level): `cyclopts`, `python-picard`, `pyvista`, `pyvistaqt`, `dask`, `distributed`, `bokeh`, `statsmodels`, `meegkit`, `edfio`, `eeglabio`, `pybv`, `jupyter-server`, `nbconvert`, `vtk` e dipendenze transitive.

**Errori**: nessuno.

---

## 2. Dataset scelto

### `eeg_matchingpennies`

- **Fonte**: OSF `https://osf.io/download/8rbfk?version=1`
- **Modalità**: EEG (1 modalità, niente MEG → semplifica preprocessing)
- **Soggetti totali nel pacchetto**: 7 (`sub-05` … `sub-11`); config ufficiale ne usa **1** (`sub-05`)
- **Task**: `matchingpennies` (decoding mano alzata, 2 condizioni: `raised-left` vs `raised-right`)
- **Run length**: ~31 min (croppati a 600s = 10 min in config STEP 1)
- **Dimensione**: 716 MB decompresso (binario `.eeg/.vhdr/.vmrk` BrainVision)
- **Path locale**: `Tesi_2.0/data/eeg_matchingpennies/`

### Perché questo dataset

1. **EEG-only**: la tesi è EEG, non MEG → no maxfilter, no head_pos, niente noise compensation MEG-specifica.
2. **Più piccolo degli examples ufficiali**: solo 1 soggetto richiesto dalla config ufficiale, run da 10 min basta per validare end-to-end.
3. **Config ufficiale già esistente**: `mne_bids_pipeline/tests/configs/config_eeg_matchingpennies.py` su GitHub → riferimento garantito.
4. **Eventi presenti**: ha trigger `raised-left`/`raised-right` → permette di testare `decode=True` (ML built-in). Utile come baseline confronto futuro per il nostro ML su connettività.

### Dataset NON scelti (e motivi)

| Dataset | Motivo esclusione |
|---------|------------------|
| `ds000117` (Faces) | MEG, non EEG, N=19 |
| `ds000246`/`ds000247` (Brainstorm Auditory) | MEG |
| `ds000248` (Auditory) | MEG+EEG, ma config richiede FreeSurfer + BEM → fuori scope STEP 1 |
| `ds001810` (intracranial) | sEEG, non scalp EEG |
| `ds001971`, `ds003775` (tSSS, empty room) | MEG |
| `ds003392` (ERP CORE) | EEG ma N=40 → troppo grande per smoke test |
| `ds004107`, `ds004229` | MEG |
| `funloc`, `ds_phantom_kit` | MEG |

---

## 3. Risultato esecuzione pipeline

### Comando

```bash
cd /home/seraxel/Scrivania/Tesi_2.0
mne_bids_pipeline --config config/config_step1_matchingpennies.py --steps preprocessing
```

### Step eseguiti (12 step preprocessing, ~30s totali)

| Step | Esito | Note |
|------|-------|------|
| `init/_01_init_derivatives_dir` | ✅ done (8s) | Inizializzato HDF5 report |
| `init/_02_find_empty_room` | ⏩ skip | EEG, no MEG empty room |
| `preprocessing/_01_data_quality` | ✅ done (3s) | Bad channels detection |
| `preprocessing/_02_head_pos` | ⏩ skip | Non richiesto |
| `preprocessing/_03_maxfilter` | ⏩ skip | EEG (maxfilter è MEG-only) |
| `preprocessing/_04_frequency_filter` | ✅ done (15s) | Zapline 50Hz + LP 100Hz |
| `preprocessing/_05_regress_artifact` | ⏩ skip | Non richiesto |
| `preprocessing/_06a1_fit_ica` | ⏩ skip | Non richiesto |
| `preprocessing/_06a2_find_ica_artifacts` | ⏩ skip | Non richiesto |
| `preprocessing/_06b_run_ssp` | ⏩ skip | Non richiesto |
| `preprocessing/_07_make_epochs` | ✅ done (2s) | 100 epochs, -0.2 a 0.5s |
| `preprocessing/_08a_apply_ica` | ⏩ skip | Non richiesto |
| `preprocessing/_08b_apply_ssp` | ⏩ skip | Non richiesto |
| `preprocessing/_09_ptp_reject` | ✅ done (2s) | 0/100 rifiutati (soglia 150 µV) |

### Output prodotti in `data/derivatives/mne-bids-pipeline/`

```
sub-05/eeg/
├── sub-05_report.html                            ← report BIDS-standard
├── sub-05_report.h5                              ← report sorgente (HDF5)
├── sub-05_task-matchingpennies_proc-filt_raw.fif ← raw filtrato
├── sub-05_task-matchingpennies_epo.fif           ← epochs
├── sub-05_task-matchingpennies_proc-clean_epo.fif← epochs cleaned (PTP)
├── sub-05_task-matchingpennies_bads.tsv          ← bad channels
└── sub-05_task-matchingpennies_scores.json       ← quality scores
task-matchingpennies_log.xlsx                     ← log step Excel
dataset_description.json                          ← BIDS sidecar
_cache/                                           ← joblib cache (re-run istantaneo)
```

**Dimensione totale derivatives**: 146 MB.

### Errori

Nessuno.

---

## 4. Stato end-to-end (risposta domanda 6 prof)

> *"La pipeline funziona end-to-end?"*

**Risposta corta**: SÌ ma SOLO per quello che mne-bids-pipeline copre nativamente. NON copre i 4 step della tesi nella loro interezza.

### Cosa copre nativamente (4 gruppi di step)

| Gruppo | Step inclusi | Stato testato |
|--------|--------------|----------------|
| `preprocessing` | data quality, filter, head_pos, maxfilter, regress, ICA, SSP, epochs, PTP reject | ✅ STEP 1 |
| `sensor` | evoked, decoding (full + time-by-time), time-frequency, CSP decoding, covariance | ⏳ non testato |
| `source` | BEM surfaces, BEM solution, source space, forward, inverse, group average | ⏳ non testato (richiede FreeSurfer + MRI) |
| `report` | HTML report finale grand-average | ⏳ non testato |

### Cosa NON copre (e quindi va costruito SOPRA gli output)

| Componente brief prof | mne-bids-pipeline lo fa? | Dove va costruito |
|----------------------|--------------------------|-------------------|
| Parcellazione (aparc/Schaefer/neuromaps) | ❌ — produce STC per vertice, non per parcel | STEP 3 → `parcellation/` |
| Connettività tra parcels (wPLI/COH/PLV/imcoh) | ❌ — non integrato | STEP 4 → `connectivity/` (uso `mne-connectivity`) |
| Feature extraction (`mne-features`) | ❌ | STEP 5 → `features/` |
| ML custom (LR/SVM/RF/GB con GroupKFold) | Solo decoding sensor-level built-in | STEP 6 → `ml_training/` (uso sklearn) |
| Atlante neuromaps | ❌ | STEP 3 (decisione: backend) |

**Conclusione**: la pipeline ufficiale è end-to-end **fino al sensor decoding + source reconstruction**. Tutto ciò che riguarda parcels → connettività → ML custom va orchestrato a valle leggendo gli output `.fif` da `derivatives/`. Questo è esattamente l'architettura che il brief prof prevede ("estendere mne-bids-pipeline").

---

## 5. Configurabilità (risposta domanda 7 prof)

### Dove vivono i parametri

- File centrale: `mne_bids_pipeline/_config.py` (192 parametri tipizzati con `Literal` / dataclass).
- Config utente: file Python che fa override per assegnamento (qualsiasi `.py` passato a `--config`).
- No YAML, no TOML — solo Python (vantaggio: validazione a load-time, type hints; svantaggio: meno friendly di YAML per non-developer).

### Parametri rilevanti per il brief prof (tutti già configurabili)

| Categoria | Parametro | Tipo / valori |
|-----------|-----------|---------------|
| **Inverse / source** | `inverse_method` | `"MNE" \| "dSPM" \| "sLORETA" \| "eLORETA"` |
| | `spacing` | `"oct5" \| "oct6" \| "ico4" \| "ico5" \| int` |
| | `noise_cov` / `noise_cov_method` | covariance estimator |
| | `inverse_targets` | `["evoked"]` (NB: per RS bisogna gestire epoch noise cov a parte) |
| **Filtro** | `l_freq`, `h_freq`, `notch_freq`, `zapline_fline` | float / None |
| **ICA** | `ica_l_freq`, `ica_n_components`, `ica_reject` | float / int / dict |
| **Epoch** | `epochs_tmin`, `epochs_tmax`, `baseline`, `reject` | float / dict |
| **Decoding sensor** (built-in, non sostituisce nostro ML) | `decode`, `decoding_metric`, `decoding_n_splits`, `contrasts` | bool / str / int |
| **Channel** | `ch_types`, `eeg_template_montage` | list / str |

### Parametri NON configurabili nativamente (vanno aggiunti a valle)

| Componente brief | Parametro target | Soluzione |
|-----------------|------------------|-----------|
| Atlas (aparc / Schaefer / neuromaps) | nessun param ufficiale | STEP 3: leggere stc dai derivatives e applicare `mne.extract_label_time_course` con label set custom |
| Metrica connettività (wPLI/PLI/COH/PLV) | nessun param | STEP 4: `mne-connectivity` chiamato a valle |
| Classificatori ML custom | `decode=True` ma solo LogReg sui sensor | STEP 6: sklearn pipeline custom su feature parcellate |

### Verdetto configurabilità

**FACILE** per: filter, ICA, SSP, epochs, reject, contrasts, decoding sensor, BEM, source space, inverse method.

**MEDIO** per: source/parcellation custom (richiede script post-pipeline che legga `inverse_*.h5` o ricalcoli STC da `epochs_*.fif`).

**FUORI-SCOPE pipeline ufficiale** (da costruire noi negli step 2-6): parcellazione, connettività, mne-features, ML custom.

---

## 6. Decisioni prese in STEP 1

1. **Conda env**: `base` (non creo env nuovo — tutto già funzionante).
2. **Dataset**: `eeg_matchingpennies` (1 sub, EEG, smoke test minimale).
3. **Config Python (no YAML)** — coerente con la convenzione ufficiale; YAML→Python wrapper rimandato a STEP 2 se serve.
4. **Crop a 600s**: ridotto da 31 min nominali a 10 min reali per velocità STEP 1.
5. **Step eseguito**: solo `preprocessing` (vincolo brief: NON sensor/source/connectivity ora).
6. **Path derivatives dentro `Tesi_2.0/data/derivatives/`**: NON in `~/mne_data/` come da default → tutto isolato nel repo.
7. **No FreeSurfer setup**: STEP 1 non richiede MRI → rimando a STEP 2.

## 7. Problemi incontrati

Nessun problema bloccante. Note minori:

- Download da OSF è 330 MB compressi → 716 MB decompressi (più del previsto, ma è l'unica fonte ufficiale).
- Il pacchetto pip `mne-bids-pipeline` NON include i file `tests/configs/*.py`: ho dovuto recuperare la config ufficiale `config_eeg_matchingpennies.py` direttamente dal repo GitHub via `curl` su `raw.githubusercontent.com`.

## 8. Prossimi step (NON eseguiti — STEP 1 si ferma qui)

Quando l'operatore confermerà:
- **STEP 2**: Setup FreeSurfer (subjects_dir = fsaverage), abilitare `--steps source` su matchingpennies.
- **STEP 3**: Parcellazione — decisione tra (a) FreeSurfer aparc/Schaefer via `mne.extract_label_time_course`, (b) backend neuromaps.
- **STEP 4**: Connettività via `mne-connectivity` (non ancora richiamata).
- **STEP 5**: Feature extraction via `mne-features`.
- **STEP 6**: ML custom sklearn.

## 9. Output richiesti (riepilogo)

| Output richiesto | Sezione |
|------------------|---------|
| Comandi installazione | §1 |
| Dataset scelto | §2 |
| Risultato esecuzione pipeline | §3 |
| Stato end-to-end | §4 |
| Valutazione configurabilità | §5 |

---

**STEP 1 terminato. In attesa di conferma operatore prima di STEP 2.**
