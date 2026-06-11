# Pipeline Overview — Tesi_2.0

Diagramma e dettaglio dei 7 step della pipeline EEG source-level FC + ML.

Diagramma visuale: [figures/architecture.svg](figures/architecture.svg) — [PNG](../reports/figures/fig00_architecture.png)

---

## Diagramma generale

```
dati BIDS (eeg_matchingpennies o dataset scientifico finale)
    |
    v
STEP 1: preprocessing
    mne-bids-pipeline --config config/config_step1_<dataset>.py --steps preprocessing
    output: data/derivatives/mne-bids-pipeline/sub-XX/eeg/*_clean_epo.fif
    |
    v
STEP 2: source reconstruction
    source_reconstruction/finalize_inverse.py --subject XX --task <task>
    output: data/derivatives/mne-bids-pipeline/sub-XX/eeg/*-fwd.fif
             data/derivatives/mne-bids-pipeline/sub-XX/eeg/*-inv.fif
             data/derivatives/mne-bids-pipeline/sub-XX/eeg/*-stc-<cond>-<method>-{lh,rh}.stc
    |
    v
STEP 3: parcellazione atlastica
    parcellation/extract_label_tc.py  (via API: extract_tc_from_files)
    atlas: aparc (68 ROI) | destrieux (148 ROI) | schaefer100 | schaefer200
    output: numpy array (n_labels, n_times) + lista nomi label
    |
    v
STEP 4: connettivita funzionale spettrale
    connectivity/fc_dispatcher.py  (via API: compute_fc)
    metriche: wpli | plv | coh | imcoh | ciplv | pli | wpli2_debiased
    bande: delta(1-4) | theta(4-8) | alpha(8-13) | beta(13-30) | gamma(30-80) Hz
    output: dict { banda -> FC matrix (n_labels, n_labels) }
             salvato: data/derivatives/fc/<sub>/<atlas>_<metric>_<banda>.npz
    |
    v
STEP 5: feature extraction
    features/dispatcher.py
    - mne-features: univariate (mean, variance, kurtosis, hjorth, ...) su label_tc
    - FC flatten: upper triangle -> vettore per ogni banda
    output: X matrix (n_samples, n_features)
    |
    v
STEP 6: classificazione ML
    ml_training/ml_dispatcher.py
    algoritmi: LogReg | SVM | MLP | RF | GB
    validazione: GroupKFold (LOSO, group=subject_id)
    permutation testing: 1000 permutazioni, p-value FDR-corretto
    output: reports/ML_RESULTS_<config>.json  (BA, p-value, coefficienti)
    |
    v
STEP 7: verifica E2E + risultati
    pipeline_mne_bids/run_e2e_matchingpennies.py  (orchestratore step 1-6)
    output: reports/E2E_MATCHINGPENNIES.md
             reports/BENCH_MATRIX_RESULTS.json  (benchmark matrix 7×4×5×5)
```

---

## Dettaglio per step

### STEP 1 — Preprocessing

- **Modulo**: `mne-bids-pipeline` (strumento esterno, non modificato)
- **Config**: `config/config_step1_matchingpennies.py`
- **Input**: `data/eeg_matchingpennies/` (BIDS raw)
- **Output**: `data/derivatives/mne-bids-pipeline/sub-XX/eeg/`
  - `*_task-matchingpennies_proc-clean_epo.fif` — epochs pulite
  - `*_report.html` — report QC
- **Log**: `reports/STEP1_LOG.md`

### STEP 2 — Source Reconstruction

- **Modulo**: `source_reconstruction/finalize_inverse.py`
- **Config**: `config/config_step2_source_matchingpennies.py`
- **Input**: output STEP 1 + `sub-05_fwd.fif` (generato da mne-bids-pipeline --steps sensor)
- **Output**:
  - `sub-XX_task-YY-inv.fif` — inverse operator
  - `sub-XX_task-YY_cond-<name>_inv-dSPM-{lh,rh}.stc` — source estimates per condizione
- **Parametri chiave**: `method=dSPM`, `loose=0.2`, `depth=0.8`, `lambda2=1/9`
- **Log**: `reports/STEP2_LOG.md`

### STEP 3 — Parcellazione

- **Modulo**: `parcellation/extract_label_tc.py`
- **API principale**: `extract_tc_from_files(stc_path, fwd_path, atlas, mode="mean_flip")`
- **Atlasi supportati**:

  | Nome | Parametro | N ROI | File annotazione |
  |------|-----------|-------|-----------------|
  | Desikan-Killiany | `"aparc"` | ~68 | `aparc.annot` |
  | Destrieux | `"destrieux"` | ~148 | `aparc.a2009s.annot` |
  | Schaefer 100 | `"schaefer100"` | 100 | `Schaefer2018_100Parcels_7Networks_order` |
  | Schaefer 200 | `"schaefer200"` | 200 | `Schaefer2018_200Parcels_7Networks_order` |

- **Nota**: atlasi su spazio `fsaverage` (`SUBJECTS_DIR` configurabile via env `TESI_SUBJECTS_DIR`)
- **Modulo opzionale**: `parcellation/neuromaps_helper.py` — mappa annotazioni (gradients, microstructure) a ROI

### STEP 4 — Connettivita Funzionale

- **Modulo**: `connectivity/fc_dispatcher.py`
- **API principale**: `compute_fc(label_tc, sfreq, metric, *, bands, mode, n_jobs)`
- **Input**: array `(n_epochs, n_labels, n_times)`, frequenza campionamento
- **Output**: `dict[str, np.ndarray]` — chiave = nome banda, valore = FC matrix `(n_labels, n_labels)`
- **Persistenza**: `connectivity.save_fc(result, path, labels)` salva `.npz`
- **Dipendenza**: `mne_connectivity.spectral_connectivity_epochs`

### STEP 5 — Feature Extraction

- **Modulo**: `features/dispatcher.py`
- **Input**: label_tc array + FC dict
- **Output**: `X` matrix `(n_samples, n_features)`
- **Componenti**:
  - mne-features univariate su label_tc
  - upper-triangle flatten delle FC matrix per ogni banda/metrica

### STEP 6 — Machine Learning

- **Modulo**: `ml_training/ml_dispatcher.py`
- **Algoritmi**: LogisticRegression, SVM, MLP, RandomForest, GradientBoosting
- **Validazione**: `GroupKFold` con `group=subject_id` (LOSO)
- **Statistica**: permutation test 1000x, FDR correction (scipy)
- **Metrica primaria**: Balanced Accuracy (BA)
- **Output**: JSON per config, `.planning/research/METHODS_v1.md` integra i risultati

### STEP 7 — E2E

- **Modulo**: `pipeline_mne_bids/run_e2e_matchingpennies.py`
- **Scope**: dataset matchingpennies sub-05 (smoke) → dataset scientifico finale (produzione)
- **Output**: `reports/E2E_MATCHINGPENNIES.md`, `reports/BENCH_MATRIX_RESULTS.json`

---

## Config files

| File | Step | Scope |
|------|------|-------|
| `config/config_step1_matchingpennies.py` | 1 | preprocessing (192 parametri mne-bids-pipeline) |
| `config/config_step2_source_matchingpennies.py` | 2 | source reconstruction (soggetto, task, metodo, spazio) |

Config per STEP 3-7 sono definiti nei moduli Python corrispondenti (non richiedono file separati).

---

## Dataset

| Dataset | Ruolo | Soggetti | Canali | Task | Stato |
|---------|-------|----------|--------|------|-------|
| eeg_matchingpennies | Smoke test tecnico | 7 (usato: sub-05) | 61 | matchingpennies | integrato |
| ds005385 (Dortmund Vital) | Dataset scientifico attivo | 200 tot, subset N=15 | 64 | RestingState (EO/EC) | in corso |

ds005385 e' il dataset scientifico scelto (Q1 risolto 2026-05-01). Symlink attivo:
`data/raw/ds005385` → `~/Scrivania/Tesi/data/ds005385/`.
Condizioni: EO (Eyes Open) / EC (Eyes Closed). PILOT su 5 soggetti: sub-007, sub-010, sub-011, sub-026, sub-031.
Cross-ref: `config/subjects_whitelist.py`, `config/labels_ds005385.py`, `reports/DS005385_STRUCTURE.md`.
