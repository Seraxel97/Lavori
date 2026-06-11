# Features Audit — H-AUDIT-01

**Date**: 2026-05-07
**Auditor**: opus1-tesi (Opus 4.7 — complex worker)
**HEAD**: 3c01904 (post-recovery, repo restored)
**Scope**: tracciare ESATTAMENTE quali feature entrano nel modello ML finale della pipeline N=15.

**Source files audited**
- `features/dispatcher.py` (univariate + FC flatten API)
- `features/__init__.py` (modulo doc)
- `connectivity/fc_dispatcher.py` (compute_fc, metriche)
- `config/subjects_whitelist.py` (N=15, seed=42)
- `config/labels_ds005385.py` (etichette EO/EC, sfreq target 250 Hz, 64 ch EEG)
- `scripts/run_pipeline_n15.py` (Step 5 reale: build_X locale, NON usa features/dispatcher.build_X)
- `ml_training/aggregate_n15.py` (Step 6+7 LOSO + permutation)
- `reports/EXPERIMENTS_N15.md` (winner = aparc×coh×theta×svm_rbf, n_features=2278)

---

## 1. Panoramica

La code-base contiene DUE percorsi di feature distinti:

1. **API completa** in `features/dispatcher.py` → `build_X(label_tc, sfreq, fc, ...)`:
   univariate (mne-features) + FC flatten su tutte le bande passate, una sola matrice X.
2. **Pipeline N=15** in `scripts/run_pipeline_n15.py` → `build_X(atlas, metric, band, subjects)`:
   carica `data/connectivity/ds005385/<sub>_atlas-...npz` per-epoch, estrae upper-triangle FC
   per UNA banda alla volta. **NON** invoca `features/dispatcher.build_X`. Nessuna feature univariate
   entra nei risultati di `EXPERIMENTS_N15.md`.

> **Conseguenza scientifica**: il claim "Winner aparc×coh×theta×svm_rbf BA=0.867" si basa esclusivamente
> su feature di connettività funzionale (upper triangle, una banda). Le feature univariate sono codificate
> ma NON utilizzate nel run finale. L'API in `features/dispatcher.py` resta riservata a futuri esperimenti
> (univariate-only o uni+FC) ma **non** è esercitata dalla pipeline N=15.

---

## 2. mne-features univariate (definite ma NON usate dalla pipeline N=15)

Da `features/dispatcher.py:30-38` — costante `_DEFAULT_UNI_FUNCS` + `pow_freq_bands` opzionale (default ON).

| # | Funzione mne-features | Parametri | n_features per soggetto / per epoch |
|---|------------------------|-----------|-------------------------------------|
| 1 | `compute_variance` | nessuno | n_labels |
| 2 | `compute_mean` | nessuno | n_labels |
| 3 | `compute_std` | nessuno | n_labels |
| 4 | `compute_kurtosis` | nessuno | n_labels |
| 5 | `compute_hjorth_mobility` | nessuno | n_labels |
| 6 | `compute_hjorth_complexity` | nessuno | n_labels |
| 7 | `compute_rms` | nessuno | n_labels |
| 8 | `compute_pow_freq_bands` (opzionale, default ON) | `freq_bands=[1,4,8,13,30,45]` Hz (5 bande) | n_labels × 5 |

Totale univariate per epoch (default config, include `pow_freq_bands`):
```
n_uni = (7 funcs base × n_labels) + (n_labels × 5 bande pow) = 12 × n_labels
```

Per atlante:
- aparc (68 ROI)        → 12 × 68  = 816 feature univariate
- schaefer100 (100 ROI) → 12 × 100 = 1200 feature univariate
- schaefer200 (200 ROI) → 12 × 200 = 2400 feature univariate (atlante non testato in N=15)

Scope di queste funzioni nel run N=15: **NESSUNO**. La build_X locale di `run_pipeline_n15.py:55-67`
crea X solo da `mat[idx]` con `idx = np.triu_indices(n, k=1)` su un dizionario `fc_matrix` letto da
`data/connectivity/ds005385/<sub>_atlas-..._metric-..._band-..._per-epoch.npz`.

Note di sicurezza:
- Il branch `vec.shape[0] != n_labels` in `_apply_uni_epoch` (`features/dispatcher.py:76-84`) gestisce il caso
  `pow_freq_bands` (output flat `n_labels × n_bands`) ma usa l'ordine `(b, i)` (bande esterne, label interne)
  per la generazione dei nomi mentre mne-features ritorna `(label, band)`. Questo è una **discrepanza dei
  feature_names** che non altera i valori numerici ma confonderebbe interpretazione/SHAP. Marker per
  futuro fix qualora univariate venga riattivato.

---

## 3. Feature di Connettività Funzionale (effettivamente usate)

### 3.1 Metriche supportate

Da `connectivity/fc_dispatcher.py:33` — alias `Metric` Literal:

```python
Metric = Literal["coh", "imcoh", "plv", "ciplv", "pli", "wpli", "wpli2_debiased"]
```

Backend: `mne_connectivity.spectral_connectivity_epochs(method=metric, mode="multitaper",
faverage=True)` — output ridotto a una matrice NxN simmetrizzata per banda
(`fc_dispatcher.py:80-95`).

### 3.2 Atlanti

| Atlante | n_labels (n) | Coppie upper-triangle = n(n-1)/2 |
|---------|--------------|----------------------------------|
| aparc | 68 | **2278** |
| schaefer100 | 100 | **4950** |
| schaefer200 | 200 | 19900 (definito ma NON in `ATLASES` di run_pipeline_n15.py) |

Verifica: 68×67/2 = 2278 ✓ (coincide con `n_features: 2278` in `EXPERIMENTS_N15.md` per il winner aparc×coh×theta).
Verifica: 100×99/2 = 4950 ✓.

`run_pipeline_n15.py:35` — `ATLASES = ["aparc", "schaefer100"]` (schaefer200 escluso dal run N=15).

### 3.3 Bande di frequenza

Da `fc_dispatcher.py:35-41` (default) e `run_pipeline_n15.py:38-43` (override pipeline):

| Banda | Range Hz | In DEFAULT_BANDS | In pipeline N=15 |
|-------|----------|------------------|------------------|
| delta | 1–4 | sì | no (esclusa) |
| theta | 4–8 | sì | sì |
| alpha | 8–13 | sì | sì |
| beta | 13–30 | sì | sì |
| gamma | 30–45 | sì | sì |

> Pipeline N=15 esclude esplicitamente `delta` (artefatti motori/respiratori a basso SNR su EEG eyes-open/closed).

### 3.4 Tabella feature FC totali per combo testata

Con strategia **una banda per X** (la pipeline N=15 produce 32 file `X_<atlas>_<metric>_<band>.npz`):

| Atlas | Metric | Band | n_pairs (= n_features in X) | Note |
|-------|--------|------|------------------------------|------|
| aparc | coh | theta | 2278 | **WINNER** N=15 (BA=0.867) |
| aparc | coh | alpha | 2278 | |
| aparc | coh | beta | 2278 | |
| aparc | coh | gamma | 2278 | |
| aparc | wpli | (4 bande) | 2278 ciascuna | |
| aparc | plv | (4 bande) | 2278 ciascuna | top-2/3/4/6 |
| aparc | imcoh | (4 bande) | 2278 ciascuna | |
| schaefer100 | coh | (4 bande) | 4950 ciascuna | |
| schaefer100 | wpli | (4 bande) | 4950 ciascuna | |
| schaefer100 | plv | (4 bande) | 4950 ciascuna | top-7/8/9/10 |
| schaefer100 | imcoh | (4 bande) | 4950 ciascuna | |

Totale: **2 atlas × 4 metric × 4 band = 32 file X** (vedi commento `run_pipeline_n15.py:294`).

### 3.5 Feature FC API multi-banda (in `features/dispatcher.py:142-167`)

`flatten_fc(fc: dict[str, np.ndarray])` concatena upper-triangle di tutte le bande del dict in
un singolo vettore. Se attivata, produrrebbe:

```
n_fc_features_multibanda = n_bande × n(n-1)/2
```

Es. aparc × 4 bande = 9112 feature; schaefer100 × 4 bande = 19800. **Non utilizzato** nel run N=15
(che opta per "una banda per X" per evitare collinearità inter-banda).

---

## 4. Costruzione X effettiva nel run N=15

`scripts/run_pipeline_n15.py:55-77`:

```python
def build_X(atlas, metric, band, subjects):
    rows, row_labels = [], []
    for sub in subjects:                       # 15 soggetti
        for cond in CONDITIONS:                # ["EO", "EC"]
            d = np.load(fc_path(sub, atlas, cond, metric, band))
            mat = d["fc_matrix"].astype(np.float64)
            n = mat.shape[0]
            idx = np.triu_indices(n, k=1)
            rows.append(mat[idx])              # 1 vettore per (sub, cond)
            row_labels.append(f"{sub}_{cond}")
    return np.stack(rows), row_labels          # shape (30, n_pairs)
```

`build_y_groups` (`run_pipeline_n15.py:70-77`) produce:
- `y` shape (30,): EO=0 / EC=1 (binario)
- `groups` shape (30,): subject index ripetuto 2× (per LOSO GroupKFold-15)

**Conseguenza**: ogni riga di X corrisponde ad UN soggetto in UNA condizione (EO o EC). Le epoch sono
state pre-aggregate a livello di matrice FC per-soggetto (file `*_per-epoch.npz`, contenuto: media o
matrice singola, controllare `connectivity/run_fc_on_epochs.py`). Ne segue:

- n_samples (X.shape[0]) = 15 × 2 = 30 (HARD coded come `_EXPECTED_SAMPLES` in `aggregate_n15.py:43`).
- n_features (X.shape[1]) = n(n-1)/2 dell'atlante.
- Rapporto features:samples = 2278:30 = **~76:1** per aparc, **~165:1** per schaefer100. Annotato
  come threat to validity in §6.

---

## 5. Sottoinsieme effettivamente passato al classificatore

`ml_training/aggregate_n15.py:188-326` — `aggregate_classify_n15`:
- Itera (band × atlas × metric × classifier).
- Per ogni combo carica X intero (shape guard 30) → LOSO GroupKFold-15 → `Pipeline([scaler, clf])`.
- **Nessuna feature selection** (no SelectKBest, no PCA, no VarianceThreshold) né nel build né nel pipeline.
- **Nessun hyperparameter tuning** (parametri fissi in `_LOCAL_CLASSIFIERS`/`_CLASSIFIERS`).
- Tutte le 2278 (o 4950) feature entrano nel classificatore.

I classificatori esercitati nei run N=15 (vedi `run_pipeline_n15.py:319`):
`["logreg", "svm_rbf", "lda", "rf", "mlp", "gb"]` (6).

EXPERIMENTS_N15.md (versione attuale del report) dichiara solo 3 classificatori (`logreg`, `svm_rbf`, `lda`)
con 96 combos = 2 atlas × 4 metric × 4 band × 3 clf. La code-base supporta 6 → 192 combos potenziali.
Differenza tra report e capacità del codice: il report N=15 pubblicato è uno snapshot del 4-cl run.

---

## 6. Note su feature degeneri/costanti e altri warning

- **Diagonale FC**: rimossa implicitamente da `np.triu_indices(n, k=1)` (k=1 → strict upper). NESSUN
  termine di auto-connettività entra in X. OK.
- **Simmetrizzazione**: `mat = mat + mat.T - np.diag(np.diag(mat))` (`fc_dispatcher.py:94`). Per metriche
  asimmetriche (imcoh, ciplv) questa simmetrizzazione perde informazione direzionale; per metriche
  simmetriche (coh, wpli, plv, pli, wpli2_debiased) è una no-op corretta. **Threat to validity** per
  imcoh/ciplv: la simmetrizzazione MEDIA upper+lower ⇒ non è la imcoh "as published" — annotare in
  threats to validity globale.
- **NaN handling**: `run_pipeline_n15.py:107-130` traccia `nan_count` per ogni X. La metadata.json salva
  questa diagnostica. Se NaN > 0, `bootstrap_ci` e `balanced_accuracy_score` propagano NaN.
  Nessun fillna upstream — verifica esistenza NaN nei file salvati raccomandata (fuori scope di questo audit).
- **Feature constanti**: nessuna varianza-threshold applicata. ROI-pair con stessa attività (es. coppie con
  zero FC su tutti i soggetti) entrano comunque in X e contribuiscono a 0 al fit dello scaler (warnings
  silenziati in `_loso_cv` via `warnings.simplefilter("ignore")`). Possibile confonditore numerico ma non
  un bias scientifico.
- **Broadcast univariate vs FC**: il `flatten_fc` in `features/dispatcher.py:142-167` produce vettore 1D
  (n_fc_features,), poi `build_X` lo replica con `np.tile(fc_vec, (n_epochs, 1))`. Se attivato
  (non lo è in N=15), produce **feature FC costanti tra epoch dello stesso soggetto** ⇒ rischio di
  inflated CV se groups non corrispondono a soggetti. La pipeline N=15 evita il problema usando una
  riga per (sub, cond) e non per epoch.
- **Discrepanza nomi pow_freq_bands** (vedi §2): non riguarda i risultati N=15 ma marker per riattivazione
  univariate.

---

## 7. Riassunto esecutivo (TL;DR per professore)

1. Il run N=15 (`EXPERIMENTS_N15.md`) usa **solo** feature di connettività funzionale (upper triangle, una
   banda alla volta). Nessuna feature univariate (variance, mean, hjorth, pow_freq_bands) entra nel modello.
2. Per il winner `aparc × coh × theta × svm_rbf`: **2278 feature FC** per 30 campioni (15 soggetti × 2 cond).
3. La pipeline esercita 32 matrici X (2 atlanti × 4 metriche × 4 bande), ciascuna passata a 6 classificatori
   in LOSO GroupKFold-15 con bootstrap CI e permutation test (n_perm=100 nel run pubblicato; default code
   = 1000 paper-grade).
4. **Nessuna feature selection o hyperparameter tuning** è applicata. Il rapporto features:campioni
   altissimo (~76:1 per aparc, ~165:1 per schaefer100) è un threat to validity da discutere nella sezione
   "limiti" della tesi (overfitting + bias del bootstrap CI in alta dimensionalità).
5. La API `features/dispatcher.build_X` resta disponibile per esperimenti futuri uni+FC ma è **bypassata**
   dal run N=15.

---

## 8. Marker

```
[QUEUE_TASK_DONE: H-AUDIT-01]
```
