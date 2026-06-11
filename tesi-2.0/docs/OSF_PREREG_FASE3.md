# Pre-Registration FASE 3 — Età + Sesso su N=179 (ds005385)

**Data creazione**: 2026-06-04
**Autore**: Samuele (Tesi_2.0, branch main)
**Status**: PRE-REGISTRATO — scritto PRIMA del lancio ML (step 3.5-3.6)
**Anti-HARKing**: target e soglie dichiarati qui prima di qualsiasi run su N=179
**Template**: OSF Prereg Challenge (adattato)
**Nota OSF**: il submit tramite interfaccia OSF va eseguito dall'operatore (URL non inserito automaticamente).

---

## 1. Hypotheses

### H1 — Classificazione sesso (primaria)

> EEG resting-state FC (PLV theta, parcellazione aparc, N=179) permette di classificare
> il sesso biologico con balanced accuracy **significativamente superiore a chance** (BA>0.5).

- **Modello a-priori**: LogisticRegression-L2 (`solver=liblinear`), `class_weight='balanced'`
- **Feature a-priori**: `X_aparc_plv_theta` — winner FASE 1 (dataset indipendente dalla presente analisi)
- **Soglia di significatività**: α=0.05 (corretta FDR-BH su famiglia di 4 test simultanei)
- **Metrica primaria**: balanced accuracy = (sensitivity + specificity) / 2
- **Direzione**: BA_N179 ≈ BA_N100 = 0.713 (equivalenza statistica, non miglioramento atteso)
- **N=100 FASE 1**: BA=0.713, p=0.020 — replica a priori target

### H2 — Predizione età (primaria)

> EEG resting-state FC (PLV theta, parcellazione aparc, N=179) permette di predire l'età
> con MAE **inferiore al dummy predictor (mean)**, con R² > 0 significativo.

- **Modello a-priori**: RandomForestRegressor (n_estimators=100, random_state=42)
- **Feature a-priori**: `X_aparc_plv_theta` — winner FASE 1
- **Soglia di significatività**: α=0.05 (FDR-BH, permutation test su R²)
- **Metriche primarie**: MAE (anni), R² (coefficiente determinazione)
- **Target atteso**: MAE ≈ 12.52 anni (N=100 FASE 1); R² ≈ 0.097
- **N=100 FASE 1**: MAE=12.52, R²=0.097, p=0.020

### H3 — Graph-theory 8-dim sesso (secondaria)

> La firma topologica globale EEG (8 scalari graph-theory da PLV theta) è sufficiente
> per classificare il sesso con BA significativamente superiore a chance.

- N=100 FASE 1: BA_GT=0.706, p_fdr=0.026 — replica a priori
- **Non primaria**: usata solo per conferma interpretabilità

### H4 — Graph-theory 8-dim età (secondaria)

> 8 scalari GT non sono sufficienti per predire l'età (R² ≤ 0, ns).

- N=100 FASE 1: R²_GT=-0.222, p_fdr=0.863 — replica negativa attesa
- **Non primaria**: usata solo per completezza confronto

---

## 2. Study Design

### 2.1 Partecipanti

- Dataset: **ds005385** (OpenNeuro, EEG resting-state)
- N target: **179 soggetti** (whitelist determinististica `config/subjects_whitelist_n179.py`,
  seed=43, hard-rule N100⊂N179, `late_ses1==0`, sub scaricati localmente)
- Esclusioni pre-pianificate: soggetti con EDF mancante per entrambe le condizioni,
  o con < 5 epoche valide dopo pulizia

### 2.2 Variabili dipendenti

| Target | Tipo | Codifica | Fonte |
|--------|------|----------|-------|
| Sesso | Classificazione binaria | F→0, M→1 | participants.tsv `sex` |
| Età | Regressione continua | anni (float) | participants.tsv `age` |

### 2.3 Variabili indipendenti (features)

- **Baseline**: `X_aparc_plv_theta` — matrice FC appiattita (triu), PLV banda theta [4-8 Hz],
  parcellazione aparc (68 ROI), 2278 feature per (soggetto, condizione)
- **Graph-theory**: 8 scalari calcolati su FC 68×68 ricostruita da X_aparc_plv_theta:
  mean_degree, mean_strength, clustering_coeff, path_length, global_efficiency,
  local_efficiency, modularity_q, small_worldness (threshold proporzionale 20%)

---

## 3. Analysis Plan

### 3.1 Preprocessing

Identico a FASE 1 (no protocol drift):
1. Source reconstruction dSPM su fsaverage (λ²=1/9, loose=0.2, depth=0.8)
2. Parcellazione aparc (68 ROI, mode=mean_flip)
3. FC PLV per-epoch in banda theta [4-8 Hz] (multitaper)
4. Feature matrix: triu(FC) per (soggetto, condizione)

### 3.2 Machine Learning

```
Outer CV:  GroupKFold(n_splits=5), gruppi = soggetti
Inner CV:  GroupKFold(n_splits=3), solo su train fold
Scaler:    StandardScaler fit su train fold ONLY
```

**Sesso (H1)**:
- `LogisticRegression(C=GridSearch([0.01,0.1,1,10]), class_weight='balanced', solver='liblinear')`
- Metrica scoring inner: balanced_accuracy
- Metrica outer: BA per fold → media

**Età (H2)**:
- `RandomForestRegressor(n_estimators=100)` (no hyperparameter search, coerente N=100)
- Metrica outer: MAE + R² su concatenazione predizioni out-of-fold

### 3.3 Statistical Tests

**Permutation test** (n_perm=1000, subject-level):
- Shuffle: permutazione delle label PER SOGGETTO (broadcast via groups)
- H1: BA_null distribuzione → p = (#{null_BA ≥ BA_real} + 1) / (n_perm + 1)
- H2: R²_null distribuzione → p = (#{null_R² ≥ R²_real} + 1) / (n_perm + 1)

**Bootstrap CI 95%** (n_boot=1000, cluster subject-level):
- Ricampionamento a livello soggetto (non sample) per rispettare struttura dati
- Metodo: percentile bootstrap
- H1: CI_BA; H2: CI_MAE, CI_R²

**Correzione multipla (FDR-BH)**:
- Famiglia: 4 test simultanei — sex_baseline, sex_gt, age_baseline, age_gt
- α = 0.05 dopo correzione
- Implementazione: `statsmodels.stats.multitest.multipletests(method='fdr_bh')`

### 3.4 Comparison N=100 vs N=179

| Metrica | N=100 target | Criterio equivalenza |
|---------|-------------|----------------------|
| sex BA | 0.713 | |ΔBA| ≤ 0.05 AND CI N=179 ⊃ BA_N100 |
| age MAE | 12.52 anni | |ΔMAE| ≤ 2.0 anni |
| age R² | 0.097 | stessa direzione (R²>0) |

### 3.5 Seed control

```python
np.random.seed(42)
RandomForestRegressor(random_state=42)
LogisticRegression(random_state=42)
rng = np.random.default_rng(42)  # per permutation + bootstrap
```

---

## 4. Exclusion Criteria

1. Soggetto con EDF mancante per ENTRAMBE le condizioni (EO e EC): escluso da preprocessing
2. Soggetto con NaN in participants.tsv per `age` o `sex`: errore fatale (pipeline si ferma)
3. Feature matrix con shape ≠ (N×2, 2278): errore fatale (pipeline si ferma)
4. Soggetti non in whitelist_n179 (late_ses1=1 o non scaricati): già esclusi a priori da whitelist

Non sono previste esclusioni post-hoc basate sui risultati ML.

---

## 5. Statistical Thresholds

| Parametro | Valore |
|-----------|--------|
| α (per-comparison) | 0.05 |
| Correzione multipla | FDR-BH (Benjamini-Hochberg) |
| n_perm permutation | 1000 |
| n_boot bootstrap | 1000 |
| Outer K (GroupKFold) | 5 |
| Inner K (GridSearch) | 3 |
| GT threshold | 20% archi (proporzionale) |

---

## 6. Outcome Declaration

### Expected findings (basati su N=100 FASE 1):

- **sex_baseline**: BA_N179 ≈ 0.71 ± 0.03, p_fdr < 0.05 → ✅ significativo
- **sex_gt**: BA_GT_N179 ≈ 0.70 ± 0.03, p_fdr < 0.05 → ✅ significativo
- **age_baseline**: MAE ≈ 11-13 anni, R² ≈ 0.05-0.12, p_fdr < 0.05 → ✅ significativo
- **age_gt**: R²_GT < 0, p_fdr > 0.05 → ❌ non significativo (atteso)

### Boundary conditions:

- Se BA_N179 < 0.65 o p_fdr > 0.05 per sex_baseline: **risultato non replicato** —
  riportare onestamente come failure-to-replicate con analisi delle possibili cause.
- Se R²_N179 < 0 per age_baseline: **nessun effetto su N=179** — riportare onestamente.

---

## 7. Deviations Policy

Qualsiasi deviazione dal piano sopra (feature diverse, algoritmi diversi, criteri di esclusione
aggiuntivi) deve essere riportata esplicitamente come **POST-HOC** nel report finale
`reports/AGE_SEX_FASE3_N179.md`, separata dai risultati pre-pianificati.

---

*Pre-registration scritta il 2026-06-04 PRIMA del lancio di `scripts/run_ml_age_sex_n179.py`.*
*Commit atomico di questo file precede il commit dei risultati ML.*
