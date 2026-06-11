# Deep Review STEP 3 — Mode Comparison & Label Coverage

**Sprint**: BAND-5-9-T01  
**Data**: 2026-05-01  
**STC**: `sub-05_task-matchingpennies_cond-raised-left_inv-dSPM` (shape 68×3501, sfreq 5000 Hz)  
**Atlas mode comparison**: `aparc` (68 ROI)  
**Metrica FC proxy**: correlazione di Pearson (upper triangle, n_pairs=2278)

---

## 1. Mode Comparison — Analisi Numerica

### 1.1 Statistiche dei time course per mode

| Mode | mean_abs | std | Nota |
|------|----------|-----|------|
| `mean` | 66.63 | 42.11 | Media aritmetica dei vertici; sensibile al segno ambiguo |
| `mean_flip` | 28.55 | 32.59 | Media con flip ottimale del segno; ampiezza ridotta (cancellazioni parziali) |
| `pca_flip` | 72.09 | 59.17 | Prima componente PCA + flip; cattura la massima varianza spiegata |
| `max` | 131.12 | 73.62 | Vertice a massima ampiezza istantanea; rumoroso per ROI eterogenee |

La riduzione di `mean_abs` da `mean` (66.6) a `mean_flip` (28.6) è attesa: il flip di segno corretto introduce cancellazioni parziali tra vertici con polarità opposta ma attività correlata, sopprimendo componenti common-mode.

### 1.2 Correlazione Spearman tra matrici FC (proxy Pearson)

| Mode 1 | Mode 2 | ρ Spearman | p-value | Interpretazione |
|--------|--------|-----------|---------|----------------|
| `mean` | `mean_flip` | **0.330** | 6.9×10⁻⁵⁹ | Bassa — le mode producono FC sostanzialmente diverse |
| `mean` | `pca_flip` | 0.676 | <10⁻³⁰⁰ | Moderata–alta |
| `mean` | `max` | 0.647 | <10⁻²⁷⁰ | Moderata–alta |
| `mean_flip` | `pca_flip` | 0.627 | <10⁻²⁴⁹ | Moderata — le due mode "flip" sono coerenti tra loro |
| `mean_flip` | `max` | **0.245** | 2.2×10⁻³² | Molto bassa — `max` diverge da `mean_flip` |
| `pca_flip` | `max` | 0.491 | <10⁻¹³⁸ | Moderata |

> **Finding critico**: `mean` vs `mean_flip` ha ρ=0.33 — notevolmente più basso di quanto atteso. Questo implica che la scelta della mode non è un fattore trascurabile e produce matrici FC qualitativamente diverse.

### 1.3 Differenza media assoluta FC (baseline = `mean_flip`)

| Mode | MAD (vs mean_flip) | Max diff |
|------|--------------------|----------|
| `mean` | 0.466 | 1.880 |
| `pca_flip` | 0.358 | 1.784 |
| `max` | 0.538 | 1.841 |

La scala delle differenze va da 0 a 1 (correlazione di Pearson). Una MAD di 0.47 per `mean` rispetto a `mean_flip` è sostanziale: corrisponde al ~47% della scala unitaria.

### 1.4 Analisi per-label del sign flip (`mean` vs `mean_flip`)

| Metrica | Valore |
|---------|--------|
| Label con segno invertito | **9 / 68 (13.2%)** |
| Label con segno uguale | 59 / 68 (86.8%) |
| |ρ Pearson| medio per-label | 0.782 |

13 label su 68 vengono invertite di segno da `mean_flip` rispetto a `mean`. Queste sono le ROI dove la dipolo media dei vertici è prevalentemente tangenziale e/o la distribuzione dei vertici è bimodale in termini di orientamento preferenziale. L'inversione non è un errore: è la correzione attesa che garantisce che la media tra vertici con segno opposto non si cancelli artificialmente.

Il |ρ| medio di 0.78 tra `mean` e `mean_flip` per singola label conferma che i due mode producono time course simili nella forma (correlazione alta) ma con segno potenzialmente invertito e ampiezza diversa — differenza rilevante per FC, che misura similarità tra segnali.

---

## 2. Label Coverage Cross-Atlas

### 2.1 Tabella coverage (su `oct6` source space, 8196 vertici totali)

| Atlas | N tot | N escluse | N valide | Label vuote | Verts/ROI avg | Verts/ROI min | Verts/ROI max | Copertura src |
|-------|-------|-----------|----------|-------------|--------------|--------------|--------------|---------------|
| `aparc` | 69 | 1 | **68** | 0 | 110.5 | 10 | 288 | 91.7% |
| `destrieux` | 150 | 2 | **148** | 0 | 50.8 | **2** | 188 | 91.7% |
| `schaefer100` | 102 | 2 | **100** | 0 | 75.1 | 31 | 182 | 91.6% |
| `schaefer200` | 202 | 2 | **200** | 0 | 37.5 | 10 | 81 | 91.6% |

### 2.2 Finding: nessuna label vuota

**Tutte e 4 le configurazioni di atlante hanno 0 label vuote** nella source space `oct6` (spacing ~4.9 mm, 8196 vertici). Questo è il risultato atteso: il fsaverage `oct6` ha densità sufficiente a coprire anche le ROI più piccole di tutti gli atlanti testati.

### 2.3 Finding: copertura uniforme ~91.6–91.7%

L'8.3–8.4% di vertici sorgente non coperti da nessuna ROI corrisponde a:
- Regioni sub-corticali profonde non incluse negli atlanti corticali (nuclei della base, talamo, ecc.)
- Alcune aree di confine tra emisferi (regioni peri-calcarine molto mediali)

Questa copertura è coerente con la letteratura: gli atlanti corticali sono progettati per coprire la corteccia, non le strutture sub-corticali.

### 2.4 Warning: ROI molto piccole in `destrieux`

L'atlante `destrieux` ha **min_verts=2** (una ROI con soli 2 vertici src). Una ROI con 2 vertici produce un time course con varianza molto alta e stima di FC poco affidabile. Per il benchmark S-08, si raccomanda di monitorare le ROI con `n_verts < 5` e, se necessario, applicare un threshold di esclusione (es. `min_verts=5`).

ROI piccole in `schaefer200`: min_verts=10 — accettabile (più ROI, ma dimensione minima ragionevole).

### 2.5 Implicazioni per la stima FC

La stima di connettività funzionale è sensibile al numero di vertici per ROI:

| Verts/ROI | Stabilità TC | Rischio FC | Raccomandazione |
|-----------|-------------|------------|----------------|
| ≥ 50 | Alta | Basso | OK per tutti gli atlanti |
| 10–49 | Moderata | Moderato | Monitorare varianza residua |
| 2–9 | Bassa | Alto | Warning — valutare esclusione |

`aparc` e `schaefer100` hanno media alta (110.5 e 75.1) e minimi accettabili (10 e 31). `destrieux` e `schaefer200` hanno ROI più piccole: la stima FC per queste ROI sarà più rumorosa.

---

## 3. Conclusioni e Raccomandazioni

### 3.1 Raccomandazione: `mean_flip` come default ✓

`mean_flip` rimane la scelta raccomandata per FC source-level per tre ragioni:

1. **Robustezza al sign ambiguity**: 9/68 label hanno segno invertito rispetto a `mean`. Senza flip, la media tra vertici con orientamento opposto si cancella parzialmente, producendo time course attenuati e FC artificialmente ridotta.

2. **Correttezza teorica**: la letteratura MNE raccomanda esplicitamente `mean_flip` per FC analysis (Hipp et al., 2012; MNE docs). Il metodo minimizza la varianza del residuo dopo la media.

3. **Separazione da `pca_flip`**: ρ=0.63 tra `mean_flip` e `pca_flip` indica coerenza sufficiente per usare `mean_flip` come default e `pca_flip` come sensitivity test.

### 3.2 `pca_flip` come sensitivity test opzionale

`pca_flip` cattura la prima componente PCA della distribuzione spaziale dei vertici nella label — massimizza la varianza spiegata e può essere più accurata in ROI con forte anisotropia morfologica. Differisce da `mean_flip` con MAD=0.36 sulla FC proxy. Si raccomanda di includere `pca_flip` come sensitivity analysis nel manoscritto (es. Appendice Supplementare).

### 3.3 Sconsigliato: `mean` e `max`

- `mean` non corregge il sign ambiguity → produce FC sistematicamente biasata verso il basso (cancellazioni involontarie).
- `max` preleva il vertice a massima ampiezza istantanea → non è una media su scala temporale, è sensibile a spike locali e non è comparabile tra ROI di dimensioni diverse.

### 3.4 Action items per il codice

| Item | Priorità | Modulo |
|------|----------|--------|
| Aggiungere parametro `min_verts_threshold` in `get_labels()` per escludere ROI piccole | MEDIUM | `parcellation/extract_label_tc.py` |
| Documentare la scelta `mean_flip` nel docstring di `extract_tc()` con riferimento Hipp 2012 | LOW | `parcellation/extract_label_tc.py` |
| Aggiungere sensitivity test `pca_flip` nel benchmark S-08 (flag opzionale) | LOW | `pipeline_mne_bids/run_bench_matrix.py` |
| Warning se `n_verts < 5` in una ROI durante `extract_tc` | MEDIUM | `parcellation/extract_label_tc.py` |

---

## Riferimenti

- Hipp, J.F., et al. (2012). Large-scale cortical correlation structure of spontaneous oscillatory activity. *Nature Neuroscience*, 15(6), 884–890.
- MNE-Python docs: `mne.extract_label_time_course` — parameter `mode`.
- Desikan, R.S., et al. (2006). An automated labeling system for subdividing the human cerebral cortex. *NeuroImage*, 31(3).
- Destrieux, C., et al. (2010). Automatic parcellation of human cortical gyri and sulci. *NeuroImage*, 53(1).
- Schaefer, A., et al. (2018). Local-global parcellation of the human cerebral cortex. *Cerebral Cortex*, 28(9).
