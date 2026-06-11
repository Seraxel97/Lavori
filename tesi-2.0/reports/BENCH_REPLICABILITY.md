# Bench Replicability Audit — 96-combo N=15

**Task**: H-AUDIT-04  
**Worker**: sonnet1-tesi (Sonnet 4.6)  
**Timestamp**: 2026-05-07T16:48:00Z  
**Sorgenti**: `data/results/ds005385/comparison_matrix_N15.json`, `data/features/ds005385/`, `data/connectivity/ds005385/`, `reports/EXPERIMENTS_N15.md`

---

## Sommario

| Parametro | Valore |
|-----------|--------|
| Combo totali verificate | **96** (2 atlas × 4 metric × 4 band × 3 clf) |
| REPLICABLE | **96** (100%) |
| PARTIAL | 0 |
| MISSING | 0 |
| FC per-epoch files | 960/960 ✓ |
| Feature X matrices | 32/32 ✓ (zero NaN) |
| Manifest/hash | `_hashes` (32 entry) in comparison_matrix_N15.json ✓ |
| per-run MANIFEST_TEMPLATE | Non generato da run_pipeline_n15.py (template-only) |

---

## Artefatti verificati

### 1. FC matrices (per-epoch)

| Combinazione | File count | Attesi | Status |
|-------------|------------|--------|--------|
| `*_coh_alpha_per-epoch*` | 60 | 60 | ✓ |
| `*_coh_beta_per-epoch*` | 60 | 60 | ✓ |
| `*_coh_gamma_per-epoch*` | 60 | 60 | ✓ |
| `*_coh_theta_per-epoch*` | 60 | 60 | ✓ |
| `*_imcoh_alpha_per-epoch*` | 60 | 60 | ✓ |
| `*_imcoh_beta_per-epoch*` | 60 | 60 | ✓ |
| `*_imcoh_gamma_per-epoch*` | 60 | 60 | ✓ |
| `*_imcoh_theta_per-epoch*` | 60 | 60 | ✓ |
| `*_plv_alpha_per-epoch*` | 60 | 60 | ✓ |
| `*_plv_beta_per-epoch*` | 60 | 60 | ✓ |
| `*_plv_gamma_per-epoch*` | 60 | 60 | ✓ |
| `*_plv_theta_per-epoch*` | 60 | 60 | ✓ |
| `*_wpli_alpha_per-epoch*` | 60 | 60 | ✓ |
| `*_wpli_beta_per-epoch*` | 60 | 60 | ✓ |
| `*_wpli_gamma_per-epoch*` | 60 | 60 | ✓ |
| `*_wpli_theta_per-epoch*` | 60 | 60 | ✓ |
| **TOTALE** | **960** | 960 | ✓ |

Pattern: `sub-{XXX}_atlas-{atlas}_cond-{EO/EC}_metric-{metric}_band-{band}_per-epoch.npz`  
Struttura: 15 soggetti × 2 condizioni × 2 atlas = 60 per metric×band combinazione.

### 2. Feature X matrices

| File | Shape | NaN | Status |
|------|-------|-----|--------|
| X_aparc_coh_alpha.npz | (30, 2278) | 0 | ✓ |
| X_aparc_coh_beta.npz | (30, 2278) | 0 | ✓ |
| X_aparc_coh_gamma.npz | (30, 2278) | 0 | ✓ |
| X_aparc_coh_theta.npz | (30, 2278) | 0 | ✓ |
| X_aparc_imcoh_alpha.npz | (30, 2278) | 0 | ✓ |
| X_aparc_imcoh_beta.npz | (30, 2278) | 0 | ✓ |
| X_aparc_imcoh_gamma.npz | (30, 2278) | 0 | ✓ |
| X_aparc_imcoh_theta.npz | (30, 2278) | 0 | ✓ |
| X_aparc_plv_alpha.npz | (30, 2278) | 0 | ✓ |
| X_aparc_plv_beta.npz | (30, 2278) | 0 | ✓ |
| X_aparc_plv_gamma.npz | (30, 2278) | 0 | ✓ |
| X_aparc_plv_theta.npz | (30, 2278) | 0 | ✓ |
| X_aparc_wpli_alpha.npz | (30, 2278) | 0 | ✓ |
| X_aparc_wpli_beta.npz | (30, 2278) | 0 | ✓ |
| X_aparc_wpli_gamma.npz | (30, 2278) | 0 | ✓ |
| X_aparc_wpli_theta.npz | (30, 2278) | 0 | ✓ |
| X_schaefer100_coh_alpha.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_coh_beta.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_coh_gamma.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_coh_theta.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_imcoh_alpha.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_imcoh_beta.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_imcoh_gamma.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_imcoh_theta.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_plv_alpha.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_plv_beta.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_plv_gamma.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_plv_theta.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_wpli_alpha.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_wpli_beta.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_wpli_gamma.npz | (30, 4950) | 0 | ✓ |
| X_schaefer100_wpli_theta.npz | (30, 4950) | 0 | ✓ |

`y.npy`: shape (30,), bilanciato {EO: 15, EC: 15}. `groups.npy`: shape (30,), 15 soggetti LOSO-ready.  
`metadata.json`: presente (provenance completa: subjects, conditions, atlases, metrics, bands, row_order).

### 3. Manifest / Provenance

| Artefatto | Presente | Note |
|-----------|----------|------|
| `data/features/ds005385/metadata.json` | ✓ | Provenance completa (subjects, conditions, atlases, metrics, bands) |
| `data/results/ds005385/comparison_matrix_N15.json` → `_hashes` | ✓ | 32 X-matrix fingerprint (per atlas×metric×band) |
| `data/results/ds005385/comparison_matrix_N15.json` → `_meta` | ✓ | n_perm=100, seed=42, n_subjects=15, atlases, metrics, clfs |
| `common/run_id.py` | ✓ | Modulo per generare run-id (non chiamato da run_pipeline_n15.py) |
| `common/reproducibility.py` | ✓ | Manifest builder (non chiamato da run_pipeline_n15.py) |
| `reports/MANIFEST_TEMPLATE.json` | ✓ | Template schema (non istanziato per questo run) |

**Nota**: `run_pipeline_n15.py` non chiama `common.reproducibility.build_manifest()` — mancano manifest JSON per run individuali. L'integrità è garantita dagli `_hashes` per gli X e dal `metadata.json` delle features. Per paper-grade: aggiungere chiamata a `save_manifest()` in run_pipeline_n15.py (costo: ~5 righe, impact: MEDIUM).

---

## Tabella combo 96 — stato replicabilità

`fc_matrix` = 30/30 per-epoch file per atlas×metric×band.  
`features` = X_{atlas}_{metric}_{band}.npz presente.  
`manifest` = `_hashes` entry in comparison_matrix_N15.json.

| combo_id | atlas | metric | band | clf | fc_matrix | features | manifest | status |
|----------|-------|--------|------|-----|-----------|----------|----------|--------|
| aparc×wpli×theta×logreg | aparc | wpli | theta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×wpli×theta×svm_rbf | aparc | wpli | theta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×wpli×theta×lda | aparc | wpli | theta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×wpli×alpha×logreg | aparc | wpli | alpha | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×wpli×alpha×svm_rbf | aparc | wpli | alpha | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×wpli×alpha×lda | aparc | wpli | alpha | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×wpli×beta×logreg | aparc | wpli | beta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×wpli×beta×svm_rbf | aparc | wpli | beta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×wpli×beta×lda | aparc | wpli | beta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×wpli×gamma×logreg | aparc | wpli | gamma | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×wpli×gamma×svm_rbf | aparc | wpli | gamma | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×wpli×gamma×lda | aparc | wpli | gamma | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×coh×theta×logreg | aparc | coh | theta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| **aparc×coh×theta×svm_rbf** | aparc | coh | theta | svm_rbf | ✓ | ✓ | ✓ | **REPLICABLE** |
| aparc×coh×theta×lda | aparc | coh | theta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×coh×alpha×logreg | aparc | coh | alpha | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×coh×alpha×svm_rbf | aparc | coh | alpha | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×coh×alpha×lda | aparc | coh | alpha | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×coh×beta×logreg | aparc | coh | beta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×coh×beta×svm_rbf | aparc | coh | beta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×coh×beta×lda | aparc | coh | beta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×coh×gamma×logreg | aparc | coh | gamma | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×coh×gamma×svm_rbf | aparc | coh | gamma | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×coh×gamma×lda | aparc | coh | gamma | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×theta×logreg | aparc | plv | theta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×theta×svm_rbf | aparc | plv | theta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×theta×lda | aparc | plv | theta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×alpha×logreg | aparc | plv | alpha | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×alpha×svm_rbf | aparc | plv | alpha | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×alpha×lda | aparc | plv | alpha | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×beta×logreg | aparc | plv | beta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×beta×svm_rbf | aparc | plv | beta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×beta×lda | aparc | plv | beta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×gamma×logreg | aparc | plv | gamma | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×gamma×svm_rbf | aparc | plv | gamma | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×plv×gamma×lda | aparc | plv | gamma | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×theta×logreg | aparc | imcoh | theta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×theta×svm_rbf | aparc | imcoh | theta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×theta×lda | aparc | imcoh | theta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×alpha×logreg | aparc | imcoh | alpha | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×alpha×svm_rbf | aparc | imcoh | alpha | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×alpha×lda | aparc | imcoh | alpha | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×beta×logreg | aparc | imcoh | beta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×beta×svm_rbf | aparc | imcoh | beta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×beta×lda | aparc | imcoh | beta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×gamma×logreg | aparc | imcoh | gamma | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×gamma×svm_rbf | aparc | imcoh | gamma | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| aparc×imcoh×gamma×lda | aparc | imcoh | gamma | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×theta×logreg | schaefer100 | wpli | theta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×theta×svm_rbf | schaefer100 | wpli | theta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×theta×lda | schaefer100 | wpli | theta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×alpha×logreg | schaefer100 | wpli | alpha | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×alpha×svm_rbf | schaefer100 | wpli | alpha | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×alpha×lda | schaefer100 | wpli | alpha | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×beta×logreg | schaefer100 | wpli | beta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×beta×svm_rbf | schaefer100 | wpli | beta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×beta×lda | schaefer100 | wpli | beta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×gamma×logreg | schaefer100 | wpli | gamma | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×gamma×svm_rbf | schaefer100 | wpli | gamma | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×wpli×gamma×lda | schaefer100 | wpli | gamma | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×theta×logreg | schaefer100 | coh | theta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×theta×svm_rbf | schaefer100 | coh | theta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×theta×lda | schaefer100 | coh | theta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×alpha×logreg | schaefer100 | coh | alpha | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×alpha×svm_rbf | schaefer100 | coh | alpha | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×alpha×lda | schaefer100 | coh | alpha | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×beta×logreg | schaefer100 | coh | beta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×beta×svm_rbf | schaefer100 | coh | beta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×beta×lda | schaefer100 | coh | beta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×gamma×logreg | schaefer100 | coh | gamma | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×gamma×svm_rbf | schaefer100 | coh | gamma | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×coh×gamma×lda | schaefer100 | coh | gamma | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×theta×logreg | schaefer100 | plv | theta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×theta×svm_rbf | schaefer100 | plv | theta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×theta×lda | schaefer100 | plv | theta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×alpha×logreg | schaefer100 | plv | alpha | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×alpha×svm_rbf | schaefer100 | plv | alpha | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×alpha×lda | schaefer100 | plv | alpha | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×beta×logreg | schaefer100 | plv | beta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×beta×svm_rbf | schaefer100 | plv | beta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×beta×lda | schaefer100 | plv | beta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×gamma×logreg | schaefer100 | plv | gamma | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×gamma×svm_rbf | schaefer100 | plv | gamma | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×plv×gamma×lda | schaefer100 | plv | gamma | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×theta×logreg | schaefer100 | imcoh | theta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×theta×svm_rbf | schaefer100 | imcoh | theta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×theta×lda | schaefer100 | imcoh | theta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×alpha×logreg | schaefer100 | imcoh | alpha | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×alpha×svm_rbf | schaefer100 | imcoh | alpha | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×alpha×lda | schaefer100 | imcoh | alpha | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×beta×logreg | schaefer100 | imcoh | beta | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×beta×svm_rbf | schaefer100 | imcoh | beta | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×beta×lda | schaefer100 | imcoh | beta | lda | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×gamma×logreg | schaefer100 | imcoh | gamma | logreg | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×gamma×svm_rbf | schaefer100 | imcoh | gamma | svm_rbf | ✓ | ✓ | ✓ | REPLICABLE |
| schaefer100×imcoh×gamma×lda | schaefer100 | imcoh | gamma | lda | ✓ | ✓ | ✓ | REPLICABLE |

---

## Top-3 winner replicability

### 1. Winner: aparc×coh×theta×svm_rbf (BA=0.867, p_perm=0.0000)

| Artefatto | Path | Status |
|-----------|------|--------|
| FC per-epoch (aparc×coh×theta) | `data/connectivity/ds005385/*_atlas-aparc_*_metric-coh_band-theta_per-epoch.npz` | ✓ 30/30 |
| Feature X | `data/features/ds005385/X_aparc_coh_theta.npz` | ✓ shape=(30, 2278), NaN=0 |
| X hash | `comparison_matrix_N15.json._hashes["aparc_coh_theta"]` = `d922119068fb5641` | ✓ |
| Result | `comparison_matrix_N15.json.winner` | ✓ BA=0.8667, p=0.0 |

**Status: REPLICABLE**

### 2. aparc×plv×theta×logreg (BA=0.867)

| Artefatto | Status |
|-----------|--------|
| FC per-epoch (aparc×plv×theta) | ✓ 30/30 |
| X_aparc_plv_theta.npz | ✓ shape=(30, 2278) |
| Result in comparison_matrix_N15.json | ✓ BA=0.867 |

**Status: REPLICABLE**

### 3. aparc×plv×alpha×logreg (BA=0.867)

| Artefatto | Status |
|-----------|--------|
| FC per-epoch (aparc×plv×alpha) | ✓ 30/30 |
| X_aparc_plv_alpha.npz | ✓ shape=(30, 2278) |
| Result in comparison_matrix_N15.json | ✓ BA=0.867 |

**Status: REPLICABLE**

---

## Recovery path per MISSING

**Nessun artefatto MISSING** — tutte le 96 combo sono completamente ricostruibili da artefatti su disco.

### Unica pendenza (non bloccante): manifest per-run

`run_pipeline_n15.py` non chiama `common.reproducibility.build_manifest()`. Per paper-grade:

```python
# Da aggiungere in scripts/run_pipeline_n15.py dopo Step 7
from common.reproducibility import build_manifest, save_manifest
manifest = build_manifest(run_id, config_path="config/subjects_whitelist.py")
save_manifest(manifest, "reports/")
```

Costo: ~5 righe. Step interessato: Step 6+7 (ML). Nessun rerun necessario — aggiunta possibile in autonomia pre-paper.

---

## Note BENCH_MATRIX_RESULTS.json

Il file `reports/BENCH_MATRIX_RESULTS.json` è un **placeholder** (700 run = 7 metric × 4 atlas × 5 algo × 5 bande — formato matchingpennies) e **non contiene** i risultati reali del grid N=15. I risultati reali sono in:
- `data/results/ds005385/comparison_matrix_N15.json` (96 entry, JSON strutturato)
- `reports/EXPERIMENTS_N15.md` (sintesi human-readable)

---

## Verdetto finale

**Bench grid 96-combo: REPLICABILE COMPLETO (96/96 = 100%)**

Tutti i componenti richiesti per la replica sono presenti e integri:
- 960/960 FC per-epoch files
- 32/32 feature X matrices (zero NaN)
- 96/96 risultati in comparison_matrix_N15.json
- Winner `aparc×coh×theta×svm_rbf` **verificato REPLICABLE**

L'unica pendenza è l'integrazione del manifest builder (`common/reproducibility.py`) in `run_pipeline_n15.py` per audit paper-grade, ma non blocca la replica dei risultati attuali.

---

[QUEUE_TASK_DONE: H-AUDIT-04]
