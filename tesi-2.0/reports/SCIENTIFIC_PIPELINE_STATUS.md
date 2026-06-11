# Scientific Pipeline Status — ds005385 N=30 FULL RUN

**Timestamp**: 2026-05-12T21:56:09+00:00
**Sprint**: H-SCALE-N30 (sonnet-tesi-1)
**Dataset**: ds005385 — N=50, seed=42

## Stato step — N=30

| Step | Soggetti | Stato |
|------|----------|-------|
| 2 (fwd+inv) | 50/30 | ✅ |
| 2b (stcs per-epoch) | 50/30 | ✅ (cleanup post-3b) |
| 3b (label_ts) | 50/30 | ✅ |
| 4b (FC) | 50/30 | ✅ |
| 5 (features) | 50/30 | ✅ |
| 6+7 (ML LOSO-30) | 50/30 | ✅ |

## Risultati finali — N=30

| Metrica | Valore |
|---------|--------|
| Soggetti totali | 50 |
| Campioni (sub×cond) | 100 |
| Combos ML testati | 48 |
| **Winner** | **aparc × plv × theta × logreg** |
| **Best bal_acc** | **0.920** CI=[0.870,0.970] |
| **p_perm** | **0.0000** |
| Wall-clock totale | ~451 min |
| Verdict pipeline | **PASS** ✅ |

## Best combos (top 5)

| Atlas | Metric | Band | Classifier | Bal.Acc | CI 95% | p_perm |
|-------|--------|------|------------|---------|--------|--------|
| aparc | plv | theta | logreg | **0.920** | [0.870,0.970] | 0.0000 |
| aparc | plv | alpha | logreg | **0.920** | [0.870,0.970] | 0.0000 |
| aparc | plv | alpha | svm_rbf | **0.920** | [0.870,0.970] | 0.0000 |
| aparc | coh | theta | svm_rbf | **0.890** | [0.830,0.940] | 0.0000 |
| aparc | plv | theta | svm_rbf | **0.890** | [0.830,0.950] | 0.0000 |

