# Esperimenti N=15 — ds005385 Full Run (S-ML-EXTEND)

**Timestamp**: 2026-05-06T00:30+02:00
**Sprint**: S-ML-EXTEND (sonnet-tesi-2)
**Dataset**: ds005385 — N=15, seed=42 | Campioni: 30 (EO + EC)
**CV**: LOSO GroupKFold-15 | **n_permutations**: 100
**Classifiers**: logreg, svm_rbf, lda | **Bands**: theta, alpha, beta, gamma
**Combos testati**: 96 (2 atlas × 4 metric × 4 band × 3 clf)

---

## Winner

**aparc × coh × theta × svm_rbf**
- Balanced Accuracy: **0.867** ±0.221
- CI 95%: [0.766, 0.967]
- p_perm: **0.0000** ✅
- n_subjects: 15 | n_features: 2278

### Confronto con runs precedenti

| Run | Config | BA |
|-----|--------|----|
| PILOT N=5 (alpha) | schaefer100 × coh × alpha × logreg | 0.900 |
| N=15 alpha (3 clf) | aparc × coh × alpha × svm_rbf | 0.833 |
| **N=15 full 4-band (3 clf)** | **aparc × coh × theta × svm_rbf** | **0.867** |

---

## Top-10 configurazioni

| # | Atlas | Metric | Band | Classifier | BA | ±std | CI 95% | p_perm | Sig |
|---|-------|--------|------|------------|----|------|--------|--------|-----|
| 1 | aparc | coh | theta | svm_rbf | **0.867** | 0.221 | [0.766,0.967] | 0.0000 | ✅ |
| 2 | aparc | plv | theta | logreg | **0.867** | 0.287 | [0.700,1.000] | 0.0000 | ✅ |
| 3 | aparc | plv | alpha | logreg | **0.867** | 0.221 | [0.733,0.967] | 0.0000 | ✅ |
| 4 | aparc | plv | alpha | lda | **0.867** | 0.221 | [0.733,0.967] | 0.0000 | ✅ |
| 5 | aparc | coh | theta | lda | **0.833** | 0.236 | [0.700,0.933] | 0.0000 | ✅ |
| 6 | aparc | plv | theta | svm_rbf | **0.833** | 0.298 | [0.667,0.967] | 0.0000 | ✅ |
| 7 | schaefer100 | coh | theta | logreg | **0.833** | 0.236 | [0.700,0.933] | 0.0000 | ✅ |
| 8 | schaefer100 | plv | theta | logreg | **0.833** | 0.298 | [0.667,0.967] | 0.0000 | ✅ |
| 9 | schaefer100 | plv | theta | svm_rbf | **0.833** | 0.298 | [0.667,0.967] | 0.0000 | ✅ |
| 10 | schaefer100 | plv | theta | lda | **0.833** | 0.298 | [0.667,0.967] | 0.0000 | ✅ |

---

## Summary

| Metrica | Valore |
|---------|--------|
| Soggetti | 15 |
| Combos testati | 96 |
| Combos significativi (p<0.05) | ✅/96 |
| Winner | aparc × coh × theta × svm_rbf |
| Best BA | 0.867 CI=[0.766,0.967] |
| Best p_perm | 0.0000 |
| Wall-clock aggregate | 1676.7s (~28 min) |
| Verdict | **PASS** ✅ |

---

## Note metodologiche

- RF e GB esclusi da questo run (n_estimators=200 troppo lento per 96 combos × 100 perm).
- Theta band (4-8 Hz) emerge come banda più informativa — coerente con letteratura resting-state.
- coh e plv dominano rispetto a wpli e imcoh nelle configurazioni top.
- n_perm=100 (paper-grade: 1000 — da rieseguire su cluster con --n-perm 1000).
