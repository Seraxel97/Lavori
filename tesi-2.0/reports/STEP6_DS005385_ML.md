# STEP 6 ML Classification — ds005385 PILOT

**Timestamp**: 2026-05-02T13:23:51+00:00
**Sprint**: S-106 (sonnet1-ts)

> ⚠️ **n=10 campioni, 5-fold LOSO = 1 test sample per fold.**
> Risultati INDICATIVI per pipeline validation. Non statisticamente robusti.

---

## Risultati aggregate (balanced_accuracy ± std)

### aparc × wpli

| Classifier | Accuracy | Bal.Acc | AUC | F1 |
|------------|----------|---------|-----|----|
| logreg | 0.700±0.245 | 0.700±0.245 | 1.000±0.000 | 0.533±0.452 |
| svm_rbf | 0.700±0.245 | 0.700±0.245 | 0.000±0.000 | 0.667±0.365 |
| lda | 0.700±0.245 | 0.700±0.245 | 0.800±0.400 | 0.533±0.452 |

### aparc × coh

| Classifier | Accuracy | Bal.Acc | AUC | F1 |
|------------|----------|---------|-----|----|
| logreg | 0.800±0.245 | 0.800±0.245 | 1.000±0.000 | 0.733±0.389 |
| svm_rbf | 0.800±0.245 | 0.800±0.245 | 0.400±0.490 | 0.733±0.389 |
| lda | 0.700±0.245 | 0.700±0.245 | 1.000±0.000 | 0.667±0.365 |

### schaefer100 × wpli

| Classifier | Accuracy | Bal.Acc | AUC | F1 |
|------------|----------|---------|-----|----|
| logreg | 0.700±0.245 | 0.700±0.245 | 1.000±0.000 | 0.533±0.452 |
| svm_rbf | 0.600±0.200 | 0.600±0.200 | 0.000±0.000 | 0.600±0.327 |
| lda | 0.700±0.245 | 0.700±0.245 | 0.800±0.400 | 0.533±0.452 |

### schaefer100 × coh

| Classifier | Accuracy | Bal.Acc | AUC | F1 |
|------------|----------|---------|-----|----|
| logreg | 0.900±0.200 | 0.900±0.200 | 1.000±0.000 | 0.933±0.133 |
| svm_rbf | 0.800±0.245 | 0.800±0.245 | 0.200±0.400 | 0.733±0.389 |
| lda | 0.800±0.245 | 0.800±0.245 | 0.800±0.400 | 0.867±0.163 |

---

## Best classifier per atlas×metric (balanced_accuracy)

| Atlas | Metric | Best clf | Bal.Acc |
|-------|--------|----------|---------|
| aparc | wpli | logreg | 0.700 |
| aparc | coh | logreg | 0.800 |
| schaefer100 | wpli | logreg | 0.700 |
| schaefer100 | coh | logreg | 0.900 |

---

## Summary

| Metrica | Valore |
|---------|--------|
| Combinazioni testate | 12 (4 atlas×metric × 3 clf) |
| Wall-clock totale | 0.36s |
| Nota | n=10, risultati indicativi |
| Figura | reports/figures/fig04_ml.png |
| Verdict | **PASS** ✅ |

