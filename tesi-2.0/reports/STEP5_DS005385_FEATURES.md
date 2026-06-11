# STEP 5 Feature Extraction — ds005385 PILOT

**Timestamp**: 2026-05-02T13:20:03+00:00
**Sprint**: S-105 (sonnet1-ts)

## Configurazione

- Soggetti: ['sub-007', 'sub-010', 'sub-011', 'sub-026', 'sub-031']
- Condizioni: EO=0, EC=1
- y = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
- groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

---

## Output files

| Atlas | Metric | Shape X | n_features | Range | Mean | NaN | Time (s) |
|-------|--------|---------|------------|-------|------|-----|----------|
| aparc | wpli | [10, 2278] | 2278 | [0.0159, 0.8080] | 0.1662 | 0 | 0.007 |
| aparc | coh | [10, 2278] | 2278 | [0.0099, 0.9790] | 0.1996 | 0 | 0.004 |
| schaefer100 | wpli | [10, 4950] | 4950 | [0.0109, 0.8783] | 0.1668 | 0 | 0.007 |
| schaefer100 | coh | [10, 4950] | 4950 | [0.0086, 0.9916] | 0.1999 | 0 | 0.005 |

---

## Sanity checks

- y unique: [0, 1] → {0: 5, 1: 5} (bilanciato ✅)
- groups unique: [0, 1, 2, 3, 4] → 5 soggetti ✅
- NaN count = 0 per tutte le combinazioni ✅

## File prodotti

```
data/features/ds005385/X_aparc_wpli_alpha.npz     shape=[10, 2278]
data/features/ds005385/X_aparc_coh_alpha.npz      shape=[10, 2278]
data/features/ds005385/X_schaefer100_wpli_alpha.npz shape=[10, 4950]
data/features/ds005385/X_schaefer100_coh_alpha.npz  shape=[10, 4950]
data/features/ds005385/y.npy                       shape=(10,)
data/features/ds005385/groups.npy                  shape=(10,)
data/features/ds005385/metadata.json
```

---

## Summary

| Metrica | Valore |
|---------|--------|
| File X prodotti | 4 / 4 |
| Wall-clock totale | 0.06s |
| Verdict | **PASS** ✅ |

