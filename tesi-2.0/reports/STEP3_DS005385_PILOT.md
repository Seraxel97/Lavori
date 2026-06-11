# STEP 3 Parcellation — ds005385 PILOT

**Timestamp**: 2026-05-01T17:46:54+00:00
**Sprint**: S-103 (sonnet1-ts)
**Soggetti**: ['sub-007', 'sub-010', 'sub-011', 'sub-026', 'sub-031']
**Atlases**: ['aparc', 'schaefer100']
**Condizioni**: ['EO', 'EC']
**Mode**: `mean_flip`

---

## Risultati (sub × atlas × cond)

| Sub | Atlas | Cond | n_labels | n_times | Wall (s) |
|-----|-------|------|----------|---------|----------|
| sub-007 | aparc | EO | 68 | 501 | 0.888 |
| sub-007 | aparc | EC | 68 | 501 | 0.176 |
| sub-007 | schaefer100 | EO | 100 | 501 | 0.197 |
| sub-007 | schaefer100 | EC | 100 | 501 | 0.195 |
| sub-010 | aparc | EO | 68 | 501 | 0.172 |
| sub-010 | aparc | EC | 68 | 501 | 0.171 |
| sub-010 | schaefer100 | EO | 100 | 501 | 0.195 |
| sub-010 | schaefer100 | EC | 100 | 501 | 0.195 |
| sub-011 | aparc | EO | 68 | 501 | 0.17 |
| sub-011 | aparc | EC | 68 | 501 | 0.169 |
| sub-011 | schaefer100 | EO | 100 | 501 | 0.197 |
| sub-011 | schaefer100 | EC | 100 | 501 | 0.195 |
| sub-026 | aparc | EO | 68 | 501 | 0.171 |
| sub-026 | aparc | EC | 68 | 501 | 0.168 |
| sub-026 | schaefer100 | EO | 100 | 501 | 0.199 |
| sub-026 | schaefer100 | EC | 100 | 501 | 0.367 |
| sub-031 | aparc | EO | 68 | 501 | 0.34 |
| sub-031 | aparc | EC | 68 | 501 | 0.181 |
| sub-031 | schaefer100 | EO | 100 | 501 | 0.516 |
| sub-031 | schaefer100 | EC | 100 | 501 | 0.197 |

---

## Esempio output — sub-007 × aparc × EO

**File**: `data/label_ts/ds005385/sub-007_atlas-aparc_cond-EO.npz`

```python
import numpy as np
d = np.load('data/label_ts/ds005385/sub-007_atlas-aparc_cond-EO.npz', allow_pickle=True)
# Keys: label_tc, label_names, atlas, subject, condition, task, mode
# label_tc.shape = (68, 501)
# label_names[:5] = ['bankssts-lh', 'bankssts-rh', 'caudalanteriorcingulate-lh', 'caudalanteriorcingulate-rh', 'caudalmiddlefrontal-lh']
```

---

## Summary

| Metrica | Valore |
|---------|--------|
| File .npz prodotti | 20 / 20 |
| Atlases | aparc, schaefer100 |
| Wall-clock totale | 5.2s |
| Verdict | **PASS** ✅ |

