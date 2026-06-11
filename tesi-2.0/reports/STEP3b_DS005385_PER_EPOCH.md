# STEP 3b Per-Epoch Parcellation — ds005385 PILOT

**Timestamp**: 2026-05-02T03:28:45+00:00
**Sprint**: S-103b (sonnet1-ts)
**Soggetti**: ['sub-007', 'sub-010', 'sub-011', 'sub-026', 'sub-031']
**Atlases**: ['aparc', 'schaefer100']  **Mode**: mean_flip

---

## Risultati (20 estrazioni)

| Sub | Atlas | Cond | Shape (ep,lbl,t) | ep0 mean | Time (s) |
|-----|-------|------|-----------------|----------|----------|
| sub-007 | aparc | EO | (122,68,501) | 9.1258 | 13.2 |
| sub-007 | aparc | EC | (124,68,501) | 8.585 | 12.3 |
| sub-007 | schaefer100 | EO | (122,100,501) | 9.1311 | 11.2 |
| sub-007 | schaefer100 | EC | (124,100,501) | 8.9326 | 12.3 |
| sub-010 | aparc | EO | (122,68,501) | 6.659 | 10.4 |
| sub-010 | aparc | EC | (122,68,501) | 14.5185 | 12.6 |
| sub-010 | schaefer100 | EO | (122,100,501) | 6.8052 | 11.2 |
| sub-010 | schaefer100 | EC | (122,100,501) | 14.9995 | 11.5 |
| sub-011 | aparc | EO | (121,68,501) | 10.3204 | 11.2 |
| sub-011 | aparc | EC | (120,68,501) | 10.1632 | 10.1 |
| sub-011 | schaefer100 | EO | (121,100,501) | 9.8087 | 10.9 |
| sub-011 | schaefer100 | EC | (120,100,501) | 10.2279 | 11.5 |
| sub-026 | aparc | EO | (122,68,501) | 35.7869 | 10.9 |
| sub-026 | aparc | EC | (124,68,501) | 7.5351 | 10.7 |
| sub-026 | schaefer100 | EO | (122,100,501) | 40.7446 | 10.6 |
| sub-026 | schaefer100 | EC | (124,100,501) | 7.4728 | 11.4 |
| sub-031 | aparc | EO | (120,68,501) | 9.7103 | 10.1 |
| sub-031 | aparc | EC | (120,68,501) | 10.6361 | 10.8 |
| sub-031 | schaefer100 | EO | (120,100,501) | 8.52 | 10.3 |
| sub-031 | schaefer100 | EC | (120,100,501) | 11.518 | 10.8 |

---

## Esempio — sub-007 × aparc × EO

**File**: `data/label_ts/ds005385/sub-007_atlas-aparc_cond-EO_per-epoch.npz`
**Shape**: `label_tc = (122, 68, 501)`
**Epoch means**: ep0=9.1258, ep1=8.9182, ep2=14.4307

```python
import numpy as np
d = np.load('data/label_ts/ds005385/sub-007_atlas-aparc_cond-EO_per-epoch.npz', allow_pickle=True)
# Keys: label_tc, label_names, subject, condition, atlas, mode, n_epochs, sfreq
# label_tc.shape = (122, 68, 501)
```

---

## Summary

| Metrica | Valore |
|---------|--------|
| File .npz prodotti | 20 / 20 |
| Wall-clock totale | 226.2s |
| Verdict | **PASS** ✅ |

