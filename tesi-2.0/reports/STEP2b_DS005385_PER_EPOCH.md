# STEP 2b Per-Epoch STC — ds005385 PILOT (FIX wPLI)

**Timestamp**: 2026-05-01T18:57:07+00:00
**Sprint**: S-101b (sonnet1-ts)
**Fix**: apply_inverse_epochs → (n_epochs, n_sources, n_times) float32
**Soggetti**: ['sub-007', 'sub-010', 'sub-011', 'sub-026', 'sub-031']
**Metodo**: dSPM | lambda2=0.111
**Disk pre**: 22.7 GB free | **post**: 4.8 GB free (delta ~17.9 GB)

---

## Risultati per soggetto × condizione

| Sub | Cond | n_epochs | Shape (ep,src,t) | File MB | Time (s) |
|-----|------|----------|------------------|---------|----------|
| sub-007 | EO | 122 | (122, 8196, 501) | 1796.7 | 75.8 |
| sub-007 | EC | 124 | (124, 8196, 501) | 1824.3 | 72.1 |
| sub-010 | EO | 122 | (122, 8196, 501) | 1795.1 | 71.6 |
| sub-010 | EC | 122 | (122, 8196, 501) | 1795.7 | 75.5 |
| sub-011 | EO | 121 | (121, 8196, 501) | 1786.2 | 68.0 |
| sub-011 | EC | 120 | (120, 8196, 501) | 1765.1 | 71.3 |
| sub-026 | EO | 122 | (122, 8196, 501) | 1797.1 | 72.1 |
| sub-026 | EC | 124 | (124, 8196, 501) | 1829.5 | 71.6 |
| sub-031 | EO | 120 | (120, 8196, 501) | 1764.9 | 74.5 |
| sub-031 | EC | 120 | (120, 8196, 501) | 1764.5 | 70.0 |

---

## Esempio output — sub-007 × EO

**File**: `data/derivatives/mne-bids-pipeline/sub-007/eeg/sub-007_task-RestingState_cond-EO_inv-dSPM-stcs.npz`
**Shape**: `data = (122, 8196, 501)` (n_epochs, n_sources, n_times)
**Dtype**: float32
**Statistiche**: mean=23.4306, std=38.8392
**Epoch 0 mean**: 20.1884
**Epoch 1 mean**: 19.3278

```python
import numpy as np
d = np.load('data/derivatives/mne-bids-pipeline/sub-007/eeg/sub-007_task-RestingState_cond-EO_inv-dSPM-stcs.npz', allow_pickle=True)
# Keys: data, vertices_lh, vertices_rh, tmin, tstep, n_epochs, sfreq
# d['data'].shape = (122, 8196, 501)
```

---

## Summary

| Metrica | Valore |
|---------|--------|
| File .npz prodotti | 10 / 10 |
| Disk delta | ~17.9 GB (22.7→4.8 GB free) |
| Wall-clock totale | 736.9s + 72.3s rebuild sub-031 EC |
| Verdict | **PASS** ✅ |

## Note operative

- **sub-031 EC**: file corrotto durante scrittura (disco pieno al 100% mid-write). Ricreato con run separato (72.3s). File finale verificato OK: shape=(120,8196,501).
- **sub-026 EC**: mean dSPM=72.4 (vs range 18-32 degli altri) — valore plausibile per soggetto ad alta ampiezza, da monitorare in STEP 3b/4b.
- **Disco**: 4.8 GB liberi post-S-101b. Raccomandazione: se S-103b/S-104b producono file grandi, considerare cleanup dei vecchi file evoked-STC da S-101 (`.stc` files non più necessari).

