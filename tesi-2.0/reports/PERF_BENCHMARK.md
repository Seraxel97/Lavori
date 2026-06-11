# Performance Benchmark — Pipeline matchingpennies

**Sprint**: S-48 → aggiornato S-PERF-INVPARC  
**Data baseline**: 2026-05-01 | **Data refactor**: 2026-05-05  
**Worker baseline**: sonnet2-ts | **Worker refactor**: sonnet-tesi-2  
**Soggetto**: sub-05  
**Atlas**: aparc (Desikan-Killiany, 68 ROI)  
**Metrica FC**: wPLI  
**N epoch benchmark**: 20 (subset limitato per timing; full pipeline usa tutte le epoch)

---

## Tempi per step — POST-REFACTOR (S-PERF-INVPARC)

| Step | Prima (s) | Dopo (s) | Speedup | Note |
|------|-----------|----------|---------|------|
| `load_epochs` | 0.062 | 0.060 | 1.0x | — |
| `build_y` | 0.001 | 0.001 | 1.0x | — |
| `inverse_parcellation` | **5.490** | **1.533** | **3.58x** | Batch `apply_inverse_epochs` + `extract_tc_batch` |
| `functional_connectivity` | 0.316 | 0.322 | 1.0x | — |
| `feature_extraction` | 0.191 | 0.040 | 4.8x | Varianza run-to-run |
| `ml_training` | 3.797 | 4.020 | 1.0x | Varianza run-to-run |
| **TOTALE** | **9.857** | **5.976** | **1.65x** | Riduzione 39% |

**Speedup `inverse_parcellation`: 3.58x — riduzione 72% (target era 40%)**

---

## Refactor S-PERF-INVPARC — dettaglio

### Cosa è cambiato

**Prima** (S-48, codice baseline):
```python
for i in range(len(epochs)):
    stc = apply_inverse(epochs[i].average(), inv_op, ...)
    tc_i, _ = extract_tc(stc, atlas, src)
    tc_list.append(tc_i)
label_tc = np.stack(tc_list, axis=0)
```

**Dopo** (S-PERF-INVPARC):
```python
stcs = list(apply_inverse_epochs(epochs, inv_op, ...))  # 1 call batch
label_tc, _ = extract_tc_batch(stcs, atlas, src)         # 1 call batch
```

### Perché è più veloce

1. **`apply_inverse_epochs` vs loop di `apply_inverse`**: `apply_inverse_epochs` esegue una sola volta `prepare_inverse_operator` (costo O(1) invece di O(N_epochs)), poi applica la matrice inversa precomputata a tutte le epoch in sequenza.
2. **`extract_tc_batch` vs loop di `extract_tc`**: `mne.extract_label_time_course` chiamato con una lista di STC invece di N chiamate separate — la preparazione delle label (morfing) avviene una sola volta.

### Nuova funzione `extract_tc_batch`

Aggiunta a `parcellation/extract_label_tc.py`:
- Input: `list[SourceEstimate]`, `atlas`, `src`
- Output: `np.ndarray (n_stcs, n_labels, n_times)`, `list[str]`
- Usata da `bench_steps.py` (e disponibile per future pipeline)

---

## Step ancora ottimizzabili

### 1. `ml_training` — ora bottleneck dominante (67% del totale)

5 algoritmi × GroupKFold (5 fold) = 25 fit. Scala male con N soggetti.

**Soluzioni raccomandate**:
- Parallelizzare con `n_jobs=-1` dove supportato (sklearn)
- Separare fit-e-predict da scoring per early stopping

### 2. `functional_connectivity` — 5.4%

wPLI su bande alpha+beta: 0.3s per 20 epoch. Trascurabile. Non è bottleneck.

### 3. `load_epochs` + `build_y` — < 1%

I/O e preprocessing trascurabili. Nessuna ottimizzazione necessaria.

---

## Riproduzione

```bash
PYTHONPATH=/path/to/Tesi_2.0 python -m analysis.bench_steps \
  --subject sub-05 \
  --deriv-root data/derivatives/mne-bids-pipeline \
  --atlas aparc \
  --metric wpli \
  --output reports/PERF_BENCHMARK_raw.json
```

Raw JSON: `reports/PERF_BENCHMARK_raw.json`
