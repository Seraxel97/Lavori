# Performance Profile — Bench Matrix Hot Path (S-17)

**Data**: 2026-05-01  
**Strumento**: cProfile (stdlib), Python 3.13.12  
**Script**: `analysis/profile_bench.py`  
**Stats raw**: `reports/PROFILE_STATS.txt`

---

## Configurazione del profiling

| Parametro | Valore |
|-----------|--------|
| Atlas | `aparc` (68 ROI) |
| Metriche FC | `wpli`, `plv` (2) |
| Algoritmo ML | `logreg` |
| Bande | `alpha`, `beta` (2) |
| Run totali profiled | 4 (2 metriche × 2 bande) |
| Epoch | 20 (sub-set; sfreq decimata a 500 Hz) |
| Soggetto | `sub-05` matchingpennies |
| **Wall-clock totale** | **8.83 s** |
| Function calls totali | 2 569 306 |

---

## Top-10 Hot Path (per cumulative time)

| Rank | Funzione | Modulo | ncalls | cumtime (s) | % tot |
|------|----------|--------|--------|-------------|-------|
| 1 | `extract_tc` | `parcellation/extract_label_tc.py:78` | 20 | 4.09 | 46.3% |
| 2 | `extract_label_time_course` | `mne/source_estimate.py:3747` | 20 | 3.57 | 40.5% |
| 3 | `_gen_extract_label_time_course` | `mne/source_estimate.py:3653` | 40 | 3.57 | 40.5% |
| 4 | `apply_inverse` | `mne/minimum_norm/inverse.py:909` | 20 | 3.47 | 39.3% |
| 5 | `_apply_inverse` | `mne/minimum_norm/inverse.py:1039` | 20 | 3.47 | 39.3% |
| 6 | `_prepare_label_extraction` | `mne/source_estimate.py:3426` | 20 | 3.35 | 38.0% |
| 7 | `intersect1d` | `numpy/lib/_arraysetops_impl.py:666` | 2732 | 2.97 | 33.6% |
| 8 | `prepare_inverse_operator` | `mne/minimum_norm/inverse.py:596` | 20 | 2.12 | 24.0% |
| 9 | `label_sign_flip` | `mne/label.py:1448` | 1360 | 1.72 | 19.5% |
| 10 | `compute_fc` | `connectivity/fc_dispatcher.py:47` | 4 | 0.53 | 6.0% |

---

## Analisi dei bottleneck

### Bottleneck #1 — `_prepare_label_extraction` + `intersect1d` (38% del tempo)

**Causa**: `mne.extract_label_time_course` chiama `_prepare_label_extraction` per ogni epoch, che a sua volta esegue 2732 chiamate a `numpy.intersect1d` per trovare i vertici di ogni label nel source space. Questo calcolo è **identico per ogni epoch** (le label non cambiano tra un'epoch e l'altra), ma viene ripetuto 20 volte.

**Impatto**: ~3.35 s su 8.83 s totali (38%).

### Bottleneck #2 — `prepare_inverse_operator` (24% del tempo)

**Causa**: `apply_inverse` chiama `prepare_inverse_operator` ad ogni epoch (20 chiamate, 0.1 s/call = 2.0 s totali). Questa funzione ricalcola la decomposizione dell'operatore inverso (SVD, whitening) che è invariante tra le epoch. Non viene cachata internamente da MNE quando si chiama `apply_inverse` su oggetti `Epochs` singoli.

**Impatto**: ~2.12 s su 8.83 s (24%).

### Bottleneck #3 — `label_sign_flip` + `sort` (19.5% del tempo)

**Causa**: per la modalità `mean_flip`, MNE calcola il flip di segno ottimale per ogni label in ogni epoch tramite SVD locale (1360 chiamate = 68 label × 20 epoch). Il sort interno su array numpy (~16k chiamate) è correlato all'ordinamento dei vertici in `_gen_extract_label_time_course`.

**Impatto**: ~1.72 s su 8.83 s (19.5%).

### Bottleneck #4 — `compute_fc / spectral_connectivity_epochs` (6%)

**Causa**: la stima spettrale multitaper su (20 epoch, 68 ROI, 351 timepoints) per 4 combinazioni (metrica × banda). Questo tempo è già relativamente contenuto grazie alla FFT su dati decimati (500 Hz).

**Impatto**: ~0.53 s — trascurabile rispetto agli altri bottleneck.

---

## Proiezione per il benchmark completo (700 run)

Con i dati del profiling su 20 epoch e 4 run:

| Scenario | Tempo stimato |
|----------|---------------|
| 4 run (profiling subset) | 8.8 s |
| 700 run, 20 epoch, 1 atlas | ~1540 s (26 min) |
| 700 run, 20 epoch, 4 atlas | ~6160 s (103 min) |
| 700 run, 100 epoch, 4 atlas | ~30800 s (8.5 h) |

Il bottleneck `prepare_inverse_operator` scala linearmente con `n_epochs × n_atlas`.  
Il bottleneck `_prepare_label_extraction` scala con `n_epochs × n_atlas × n_roi` (peggiora con atlanti più granulari).

---

## Raccomandazioni di ottimizzazione

### 1. Cache `prepare_inverse_operator` (impatto: −24%)

Passare l'operatore già preparato a `apply_inverse` tramite il parametro `prepared=True` dopo una chiamata a `mne.minimum_norm.prepare_inverse_operator()` all'esterno del loop. Questo elimina le 20 re-preparazioni per soggetto.

```python
inv_prepared = mne.minimum_norm.prepare_inverse_operator(
    inv_op, nave=1, lambda2=lambda2, method=method
)
stc = mne.minimum_norm.apply_inverse(evoked, inv_prepared, prepared=True, ...)
```

**Risparmio stimato**: ~2.0 s su 8.8 s (−23%).

### 2. Cache `_prepare_label_extraction` tra epoch (impatto: −38%)

Il calcolo `intersect1d` per trovare i vertici di ciascuna label è identico per ogni epoch. MNE non esegue questo caching internamente. La soluzione è estrarre i time course dall'array STC direttamente, pre-calcolando i vertici-per-label una volta sola:

```python
# Pre-calcola una volta
label_verts = [np.intersect1d(lbl.vertices, src_verts) for lbl in labels]
# Per ogni epoch: indicizzazione diretta invece di extract_label_time_course
tc = np.array([stc.data[verts].mean(axis=0) for verts in label_verts])
```

Alternativa: usare `extract_label_time_course` con `src` fisso e sfruttare il parametro `return_generator=True` disponibile in MNE >= 1.6 per lazy evaluation.

**Risparmio stimato**: ~3.0 s su 8.8 s (−34%).

### 3. Parallelismo per le epoch (`n_jobs`)

Il loop `for i in range(len(epochs))` è embarrassingly parallel. Usare `joblib.Parallel` o il parametro `n_jobs` nativo di alcune funzioni MNE:

```python
from joblib import Parallel, delayed
tc_list = Parallel(n_jobs=-1)(
    delayed(_process_epoch)(epochs[i], inv_prepared, labels, label_verts)
    for i in range(len(epochs))
)
```

Su sistemi multi-core (4–8 core) il guadagno teorico per il loop è 4–8×, ma limitato dall'overhead GIL di Python su operazioni non NumPy-pure. Stima realistica: 2–3× speedup.

**Risparmio stimato**: 50–66% del tempo di inversione/parcellazione (con n_jobs=4).

### 4. Pre-calcolo label_tc per atlas (già nel bench, ma verificare caching)

`run_bench_matrix.py` già calcola `label_tc` una volta per atlas e lo riusa per tutte le metriche × bande. Questo è il design ottimale. Verificare che il tensor `label_tc` non venga copiato inutilmente nelle chiamate a `compute_fc` e `build_X`.

### 5. Riduzione sfreq più aggressiva per FC

La decimazione a 500 Hz è già ottimale per bande fino a gamma (45 Hz, ben al di sotto di Nyquist = 250 Hz). Una riduzione a 250 Hz taglierebbe a metà il numero di campioni e accelererebbe sia l'inversione che la FC senza perdere informazione per le 5 bande standard.

**Risparmio stimato**: ~30% su `_apply_inverse` e `compute_fc`.

---

## Riepilogo impatti

| Raccomandazione | Impatto atteso | Complessità |
|----------------|----------------|-------------|
| Cache `prepare_inverse_operator` | −23% | Bassa (1 riga) |
| Cache `_prepare_label_extraction` | −34% | Media (refactor loop) |
| Parallelismo epoch (n_jobs=4) | −50% loop inv | Media (joblib) |
| Sfreq 250 Hz invece di 500 | −30% inv+FC | Bassa (1 param) |
| Combinazione 1+2+3 | **−70–80% totale** | Media |

Con tutte le ottimizzazioni applicate, il benchmark completo (700 run, 100 epoch) scenderebbe da ~8.5 h a ~2–3 h su hardware standard.
