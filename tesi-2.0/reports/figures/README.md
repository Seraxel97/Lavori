# Manuscript Figures — Status Overview

**Progetto**: Tesi_2.0 EEG Source-Level FC Classification  
**Data ultimo aggiornamento**: 2026-05-01  
**Sprint**: S-26

Tutte le figure sono in stato `pending` fino all'esecuzione del benchmark completo (S-08)
su dataset multi-soggetto (S-07, Q1 pendente). Le figure che richiedono solo dati
gia' disponibili (fig01, fig02, fig03) possono essere generate in qualsiasi momento.

---

## Tabella stato

| File | Figura | Titolo | Status | Dati disponibili? | Tool |
|------|--------|--------|--------|-------------------|------|
| `fig01_pipeline.placeholder` | Fig. 1 | Pipeline overview diagram | pending | Si (schematic) | draw.io / matplotlib |
| `fig02_topo.placeholder` | Fig. 2 | EEG topoplot sensor-level | pending | Si (ave.fif sub-05) | mne.viz |
| `fig03_fc.placeholder` | Fig. 3 | FC matrix heatmap (wpli, aparc) | pending | Si (label_tc sub-05) | seaborn heatmap |
| `fig04_ml.placeholder` | Fig. 4 | Confusion matrix + ROC curves | pending | No (attende S-08 multi-subj) | sklearn.metrics |
| `fig05_results.placeholder` | Fig. 5 | Top-10 BA benchmark bar chart | pending | No (attende S-08 full run) | matplotlib + pandas |

---

## Note per generazione

### Figure generabili ora (dati sub-05 disponibili)

**Fig. 1** — Diagramma pipeline: non richiede dati, e' uno schematic. Candidato per
generazione manuale con draw.io o con un semplice script matplotlib (flow chart).

**Fig. 2** — Topoplot EEG: richiede `sub-05_task-matchingpennies_ave.fif` (presente in
`data/derivatives/mne-bids-pipeline/sub-05/eeg/`). Generabile con `mne.Evoked.plot_topomap`.

**Fig. 3** — FC heatmap: richiede l'esecuzione di `connectivity/fc_dispatcher.compute_fc`
sul label_tc di sub-05 (circa 30 s con l'inversione su 20 epoch). Generabile con seaborn.

### Figure che attendono il benchmark completo

**Fig. 4** e **Fig. 5** richiedono:
1. Dataset multi-soggetto (S-07, Q1 operatore pendente)
2. Esecuzione `pipeline_mne_bids/run_bench_matrix.py` (700 run, stimati 2-3 h)
3. `reports/BENCH_MATRIX_RESULTS.json` popolato con risultati reali

---

## Path di output attesi

```
reports/figures/
    fig01_pipeline.png    (300 dpi, 180mm larghezza, due colonne IEEE)
    fig02_topo.png        (300 dpi, 90mm larghezza, una colonna IEEE)
    fig03_fc.png          (300 dpi, 180mm larghezza)
    fig04_ml.png          (300 dpi, 180mm larghezza)
    fig05_results.png     (300 dpi, 180mm larghezza)
```

---

## Riferimenti incrocio manoscritto

| Figura | Sezione MAIN_v1.md |
|--------|--------------------|
| Fig. 1 | §2 Methods — pipeline overview |
| Fig. 2 | §2.2 Preprocessing — sensor-level EEG |
| Fig. 3 | §2.5 Functional connectivity |
| Fig. 4 | §3.1 Benchmark results |
| Fig. 5 | §3.2 Top-10 configurations |
