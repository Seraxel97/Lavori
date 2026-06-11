# ML Scientific Dashboard — Tesi 2.0

Dashboard interattiva per esplorare i risultati di classificazione EEG source-level
(ds005385, N=50, EC vs EO, LOSO GroupKFold).

## Avvio rapido

```bash
cd /home/seraxel/Scrivania/Tesi_2.0
streamlit run dashboard/app.py --server.port 8501
```

Apri: http://localhost:8501

## Funzionalità

- **Selettori**: parcellation × connectivity metric × band × classifier
- **Run combo**: LOSO CV + bootstrap CI + ROC in <2s per logreg
- **Load reports/**: carica risultati ml_results.json esistenti
- **Load N=50 winner**: aparc×plv×theta×logreg, BA=0.920, p<0.001
- **Plot**: confusion matrix, ROC+AUC, learning curve per fold, permutation null, feature importance (logreg coef)
- **Export**: PNG/SVG/PDF pub-quality via matplotlib (300 DPI)

## Dipendenze

```bash
pip install -r dashboard/requirements.txt
```

## Dataset richiesti

```
data/features/ds005385/X_{atlas}_{metric}_{band}.npz
data/features/ds005385/y.npy
data/features/ds005385/groups.npy
```

Combo disponibili: aparc/schaefer100 × plv/wpli/coh/imcoh × theta/alpha/beta/gamma.
