# Cross-Dataset Transfer: ds005385 → ds004504

**Data**: 2026-06-04
**Feature**: X_aparc_plv_theta
**Train**: ds005385 N=200 osservazioni (N=100 soggetti)
**Test**: ds004504 N=49 osservazioni (N=49 soggetti)

## Risultati Transfer

| Task | Metrica | Transfer (train=ds005385, test=ds004504) | In-dataset ds005385 (5-fold CV) |
|------|---------|------------------------------------------|---------------------------------|
| Sesso | BA | 0.500 | 0.741 |
| Età | MAE (anni) | 22.25 | 12.52 |

## Interpretazione

- **Sesso transfer BA = 0.500**: vicino a chance (0.5). Generalizzazione limitata.
- **Età transfer MAE = 22.25 anni**: confronto con in-dataset 12.52 anni.
- Gap transfer vs in-dataset: sex BA +0.241 | age MAE +9.73

## Note metodologiche

- Scaler fit SOLO su train (ds005385), transform su test (ds004504) — no data leakage
- Feature: X_aparc_plv_theta — miglior combo da analisi N100 ds005385
- Dataset eterogeni: ds005385 = adulti sani (20–70 anni), ds004504 = Alzheimer + controlli (49–79 anni)
