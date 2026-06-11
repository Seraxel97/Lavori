# F1 Sandbox — Report Calibrazione

**Data run**: `<run_ts>`
**Nicchia**: `<nicchia>`
**Branch**: `seraxel/selly-f1-sandbox-sim`

## Metriche di calibrazione

| Metrica | Valore |
|---|---|
| Trade simulati | `<n_trades_executed>` / `<n_trades_target>` |
| Prediction accuracy rate | `<prediction_accuracy_rate>` |
| MAE (mean absolute error) | `<mean_abs_error>` |
| Mean error (bias) | `<mean_error>` |
| Bootstrap CI 95% | `[<bootstrap_CI_95[0]>, <bootstrap_CI_95[1]>]` |
| Sign test p-value | `<sign_test_p>` |
| **Calibration PASS** | `<calibration_pass>` |

## Interpretazione

- **Accuracy > 0.6** + **p < 0.05** → modello better-than-random → VERDE per F2 live-paper
- **Accuracy ≤ 0.6** o **p ≥ 0.05** → ricalibrare threshold / features prima di F2
- **MAE < 0.5** → errore medio contenuto (accettabile per arbitrage a basso volume)

## Trade sample (top 5 per predicted ROI)

```json
<top_5_trades_json>
```

## Conclusioni

<!-- Compilare dopo aver visto i risultati -->

- Threshold ROI 1.0 è [ ] troppo aggressivo / [ ] calibrato / [ ] troppo conservativo
- Source dati più affidabile: `<source>`
- Azione raccomandata per F2: `<azione>`

## Prossimi step

- [ ] F2: live-paper trading con €0 reali (listing manuale simulato)
- [ ] F3: live trading con budget operatore autorizzato
