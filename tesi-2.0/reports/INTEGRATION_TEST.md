# Integration Test Report — S-56

**Data**: 2026-05-01  
**Worker**: sonnet1-ts  
**Branch**: quality/integration-final  
**Verdict**: PASS

---

## Test eseguiti

| Test | Risultato | Tempo |
|------|-----------|-------|
| `test_e2e_matchingpennies_full` | PASS | ~8s |
| `test_mock_dataset_smoke` | PASS | ~2s |
| **Totale** | **2/2 PASS** | **~10s** |

---

## test_e2e_matchingpennies_full

Pipeline completa su `sub-05` reale (matchingpennies).

| Parametro | Valore |
|-----------|--------|
| Atlas | aparc (68 ROI) |
| Metrica FC | wpli |
| Epoch max | 8 |
| Bande FC | alpha (8-13 Hz) |
| CV splits | 2 |

**Output verificati**:
- `n_epochs = 8` (≥1 ✓)
- `label_tc.shape = (8, 68, 351)` — 8 STC prodotte (≥1 ✓)
- `FC alpha: (68, 68), mean_upper=0.385` — 1 FC matrix (≥1 ✓)
- `n_features = 3094`
- Risultati ML (≥1 prediction ✓): logreg BA=0.50, svm BA=0.50, mlp BA=0.25, rf BA=0.50, gb BA=0.50
- Report E2E scritto ✓

**Bug corretto**: `run_e2e_matchingpennies.py:141` — iterazione `for i, ep in enumerate(epochs)` restituiva `ndarray` anziché `Epochs`. Corretto con `epochs[i].average()`.

---

## test_mock_dataset_smoke

Generazione 2 soggetti mock BIDS (generate_mock_bids, S-29).

- `dataset_description.json` ✓
- `participants.tsv` ✓
- `sub-01/eeg/*.vhdr` ✓
- `validate_bids` → OK ✓

---

## Vincoli rispettati

- Wall time: 10.6s << 900s ✓
- 2 file output: `tests/test_integration_final.py` + `reports/INTEGRATION_TEST.md` ✓
- pytest standard ✓
