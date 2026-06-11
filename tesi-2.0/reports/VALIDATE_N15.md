# N=15 Outputs — Validation Report

**Timestamp**: 2026-05-05T20:57:34.907527+02:00  
**Sprint**: S-VALIDATE-N15  
**Data root**: `data/{label_ts,connectivity,features}/ds005385/`  
**Subjects**: 15 (sub-007, sub-010, sub-011, sub-026, sub-031, ...)

## Summary

| Categoria | File attesi | Trovati | Validi | Anomalie |
|-----------|-------------|---------|--------|----------|
| label_tc per-epoch | 60 | 60 | 60 | 0 |
| FC matrices (per-epoch) | 120 | 120 | 120 | 0 |
| Feature matrices X | 4 | 4 | 4 | 0 |

**Verdict**: ✅ **PASS**

## Anomalie dettagliate

Nessuna anomalia rilevata.

## Range fisiologico — wPLI off-diagonal (per sub, atlas=aparc)

| Sub | EO median | EC median | EO max | EC max |
|-----|-----------|-----------|--------|--------|
| sub-007 | 0.1255 | 0.1077 | 0.5215 | 0.4349 |
| sub-010 | 0.1791 | 0.1597 | 0.8341 | 0.7977 |
| sub-011 | 0.1496 | 0.1241 | 0.5957 | 0.4868 |
| sub-026 | 0.1981 | 0.1748 | 0.6766 | 0.8783 |
| sub-031 | 0.1133 | 0.1248 | 0.6311 | 0.7873 |
| sub-033 | 0.1215 | 0.1375 | 0.6005 | 0.7801 |
| sub-041 | 0.1360 | 0.2014 | 0.5039 | 0.7854 |
| sub-066 | 0.1285 | 0.1442 | 0.5145 | 0.7549 |
| sub-071 | 0.1073 | 0.1829 | 0.4317 | 0.7329 |
| sub-080 | 0.1393 | 0.1416 | 0.5156 | 0.9456 |
| sub-125 | 0.1231 | 0.1176 | 0.6803 | 0.4715 |
| sub-157 | 0.1389 | 0.1141 | 0.7685 | 0.5040 |
| sub-169 | 0.1173 | 0.1173 | 0.4836 | 0.5278 |
| sub-185 | 0.1470 | 0.1094 | 0.5474 | 0.6528 |
| sub-195 | 0.1243 | 0.1257 | 0.4841 | 0.8460 |

## Range — coh off-diagonal (per sub, atlas=aparc)

| Sub | EO median | EC median |
|-----|-----------|-----------|
| sub-007 | 0.1232 | 0.0591 |
| sub-010 | 0.2493 | 0.1015 |
| sub-011 | 0.2403 | 0.0679 |
| sub-026 | 0.4288 | 0.1186 |
| sub-031 | 0.0601 | 0.0744 |
| sub-033 | 0.0742 | 0.0684 |
| sub-041 | 0.2091 | 0.3858 |
| sub-066 | 0.1187 | 0.1449 |
| sub-071 | 0.0594 | 0.3117 |
| sub-080 | 0.2555 | 0.0980 |
| sub-125 | 0.0869 | 0.0612 |
| sub-157 | 0.1978 | 0.0623 |
| sub-169 | 0.0546 | 0.0669 |
| sub-185 | 0.2887 | 0.0550 |
| sub-195 | 0.0667 | 0.0675 |

## Note

- Validazione read-only — nessuna modifica ai dati.
- label_tc: chiave `label_tc`, shape `(n_epochs, n_labels, n_times)`.
- FC: chiave `fc_matrix`, simmetria verificata (`max|M - M^T| < 1e-6`).
- Features: `X` da npz, `y`/`groups` da `.npy` separati.
- wPLI atteso: off-diagonal in [0, 1], range fisiologico ~[0.05, 0.50].
- coh atteso: off-diagonal in [0, 1].
