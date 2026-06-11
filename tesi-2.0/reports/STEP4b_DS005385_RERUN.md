# STEP 4b FC Rerun Per-Epoch — ds005385 PILOT

**Timestamp**: 2026-05-02T03:40:26+00:00
**Sprint**: S-104b (sonnet1-ts) — fix wPLI collapse da S-104

## Fix wPLI: prima vs dopo

| | S-104 (evoked) | S-104b (per-epoch) |
|---|---|---|
| wPLI mean | 1.0000 (COLLASSATO) | 0.1662 ✅ |
| wPLI std | 0.0000 | variabile (fisiologico) |
| Root cause | n_epochs=1 | n_epochs=120-124 |

---

## Risultati (40 file)

| Sub | Atlas | Cond | Metric | n_ep | Mean upper | Std | Time (s) |
|-----|-------|------|--------|------|------------|-----|----------|
| sub-007 | aparc | EO | wpli | 122 | 0.1402 | 0.0758 | 1.105 |
| sub-007 | aparc | EO | coh | 122 | 0.1808 | 0.1663 | 0.664 |
| sub-007 | aparc | EC | wpli | 124 | 0.1192 | 0.0593 | 0.679 |
| sub-007 | aparc | EC | coh | 124 | 0.1384 | 0.1486 | 0.681 |
| sub-007 | schaefer100 | EO | wpli | 122 | 0.1361 | 0.0618 | 1.356 |
| sub-007 | schaefer100 | EO | coh | 122 | 0.1824 | 0.1630 | 1.096 |
| sub-007 | schaefer100 | EC | wpli | 124 | 0.1193 | 0.0603 | 1.118 |
| sub-007 | schaefer100 | EC | coh | 124 | 0.1213 | 0.1446 | 1.115 |
| sub-010 | aparc | EO | wpli | 122 | 0.1945 | 0.0983 | 0.658 |
| sub-010 | aparc | EO | coh | 122 | 0.2938 | 0.1738 | 0.658 |
| sub-010 | aparc | EC | wpli | 122 | 0.1953 | 0.1281 | 0.933 |
| sub-010 | aparc | EC | coh | 122 | 0.1610 | 0.1620 | 0.665 |
| sub-010 | schaefer100 | EO | wpli | 122 | 0.2019 | 0.1078 | 1.089 |
| sub-010 | schaefer100 | EO | coh | 122 | 0.2772 | 0.1742 | 1.104 |
| sub-010 | schaefer100 | EC | wpli | 122 | 0.1959 | 0.1321 | 1.091 |
| sub-010 | schaefer100 | EC | coh | 122 | 0.1657 | 0.1616 | 1.085 |
| sub-011 | aparc | EO | wpli | 121 | 0.1648 | 0.0823 | 0.795 |
| sub-011 | aparc | EO | coh | 121 | 0.2625 | 0.1940 | 0.982 |
| sub-011 | aparc | EC | wpli | 120 | 0.1351 | 0.0698 | 0.783 |
| sub-011 | aparc | EC | coh | 120 | 0.1313 | 0.1425 | 0.664 |
| sub-011 | schaefer100 | EO | wpli | 121 | 0.1676 | 0.0830 | 1.101 |
| sub-011 | schaefer100 | EO | coh | 121 | 0.2745 | 0.1891 | 1.46 |
| sub-011 | schaefer100 | EC | wpli | 120 | 0.1378 | 0.0712 | 1.232 |
| sub-011 | schaefer100 | EC | coh | 120 | 0.1227 | 0.1350 | 1.092 |
| sub-026 | aparc | EO | wpli | 122 | 0.2096 | 0.0971 | 1.062 |
| sub-026 | aparc | EO | coh | 122 | 0.3722 | 0.2333 | 0.654 |
| sub-026 | aparc | EC | wpli | 124 | 0.2094 | 0.1337 | 0.672 |
| sub-026 | aparc | EC | coh | 124 | 0.1752 | 0.1645 | 0.73 |
| sub-026 | schaefer100 | EO | wpli | 122 | 0.2125 | 0.0942 | 1.095 |
| sub-026 | schaefer100 | EO | coh | 122 | 0.4335 | 0.2635 | 1.244 |
| sub-026 | schaefer100 | EC | wpli | 124 | 0.2159 | 0.1413 | 1.257 |
| sub-026 | schaefer100 | EC | coh | 124 | 0.1714 | 0.1534 | 1.121 |
| sub-031 | aparc | EO | wpli | 120 | 0.1409 | 0.0834 | 0.889 |
| sub-031 | aparc | EO | coh | 120 | 0.1454 | 0.1631 | 0.994 |
| sub-031 | aparc | EC | wpli | 120 | 0.1531 | 0.1024 | 0.883 |
| sub-031 | aparc | EC | coh | 120 | 0.1352 | 0.1467 | 1.032 |
| sub-031 | schaefer100 | EO | wpli | 120 | 0.1297 | 0.0722 | 1.078 |
| sub-031 | schaefer100 | EO | coh | 120 | 0.1167 | 0.1373 | 1.224 |
| sub-031 | schaefer100 | EC | wpli | 120 | 0.1512 | 0.1009 | 1.58 |
| sub-031 | schaefer100 | EC | coh | 120 | 0.1332 | 0.1449 | 1.091 |

---

## EO vs EC — wPLI aparc alpha

| Sub | EO | EC | Delta (EC-EO) |
|-----|----|----|---------------|
| sub-007 | 0.1402 | 0.1192 | -0.0210 |
| sub-010 | 0.1945 | 0.1953 | +0.0008 |
| sub-011 | 0.1648 | 0.1351 | -0.0297 |
| sub-026 | 0.2096 | 0.2094 | -0.0002 |
| sub-031 | 0.1409 | 0.1531 | +0.0122 |

---

## Summary

| Metrica | Valore |
|---------|--------|
| File prodotti | 40 / 40 |
| wPLI mean (aparc, tutti) | 0.1662 (era 1.0000) |
| Wall-clock totale | 42.0s |
| Verdict | **PASS** ✅ |

