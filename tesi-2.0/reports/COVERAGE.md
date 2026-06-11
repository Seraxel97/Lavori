# Coverage Report — Tesi_2.0

**Data**: 2026-05-01  
**Comando**: `bash scripts/run_coverage.sh`  
**pytest-cov**: 7.1.0 | **pytest**: 9.0.2  
**Test eseguiti**: 508 passed, 0 failed  
**Coverage totale**: 56% (717/1271 statements)

---

## Tabella per modulo

| Modulo | Stmts | Miss | Cover % | Stato |
|--------|-------|------|---------|-------|
| `common/__init__.py` | 6 | 0 | **100%** | OK |
| `common/paths.py` | 7 | 0 | **100%** | OK |
| `common/queue_lib.py` | 33 | 1 | **97%** | OK |
| `common/dispatcher_base.py` | 17 | 1 | **94%** | OK |
| `common/state_lib.py` | 101 | 9 | **91%** | OK |
| `common/hb_lib.py` | 25 | 7 | 72% | OK |
| `common/broadcast.py` | 22 | 1 | **95%** | OK |
| `common/config_validator.py` | 20 | 20 | **0%** | LOW |
| `connectivity/__init__.py` | 3 | 0 | **100%** | OK |
| `connectivity/fc_dispatcher.py` | 40 | 7 | 82% | OK |
| `connectivity/run_fc_on_epochs.py` | 45 | 35 | **22%** | LOW |
| `dashboard/__init__.py` | 2 | 0 | **100%** | OK |
| `dashboard/plot_top_features.py` | 119 | 52 | **56%** | MED |
| `features/__init__.py` | 2 | 0 | **100%** | OK |
| `features/dispatcher.py` | 68 | 3 | **96%** | OK |
| `ml_training/__init__.py` | 3 | 0 | **100%** | OK |
| `ml_training/ml_dispatcher.py` | 52 | 0 | **100%** | OK |
| `ml_training/permutation.py` | 35 | 0 | **100%** | OK |
| `parcellation/__init__.py` | 3 | 0 | **100%** | OK |
| `parcellation/extract_label_tc.py` | 54 | 23 | **57%** | MED |
| `parcellation/neuromaps_helper.py` | 57 | 12 | **79%** | OK |
| `pipeline_mne_bids/__init__.py` | 4 | 0 | **100%** | OK |
| `pipeline_mne_bids/bench/__init__.py` | 2 | 0 | **100%** | OK |
| `pipeline_mne_bids/bench/cli.py` | 17 | 17 | **0%** | LOW |
| `pipeline_mne_bids/bench/matrix.py` | 115 | 89 | **23%** | LOW |
| `pipeline_mne_bids/bench/reporter.py` | 31 | 24 | **23%** | LOW |
| `pipeline_mne_bids/run_bench_matrix.py` | 3 | 3 | **0%** | LOW |
| `pipeline_mne_bids/run_e2e/__init__.py` | 2 | 0 | **100%** | OK |
| `pipeline_mne_bids/run_e2e/cli.py` | 15 | 15 | **0%** | LOW |
| `pipeline_mne_bids/run_e2e/exec.py` | 70 | 50 | **29%** | LOW |
| `pipeline_mne_bids/run_e2e/report.py` | 10 | 0 | **100%** | OK |
| `pipeline_mne_bids/run_e2e_matchingpennies.py` | 3 | 3 | **0%** | LOW |
| `pipeline_mne_bids/run_with_timeout.py` | 12 | 7 | **42%** | MED |
| `source_reconstruction/__init__.py` | 3 | 0 | **100%** | OK |
| `source_reconstruction/apply_inverse_epochs_rs.py` | 41 | 19 | **54%** | MED |
| `source_reconstruction/finalize_inverse.py` | 44 | 35 | **20%** | LOW |
| **TOTAL** | **1271** | **554** | **56%** | — |

---

## Top low-coverage (< 70%)

File con coverage inferiore al 70%, ordinati per impatto (stmts × miss rate):

| Priorità | Modulo | Cover % | Miss | Causa principale |
|----------|--------|---------|------|-----------------|
| HIGH | `pipeline_mne_bids/bench/matrix.py` | 23% | 89/115 | Richiede dataset reale (epochs + inv); test sintetici non coprono il loop benchmark |
| HIGH | `source_reconstruction/finalize_inverse.py` | 20% | 35/44 | Dipende da MNE forward + evoked file; no fixture sintetica |
| HIGH | `pipeline_mne_bids/run_e2e/exec.py` | 29% | 50/70 | Loop inversione per-epoch non mockato nei test |
| MED | `connectivity/run_fc_on_epochs.py` | 22% | 35/45 | Stessa causa: richiede epochs + inv reali |
| MED | `dashboard/plot_top_features.py` | 56% | 52/119 | Funzioni di visualizzazione PNG senza display headless nei test |
| MED | `parcellation/extract_label_tc.py` | 57% | 23/54 | Path `fwd_path`/`src_path` non tutti coperti; CLI block non testato |
| LOW | `pipeline_mne_bids/bench/reporter.py` | 23% | 24/31 | `write_summary` chiama `_marginal` che vuole runs non vuoti |
| LOW | `source_reconstruction/apply_inverse_epochs_rs.py` | 54% | 19/41 | `save_stcs` e `__main__` block non coperti |
| LOW | `common/config_validator.py` | 0% | 20/20 | Nessun test per validazione config; modulo nuovo |
| LOW | `pipeline_mne_bids/bench/cli.py` | 0% | 17/17 | Entry point argparse non testato |

---

## Moduli a copertura eccellente (≥ 90%)

| Modulo | Cover % |
|--------|---------|
| `ml_training/ml_dispatcher.py` | 100% |
| `ml_training/permutation.py` | 100% |
| `features/dispatcher.py` | 96% |
| `common/broadcast.py` | 95% |
| `common/dispatcher_base.py` | 94% |
| `common/state_lib.py` | 91% |
| `parcellation/neuromaps_helper.py` | 79% |

I moduli core ML (ml_dispatcher, permutation, features/dispatcher) sono completamente coperti dai test sintetici — nessuna dipendenza da dati reali.

---

## Raccomandazioni

### 1. Aumentare coverage su moduli MNE-dipendenti (HIGH)

I moduli con coverage bassa (`finalize_inverse`, `run_fc_on_epochs`, `bench/matrix`) dipendono da file MNE reali (epochs .fif, inv .fif). La strategia raccomandata è il **mocking leggero**:

```python
# Esempio: mock mne.read_epochs per test finalize_inverse
from unittest.mock import patch, MagicMock
with patch("mne.read_forward_solution", return_value=MagicMock()):
    finalize("05", "matchingpennies")
```

Priorità: `finalize_inverse.py` (funzione core, 20% coverage) → target 60%.

### 2. Test CLI con `CliRunner` o `subprocess`

`bench/cli.py` e `run_e2e/cli.py` (0%) possono essere testati con `subprocess.run` o `argparse` diretto:

```python
def test_cli_help():
    result = subprocess.run(
        ["python", "-m", "pipeline_mne_bids.bench.cli", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
```

### 3. Test `dashboard/plot_top_features.py` headless

Le funzioni di visualizzazione richiedono `matplotlib.use("Agg")` prima dell'import per girare senza display. Aggiungere `conftest.py` fixture:

```python
import matplotlib
matplotlib.use("Agg")
```

### 4. `common/config_validator.py` — test unitari immediati

Modulo senza dipendenze MNE: 0% coverage → facile portare a 90%+ con 5-6 test parametrizzati.

### 5. Target coverage per dataset finale

Quando il dataset scientifico finale (S-07) sarà disponibile, aggiungere un test di integrazione E2E (`tests/test_e2e_full.py`) con `--slow` marker che copra il percorso completo — atteso: +15% coverage totale.

---

## Riferimenti

- HTML report: `reports/coverage_html/index.html` (generare con `bash scripts/run_coverage.sh`)
- JSON raw: `reports/coverage.json`
- Script: `scripts/run_coverage.sh`
