# API Documentation Guide — Tesi_2.0

**Data**: 2026-05-01  
**Strumento**: pdoc 16.0.0  
**Sprint**: S-49  
**Entry point**: `docs/api/index.html`

---

## Generazione rapida

```bash
make docs
```

Equivalente a:

```bash
bash scripts/build_docs.sh
```

Output in `docs/api/`. Aprire `docs/api/index.html` in qualsiasi browser.

---

## Moduli documentati

| Modulo | File HTML | Funzioni pubbliche chiave |
|--------|-----------|--------------------------|
| `parcellation` | `parcellation.html` | `get_labels`, `extract_tc`, `extract_tc_from_files` |
| `connectivity` | `connectivity.html` | `compute_fc`, `save_fc`, `epochs_to_label_tc` |
| `features` | `features.html` | `build_X`, `extract_univariate`, `flatten_fc` |
| `ml_training` | `ml_training.html` | `run_cv`, `run_all_algorithms`, `permutation_test`, `fdr_correction` |
| `source_reconstruction` | `source_reconstruction.html` | `finalize`, `apply_inverse_epochs_rs`, `save_stcs` |
| `common` | `common.html` | `write_heartbeat`, `read_all_heartbeats`, `is_stale`, `JsonLogger`, `validate_dispatch_key` |
| `analysis` | `analysis.html` | `bootstrap_ci`, `cohen_d`, `statistical_power`, `load_bench`, `apply_fdr` |
| `pipeline_mne_bids` | `pipeline_mne_bids.html` | `run_e2e`, `run_bench` |
| `dashboard` | `dashboard.html` | `plot_top_edges`, `plot_top_roi` |

---

## Prerequisiti

```bash
pip install pdoc       # una-tantum
```

pdoc non richiede configurazione aggiuntiva: legge i docstring numpy-style
già presenti in tutti i moduli e genera HTML statico navigabile.

---

## Rigenerazione automatica

Per rigenerare la documentazione dopo modifiche al codice:

```bash
make docs              # equivalente a bash scripts/build_docs.sh
```

Il comando sovrascrive `docs/api/` con la versione aggiornata.

Per una preview live durante lo sviluppo (ricarica automatica nel browser):

```bash
PYTHONPATH=. pdoc parcellation connectivity features ml_training \
    source_reconstruction common analysis pipeline_mne_bids dashboard
```

Il server pdoc si avvia sulla porta 8080 di default (`http://localhost:8080`).

---

## Cross-riferimenti

| Documento | Contenuto correlato |
|-----------|---------------------|
| `docs/PIPELINE_OVERVIEW.md` | Diagramma architetturale step-by-step |
| `.planning/research/METHODS_v1.md` | Descrizione scientifica dei moduli |
| `reports/COVERAGE.md` | Coverage test per modulo (S-34) |
| `.pre-commit-config.yaml` | ruff lint automatico (garantisce docstring format) |

---

## Note

I docstring seguono lo stile **numpy** (sezioni `Parameters`, `Returns`, `Notes`),
compatibile con pdoc e sphinx-napoleon. La documentazione viene aggiornata
automaticamente ad ogni rigenerazione — nessun file di configurazione da mantenere.

Per future migrazioni a Sphinx (se richiesto per la pubblicazione della tesi):

```bash
pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme
sphinx-quickstart docs/sphinx/
# Aggiungere estensioni: sphinx.ext.autodoc, sphinx.ext.napoleon
```
