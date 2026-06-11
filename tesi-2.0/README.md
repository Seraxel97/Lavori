# Tesi_2.0 — EEG Source-Space FC Pipeline

![tests](https://img.shields.io/badge/tests-273%20passed-brightgreen)
![coverage](https://img.shields.io/badge/coverage-56%25-yellow)
![lint](https://img.shields.io/badge/lint-ruff-blue)
![docs](https://img.shields.io/badge/docs-pdoc-blue)
![python](https://img.shields.io/badge/python-3.13-blue)
![mne](https://img.shields.io/badge/mne-1.11.0-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

Estensione modulare di `mne-bids-pipeline` con parcellazione atlastica, connettivita funzionale
e classificazione ML. Progetto di tesi magistrale.

Stato corrente: vedi [PROGRESS.md](PROGRESS.md).

---

## Overview

Pipeline in 7 step sequenziali che parte da dati EEG BIDS e produce classificatori ML
con valutazione LOSO (Leave-One-Subject-Out) e permutation testing:

```
STEP 1  → preprocessing (mne-bids-pipeline)
STEP 2  → source reconstruction (dSPM, fsaverage)
STEP 3  → parcellazione (aparc / destrieux / schaefer100 / schaefer200)
STEP 4  → connettivita spettrale (wpli, plv, coh, imcoh, ciplv, pli, wpli2_debiased)
STEP 5  → feature extraction (mne-features + FC flatten)
STEP 6  → ML classification (LogReg / SVM / MLP / RF / GB, GroupKFold)
STEP 7  → verifica E2E + risultati finali
```

Diagramma dettagliato: [docs/PIPELINE_OVERVIEW.md](docs/PIPELINE_OVERVIEW.md).
Diagramma SVG: [docs/figures/architecture.svg](docs/figures/architecture.svg).

---

## Current dataset

**Dataset scientifico**: ds005385 (Dortmund Vital Study), 200 soggetti totali, task=RestingState, conditions=EO/EC.

| Parametro | Valore |
|-----------|--------|
| Dataset | ds005385 (Dortmund Vital Study) |
| Soggetti totali | 200 |
| Subset attivo | N=15 (vincolo operatore hard), random seed=42 |
| PILOT | 5 sub: sub-007, sub-010, sub-011, sub-026, sub-031 |
| Task | RestingState |
| Condizioni | EO (Eyes Open), EC (Eyes Closed) |
| Path symlink | `data/raw/ds005385` → `~/Scrivania/Tesi/data/ds005385/` |
| Dimensione | 21 GB |

Cross-reference:
- Whitelist soggetti: `config/subjects_whitelist.py`
- Label mapping EO/EC: `config/labels_ds005385.py`
- Struttura BIDS analizzata: `reports/DS005385_STRUCTURE.md`

Dataset smoke tecnico: `data/eeg_matchingpennies/` (7 soggetti, 61 canali) — usato per test pipeline.

---

## Stato

| Step | Titolo | Stato | Output principale | Sprint |
|------|--------|-------|-------------------|--------|
| 1 | Preprocessing | DONE 2026-04-28 | `data/derivatives/.../sub-05/eeg/*_clean_epo.fif` | baseline |
| 2 | Source reconstruction | DONE 2026-05-01 | `*-inv.fif`, `*-stc-*-dSPM-{lh,rh}.stc` | baseline |
| 3 | Parcellazione | DONE 2026-05-01 | `parcellation/extract_label_tc.py` (4 atlanti) | S-01 |
| 4 | FC dispatcher | DONE 2026-05-01 | `connectivity/fc_dispatcher.py` (7 metriche) | S-02 |
| 5 | Feature extraction | DONE 2026-05-01 | `features/dispatcher.py` | S-03 |
| 6 | ML classification | DONE 2026-05-01 | `ml_training/ml_dispatcher.py` (5 algo, LOSO) | S-04 |
| 7 | E2E smoke | DONE 2026-05-01 | `pipeline_mne_bids/run_e2e_matchingpennies.py` | S-05 |
| 8 | Benchmark matrix | PENDING | `reports/BENCH_MATRIX_RESULTS.json` (700 run) | S-08 |
| — | Vault integration | DONE 2026-05-01 | `070_THESIS/RESULTS_BASELINE.md` (Obsidian) | S-12 |
| — | CI workflows | DONE 2026-05-01 | `.github/workflows/{lint,test,eeg-smoke}.yml` | DR-VAULT |

Log dettagliato: [PROGRESS.md](PROGRESS.md).

---

## Developer setup

### Pre-commit hooks

Il repository usa [pre-commit](https://pre-commit.com/) per garantire qualità del codice ad ogni commit:

```bash
pip install pre-commit
pre-commit install          # installa hook pre-commit (ruff check + format)
pre-commit install --hook-type pre-push   # installa hook pre-push (pytest smoke)
```

I hook configurati in `.pre-commit-config.yaml`:

| Hook | Trigger | Azione |
|------|---------|--------|
| `ruff` | pre-commit | Linting + autofix (`--fix`) |
| `ruff-format` | pre-commit | Formattazione stile |
| `pytest-fast` | pre-push | Smoke test `tests/test_smoke.py` |

Esecuzione manuale su tutti i file:

```bash
pre-commit run --all-files
```

### Installazione

#### Prerequisiti

- conda env `base`, Python 3.13.12
- Nessun pacchetto aggiuntivo da installare: tutto e' gia' presente nell'env.

Verifica:

```bash
python -c "import mne, mne_bids, mne_connectivity, mne_features, sklearn; print('ok')"
```

Versioni attese:

| Pacchetto | Versione |
|-----------|----------|
| mne | 1.11.0 |
| mne-bids | 0.18.0 |
| mne-bids-pipeline | 1.10.1 |
| mne-connectivity | 0.8 |
| mne-features | 0.3.2 |
| scikit-learn | 1.8.0 |
| neuromaps | 0.0.5 |

---

## Quick start

I comandi piu' comuni sono disponibili tramite `make`. Eseguire `make help` per la lista completa.

```bash
make test        # pytest tests/ -v  (273 test)
make lint        # ruff check + format --check
make format      # ruff autofix + format
make run-step1   # preprocessing matchingpennies sub-05
make run-step2   # source reconstruction + finalize inverse
make bench       # benchmark matrix 700 run (stima ~2-3h con ottimizzazioni)
make clean       # rimuove _cache, __pycache__, .pytest_cache, .ruff_cache
```

### STEP 1 — Preprocessing

```bash
make run-step1
# oppure direttamente:
mne_bids_pipeline --config config/config_step1_matchingpennies.py --steps preprocessing
```

Output: `data/derivatives/mne-bids-pipeline/sub-05/eeg/*_clean_epo.fif`

### STEP 2 — Source reconstruction

```bash
make run-step2
# oppure direttamente:
mne_bids_pipeline --config config/config_step2_source_matchingpennies.py \
    --steps preprocessing,sensor,source
python source_reconstruction/finalize_inverse.py \
    --subject 05 --task matchingpennies \
    --method dSPM --loose 0.2 --depth 0.8 --lambda2 0.111
```

Output: `data/derivatives/.../sub-05/eeg/*-stc-*-dSPM-{lh,rh}.stc`

### Benchmark matrix (STEP 8)

```bash
python pipeline_mne_bids/run_bench_matrix.py \
    --subject sub-05 \
    --deriv data/derivatives/mne-bids-pipeline \
    --sfreq-target 500 \
    --n-epochs-max 0
```

Output: `reports/BENCH_MATRIX_RESULTS.json` + `reports/BENCH_MATRIX_SUMMARY.md`

### E2E smoke completo

```bash
python pipeline_mne_bids/run_e2e_matchingpennies.py
```

### Test suite

```bash
pytest tests/ -v                          # suite completa
pytest tests/test_e2e_smoke_minimal.py    # smoke sintetico CI-safe (no dataset)
```

Diagramma completo con input/output per step: [docs/PIPELINE_OVERVIEW.md](docs/PIPELINE_OVERVIEW.md).

---

## Struttura cartelle

```
Tesi_2.0/
|-- config/                 configurazioni per ogni step (override mne-bids-pipeline)
|-- connectivity/           fc_dispatcher.py: 7 metriche spettrali via mne-connectivity
|-- dashboard/              plot_top_features.py: visualizzazioni figure paper (PNG headless)
|-- data/                   dataset BIDS + derivatives (NON versionato, .gitignore)
|   |-- eeg_matchingpennies/  dataset test tecnico (716 MB)
|   `-- derivatives/          output pipeline (epochs, STC, report HTML)
|-- docs/                   documentazione developer-facing
|-- features/               dispatcher.py: mne-features + FC flatten -> vettore X
|-- ml_training/            ml_dispatcher.py: 5 algoritmi + GroupKFold + permutation
|-- parcellation/           extract_label_tc.py: 4 atlasi + neuromaps_helper.py
|-- pipeline_mne_bids/      run_e2e_matchingpennies.py: orchestratore step 1-6
|-- reports/                log esecuzione, risultati benchmark, figure
|-- source_reconstruction/  finalize_inverse.py: forward + inverse + STC
`-- tests/                  pytest suite (test unitari + smoke)
```

---

## Riferimenti

### Documentazione progetto

- Pipeline overview: [docs/PIPELINE_OVERVIEW.md](docs/PIPELINE_OVERVIEW.md)
- Metodi step-by-step: [reports/STEP2_PROPOSAL.md](reports/STEP2_PROPOSAL.md), [reports/STEP3_PROPOSAL.md](reports/STEP3_PROPOSAL.md)
- Benchmark design: [.planning/BENCH_DESIGN.md](.planning/BENCH_DESIGN.md)
- Risultati baseline: `Obsidian: [[070_THESIS/RESULTS_BASELINE]]`

### Vault Obsidian (link professorali)

`[[070_THESIS/MNE_PROFESSOR_LINKS]]` — 5 link canonici del professore su MNE,
source reconstruction e connettivita funzionale.

### Librerie

| Libreria | Docs |
|----------|------|
| MNE-Python 1.11.0 | https://mne.tools/stable/ |
| mne-bids-pipeline 1.10.1 | https://mne-bids-pipeline.readthedocs.io/ |
| mne-connectivity 0.8 | https://mne.tools/mne-connectivity/ |
| mne-features 0.3.2 | https://mne.tools/mne-features/ |
| neuromaps 0.0.5 | https://netneurolab.github.io/neuromaps/ |
| scikit-learn 1.8.0 | https://scikit-learn.org/stable/ |

Progetto sviluppato come **estensione modulare** di mne-bids-pipeline, non come pipeline standalone.
Ogni step usa esclusivamente le librerie del vincolo 2 (PROGRESS.md §Vincoli OBBLIGATORI).

---

## Keywords

eeg, source-reconstruction, functional-connectivity, machine-learning, mne-python, mne-bids-pipeline, classification, eyes-closed, eyes-open, wpli, plv, coherence, group-kfold, permutation-testing, neuromaps, schaefer-atlas, freesurfer, dspm, parcellation
