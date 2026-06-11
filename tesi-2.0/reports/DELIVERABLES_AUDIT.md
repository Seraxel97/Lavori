# Deliverables Audit Report

**Timestamp**: 2026-05-01T12:39:10.897989
**Total completed**: 103 sprints
**Deliverables present**: 60
**Deliverables missing**: 135

**Verdict**: MISSING FILES ⚠

## Sprint Breakdown

| Sprint ID | Status | Deliverables |
|-----------|--------|--------------|
| S-01 | MISSING | ✗ extract_label_tc.py (review-applied fixes), ✗ neuromaps_helper.py (opzionale, fetch mappe per ROI), ✗ test_parcellation.py (4 atlasi su STC sub-05) |
| S-02 | MISSING | ✗ fc_dispatcher.py (review-applied fixes + edge cases), ✗ run_fc_on_epochs.py (driver: epochs+inv+atlas → FC matrices), ✗ test_fc_dispatcher.py (7 metriche × 2 bande sintetici) |
| S-03 | MISSING | ✓ __init__.py, ✗ dispatcher.py (univariate + FC-flatten, output X matrix), ✓ test_features.py |
| S-04 | MISSING | ✓ __init__.py, ✗ ml_dispatcher.py (5 algoritmi, GroupKFold subject-level), ✗ permutation.py (1000 permute, p-value corretto FDR), ✓ test_ml_dispatcher.py |
| S-05 | MISSING | ✗ run_e2e_matchingpennies.py (orchestratore step 1-6), ✗ E2E_MATCHINGPENNIES.md (run log + risultati), ✓ test_e2e_matchingpennies.py |
| S-06 | MISSING | ✗ pyproject.toml (config ruff), ✗ docstring numpy-style su tutti i moduli pubblici, ✓ LINT_SWEEP.md |
| S-08 | MISSING | ✓ run_bench_matrix.py, ✓ BENCH_MATRIX_RESULTS.json, ✗ BENCH_MATRIX_SUMMARY.md (top-10 config per BA) |
| S-09 | MISSING | ✓ apply_inverse_epochs_rs.py, ✗ test_rs_epochs.py (N=10 fake epochs sintetici) |
| S-10 | MISSING | ✗ neuromaps_helper.py (fetch_annotation, parcellate), ✓ test_neuromaps_helper.py, ✓ NEUROMAPS_INTEGRATION.md |
| S-11 | MISSING | ✓ plot_top_features.py, ✗  (PNG topoplot + glass brain) |
| S-12 | MISSING | ✗ RESULTS_BASELINE.md (vault), ✗ PIPELINE_OVERVIEW.md (vault), ✗ aggiornamento THESIS_MAP.md |
| S-13 | MISSING | ✗ METHODS_v1.md (paper-style, 5000 parole target), ✗ cross-ref a config files + reports |
| S-14 | MISSING | ✓ .pre-commit-config.yaml, ✗ pyproject.toml [tool.ruff] [tool.black], ✗ README.md (sezione developer setup) |
| S-15 | PASS | ✓ MAIN_v1.md, ✓ INTRODUCTION_v1.md, ✓ DISCUSSION_v1.md |
| S-16 | PASS | ✓ compare_runs.py, ✓ fdr_helper.py, ✓ test_compare.py |
| S-17 | PASS | ✓ PERF_PROFILE.md, ✓ profile_bench.py |
| S-00b | MISSING | ✗ __init__.py (vuoto), ✗ conftest.py (fixtures: stc_path, fwd_path, evoked_path puntano a sub-05 esistente), ✗ pyproject.toml minimal con [tool.ruff] [tool.pytest.ini_options], ✗ ) |
| DR-REVIEW-fc-symmetrize | MISSING | ✗ unit test asserts mat[i,j]==mat[j,i], ✗ fix logica symmetrize se necessario |
| DR-REVIEW-fc-deadcode | MISSING | ✗ rimuovi n_labels o usa per validation precoce |
| DR-REVIEW-extract-fwd-detection | MISSING | ✗ signature: extract_tc_from_files(stc_path, src_path=None, fwd_path=None, ...) |
| DR-REVIEW-paths-env | MISSING | ✗  TESI_DERIV con default current |
| DR-REVIEW-docstrings | MISSING | ✗ numpy-style docstrings + type hints completi |
| DR-REVIEW-finalize-cli | MISSING | ✗ argparse args + docs |
| DR-REVIEW-edge-cases | MISSING | ✗ test_fc_edge.py |
| S-18 | MISSING | ✗ pyproject.toml: [tool.pytest.ini_options] testpaths=["tests"], addopts="-v", ✗ __init__.py (empty), ✗ conftest.py: fixtures stc_path, fwd_path, evoked_path, sub_id pointing to sub-05 derivatives, ✗ test_smoke.py: 1 test che importa parcellation.extract_label_tc + connectivity.fc_dispatcher senza eccezioni |
| S-19 | MISSING | ✗ README.md proper: install, env, run STEP 1 esempio, struttura cartelle (10 dir), link a vault MNE_PROFESSOR_LINKS, ✗ CONTRIBUTING.md: developer setup, commit conventions, branch naming, ✗ PIPELINE_OVERVIEW.md: ASCII diagram pipeline 7-step, link a config files |
| S-20 | MISSING | ✗ requirements.txt: pip freeze filtrato (mne 1.11.0, mne-bids-pipeline 1.10.1, neuromaps 0.0.5, mne-connectivity 0.8, mne-features 0.3.2, sklearn 1.8.0, numpy, scipy, pandas), ✗ environment.yml: conda recipe minimale equivalente, ✗ .python-version: 3.13 |
| DR-SEC-queue-locking | MISSING | ✗ queue_lib.py con flock o atomic rename, ✗ aggiornamento documentation worker queue access |
| DR-SEC-heartbeat-atomic | MISSING | ✗ hb_lib.py con atomic write helper |
| DR-SEC-path-validation | MISSING | ✗ finalize_inverse.py + extract_label_tc.py |
| DR-SEC-pipeline-timeout | MISSING | ✗ wrapper subprocess con timeout 30 min default |
| S-21 | MISSING | ✗ STEP3_PROPOSAL.md (paper-style; sezioni: obiettivo, atlas comparison, mean_flip vs PCA mode, metric trade-off, rischi), ✗ MNE_PROFESSOR_LINKS]], ✗ output: 800-1200 words, italiano + tabelle inglese (stile STEP2_PROPOSAL.md) |
| S-22 | MISSING | ✗ REVIEW_S22_<ts>.md verdict PASS|fix-list, ✗ verifica: README.md non emoji, links validi, pyproject.toml sintassi, conftest.py fixtures coerenti con file path reali |
| S-23 | MISSING | ✗ BIDS_VALIDATION.md (output bids-validator se installato, altrimenti note), ✗ fallback: python check via mne_bids.print_dir_tree + read_raw_bids smoke |
| DR-SIMPL-dispatcher-base | MISSING | ✗ dispatcher_base.py con DispatchProtocol Literal generic + validate signature |
| DR-SIMPL-bench-split | MISSING | ✗ {matrix.py,cli.py,reporter.py} |
| DR-SIMPL-path-helpers | MISSING | ✗ BIDS_ROOT singleton |
| DR-SIMPL-config-validate | MISSING | ✗ step2 |
| BAND-5-9-T01 | MISSING | ✗ DEEP_REVIEW_STEP3_band5-9.md, ✗ review concettuale + numerica |
| BAND-5-9-T02 | MISSING | ✗ DEEP_REVIEW_FC_band5-9.md, ✗ test_fc_edge_extreme.py |
| BAND-5-9-T03 | MISSING | ✓ VAULT_PATTERNS_IMPORTABLE.md, ✗ Claude-OPs]] |
| DR-VAULT-ci-matrix | MISSING | ✗ test.yml ispirato a Vesta (deps su S-14 CI hooks done) |
| DR-VAULT-heartbeat-pattern | MISSING | ✗ hb_lib.py |
| DR-VAULT-precommit | MISSING | ✗ .pre-commit-config.yaml ispirato a Vesta CONTRIBUTING.md |
| BAND-5-9-T04 | MISSING | ✗ test_parcellation_edge.py (N edge cases), ✓ DEEP_REVIEW_PARC_band5-9_T04.md |
| BAND-5-9-T05 | MISSING | ✗ lint.yml, ✓ VAULT_IMPORT_CI_band5-9_T05.md |
| S-24 | MISSING | ✗ CV_STRATEGY.md (~800 parole, paper-style) |
| S-25 | MISSING | ✓ BASELINE_COMPARISON.md, ✗ tabelle: BA, F1, MCC vs chance |
| S-26 | MISSING | ✗ {fig01_pipeline,fig02_topo,fig03_fc,fig04_ml,fig05_results}.placeholder |
| S-27 | MISSING | ✗ README.md aggiornato con: status table, run examples 7-step, badges placeholder |
| S-28 | MISSING | ✗ verify_requirements.py: parse requirements.txt, check pip versions match installed, ✓ REQ_VERIFY.md |
| DR-VAULT-eeg-ci-import | MISSING | ✓ eeg-smoke.yml, ✗ test_e2e_smoke_minimal.py (single sub minimal pipeline) |
| BAND-5-9-T06 | MISSING | ✗ fix per WARN identificati in BAND-T02, ✓ FC_CONSOLIDATION.md |
| BAND-5-9-T07 | PASS | ✓ optimize_hot_path.py, ✓ HOT_PATH_OPT.md |
| S-29 | MISSING | ✓ generate_mock_bids.py, ✗  (50 fake sub BIDS struct), ✓ test_mock_pipeline.py |
| S-30 | MISSING | ✗ README.md keywords section, ✓ TAGS.md |
| S-31 | MISSING | ✗ , features |
| S-32 | MISSING | ✗ ABSTRACT_v1.md (250 parole) |
| S-33 | MISSING | ✗ logger.py JSON structured, ✗  dir per run-id logs |
| DR-REVIEW-tests-fc-consolidate | MISSING | ✗ test_fc_dispatcher.py merged sections, ✗ rimuovi 3 file ridondanti |
| DR-REVIEW-package-init-explicit | MISSING | ✗ __all__ in: parcellation, connectivity, features, ml_training, source_reconstruction, common, analysis, pipeline_mne_bids, dashboard |
| DR-REVIEW-hb-queue-relocate | MISSING | ✓ hb_lib.py, ✓ queue_lib.py, ✗ update import path nei moduli che li usano |
| DR-REVIEW-pipeline-modular | MISSING | ✗ {exec,report}.py + wrapper compat |
| DR-REVIEW-cwd-validation | MISSING | ✗ verify_workers_cwd.sh check tmux sessions, ✓ CWD_AUDIT.md |
| DR-SEC-band59-path-env-validate | MISSING | ✗ paths.py: add Path.resolve() + check is_under expected base, ✓ test_paths_security.py |
| DR-SEC-band59-mock-dataset-validate | MISSING | ✗ , refuse system paths |
| DR-SEC-band59-ci-secrets-audit | MISSING | ✓ CI_SECURITY_AUDIT.md, ✗ checkout@v4 → @<sha>) |
| DR-SEC-band59-sbom | MISSING | ✗ SBOM.json (cyclonedx format), ✗ PIP_AUDIT.md (pip-audit output, vulnerability check) |
| DR-VAULT-schema-migration | MISSING | ✓ 001_initial_queue_schema.py, ✗ 002_heartbeat_v2.py, ✗ state_lib.py wrapper migrate-on-load |
| DR-VAULT-makefile-tasks | MISSING | ✗ Makefile (test, lint, format, run-step1, run-step2, bench), ✗ tasks.py invoke style (alternativa) |
| DR-VAULT-audit-rotation | MISSING | ✗  con DEEP_REVIEW_*.md, GAP_REVIEW_*.md, DECISION_LOG_*.md |
| DR-VAULT-jsonschema-config | MISSING | ✗ config.schema.json (JSON schema per parametri), ✗ 2 al boot |
| DR-VAULT-broadcast-log | MISSING | ✗ done, worker status, decision) |
| DR-RESEARCH-fc-leakage-2024 | MISSING | ✗ REFS_FC_LEAKAGE.md (5+ ref BibTeX), ✗ aggiungi sezione "Spatial leakage" a METHODS_v1.md |
| DR-RESEARCH-loso-cv-eeg | MISSING | ✗ REFS_LOSO_CV.md (5+ ref BibTeX), ✗ aggiunta a CV_STRATEGY.md |
| DR-RESEARCH-parcellation-eeg | MISSING | ✓ REFS_PARCELLATION.md, ✗ aggiunta a STEP3_PROPOSAL.md |
| S-34 | MISSING | ✓ run_coverage.sh, ✗ COVERAGE.md (per-module coverage table) |
| S-35 | PASS | ✓ CONCLUSION_v1.md, ✓ FUTURE_WORK_v1.md |
| S-36 | PASS | ✓ APPENDIX_A_CONFIG.md |
| S-37 | MISSING | ✗ RESULTS_BASELINE]] aggiornato con link reports |
| S-38 | MISSING | ✗ GLOSSARY.md (50+ entries: PLV, wPLI, dSPM, BIDS, ICA, ecc) |
| S-39 | PASS | ✓ repo_sanity.sh, ✓ REPO_SANITY.md |
| DR-VAULT-broadcast-emit-hooks | MISSING | ✗ queue_lib.append_sprint emit broadcast, ✗ queue_lib.update_status emit broadcast |
| DR-RESEARCH-eeg-fc-pipelines-soa | MISSING | ✗ EEG_PIPELINES_COMPARISON.md (table 5 col: pipeline, scope, parcellation, FC metrics, ML, status), ✗ aggiunta sezione "Related work" a MAIN_v1.md con 3-5 paragrafi |
| S-40 | MISSING | ✗ setup_scientific_dataset.py CLI: --dataset {ds005385|LEMON} --action {symlink|verify|prep}, ✓ DATASET_SETUP.md |
| S-41 | MISSING | ✗ stats_utility.py (ci_bootstrap, cohen_d, statistical_power), ✓ test_stats_utility.py |
| S-42 | MISSING | ✗ reproducibility.py manifest builder, ✓ MANIFEST_TEMPLATE.json |
| DR-VAULT-test-coverage-import | MISSING | ✗ .coveragerc + pyproject.toml [tool.coverage] threshold, ✗ test.yml |
| DR-RESEARCH-permutation-significance | MISSING | ✗ REFS_PERMUTATION.md (5+ BibTeX), ✗ aggiunta sezione "Permutation testing rationale" a CV_STRATEGY.md |
| S-43 | MISSING | ✗ progress.py (tqdm wrapper con ETA), ✗ integration in run_e2e_matchingpennies.py |
| S-44 | MISSING | ✗ run_schema.py: schema definition, ✓ test_run_schema.py |
| S-45 | MISSING | ✗ main.py entry-point unico, ✗ {step1,step2,bench,e2e}.py subcommand |
| S-46 | MISSING | ✓ __init__.py, ✗ synthetic.py (synthetic_epochs, synthetic_label_tc, synthetic_fc, synthetic_X_y_groups) |
| DR-VAULT-makefile-completeness | MISSING | ✗ MAKEFILE_AUDIT.md (compare targets) |
| S-48 | MISSING | ✗ PERF_BENCHMARK.md (table tempi per step), ✓ bench_steps.py |
| S-49 | MISSING | ✗ conf.py (sphinx) o pdoc CLI script, ✓ API_DOCS_GUIDE.md |
| S-50 | PASS | ✓ PAPER_QC_CHECKLIST.md |
| DR-VAULT-bibtex-import | MISSING | ✗ bibliografia.bib (master file) |
| S-52 | PASS | ✓ validate_mock_bids.py, ✓ MOCK_VALIDATION.md |
| S-53 | PASS | ✓ run_id.py, ✓ test_run_id.py |
| DR-VAULT-skill-init | PASS | ✓ SKILL_PATTERN_EVAL.md |
| S-54 | MISSING | ✗ README.md aggiornato con badges shields.io |
| S-55 | MISSING | ✗ architecture.svg (mermaid o draw.io export), ✓ fig00_architecture.png |

## Missing Deliverables

The following sprints have missing deliverables:

### S-01

- **MISSING**: parcellation/extract_label_tc.py (review-applied fixes)
- **MISSING**: parcellation/neuromaps_helper.py (opzionale, fetch mappe per ROI)
- **MISSING**: tests/test_parcellation.py (4 atlasi su STC sub-05)

### S-02

- **MISSING**: connectivity/fc_dispatcher.py (review-applied fixes + edge cases)
- **MISSING**: connectivity/run_fc_on_epochs.py (driver: epochs+inv+atlas → FC matrices)
- **MISSING**: tests/test_fc_dispatcher.py (7 metriche × 2 bande sintetici)

### S-03

- **MISSING**: features/dispatcher.py (univariate + FC-flatten, output X matrix)

### S-04

- **MISSING**: ml_training/ml_dispatcher.py (5 algoritmi, GroupKFold subject-level)
- **MISSING**: ml_training/permutation.py (1000 permute, p-value corretto FDR)

### S-05

- **MISSING**: pipeline_mne_bids/run_e2e_matchingpennies.py (orchestratore step 1-6)
- **MISSING**: reports/E2E_MATCHINGPENNIES.md (run log + risultati)

### S-06

- **MISSING**: pyproject.toml (config ruff)
- **MISSING**: docstring numpy-style su tutti i moduli pubblici

### S-08

- **MISSING**: reports/BENCH_MATRIX_SUMMARY.md (top-10 config per BA)

### S-09

- **MISSING**: tests/test_rs_epochs.py (N=10 fake epochs sintetici)

### S-10

- **MISSING**: parcellation/neuromaps_helper.py (fetch_annotation, parcellate)

### S-11

- **MISSING**: reports/figures/ (PNG topoplot + glass brain)

### S-12

- **MISSING**: 070_THESIS/RESULTS_BASELINE.md (vault)
- **MISSING**: 050_METHODS/PIPELINE_OVERVIEW.md (vault)
- **MISSING**: aggiornamento THESIS_MAP.md

### S-13

- **MISSING**: .planning/research/METHODS_v1.md (paper-style, 5000 parole target)
- **MISSING**: cross-ref a config files + reports/

### S-14

- **MISSING**: pyproject.toml [tool.ruff] [tool.black]
- **MISSING**: README.md (sezione developer setup)

### S-00b

- **MISSING**: tests/__init__.py (vuoto)
- **MISSING**: tests/conftest.py (fixtures: stc_path, fwd_path, evoked_path puntano a sub-05 esistente)
- **MISSING**: pyproject.toml minimal con [tool.ruff] [tool.pytest.ini_options]
- **MISSING**: .gitignore (data/derivatives, __pycache__, *.pyc, .planning/heartbeats_tesi/, .planning/orchestrators/)

### DR-REVIEW-fc-symmetrize

- **MISSING**: unit test asserts mat[i,j]==mat[j,i]
- **MISSING**: fix logica symmetrize se necessario

### DR-REVIEW-fc-deadcode

- **MISSING**: rimuovi n_labels o usa per validation precoce

### DR-REVIEW-extract-fwd-detection

- **MISSING**: signature: extract_tc_from_files(stc_path, src_path=None, fwd_path=None, ...)

### DR-REVIEW-paths-env

- **MISSING**: env var TESI_SUBJECTS_DIR / TESI_DERIV con default current

### DR-REVIEW-docstrings

- **MISSING**: numpy-style docstrings + type hints completi

### DR-REVIEW-finalize-cli

- **MISSING**: argparse args + docs

### DR-REVIEW-edge-cases

- **MISSING**: tests/test_fc_edge.py

### S-18

- **MISSING**: pyproject.toml: [tool.pytest.ini_options] testpaths=["tests"], addopts="-v"
- **MISSING**: tests/__init__.py (empty)
- **MISSING**: tests/conftest.py: fixtures stc_path, fwd_path, evoked_path, sub_id pointing to sub-05 derivatives
- **MISSING**: tests/test_smoke.py: 1 test che importa parcellation.extract_label_tc + connectivity.fc_dispatcher senza eccezioni

### S-19

- **MISSING**: README.md proper: install, env, run STEP 1 esempio, struttura cartelle (10 dir), link a vault MNE_PROFESSOR_LINKS
- **MISSING**: CONTRIBUTING.md: developer setup, commit conventions, branch naming
- **MISSING**: docs/PIPELINE_OVERVIEW.md: ASCII diagram pipeline 7-step, link a config files

### S-20

- **MISSING**: requirements.txt: pip freeze filtrato (mne 1.11.0, mne-bids-pipeline 1.10.1, neuromaps 0.0.5, mne-connectivity 0.8, mne-features 0.3.2, sklearn 1.8.0, numpy, scipy, pandas)
- **MISSING**: environment.yml: conda recipe minimale equivalente
- **MISSING**: .python-version: 3.13

### DR-SEC-queue-locking

- **MISSING**: utility .planning/queue_lib.py con flock o atomic rename
- **MISSING**: aggiornamento documentation worker queue access

### DR-SEC-heartbeat-atomic

- **MISSING**: utility .planning/hb_lib.py con atomic write helper

### DR-SEC-path-validation

- **MISSING**: helper validate_path(p, must_be_under) in source_reconstruction/finalize_inverse.py + extract_label_tc.py

### DR-SEC-pipeline-timeout

- **MISSING**: wrapper subprocess con timeout 30 min default

### S-21

- **MISSING**: reports/STEP3_PROPOSAL.md (paper-style; sezioni: obiettivo, atlas comparison, mean_flip vs PCA mode, metric trade-off, rischi)
- **MISSING**: cross-ref a parcellation/extract_label_tc.py (post sonnet1-ts hardening) + vault [[070_THESIS/MNE_PROFESSOR_LINKS]]
- **MISSING**: output: 800-1200 words, italiano + tabelle inglese (stile STEP2_PROPOSAL.md)

### S-22

- **MISSING**: .planning/REVIEW_S22_<ts>.md verdict PASS|fix-list
- **MISSING**: verifica: README.md non emoji, links validi, pyproject.toml sintassi, conftest.py fixtures coerenti con file path reali

### S-23

- **MISSING**: reports/BIDS_VALIDATION.md (output bids-validator se installato, altrimenti note)
- **MISSING**: fallback: python check via mne_bids.print_dir_tree + read_raw_bids smoke

### DR-SIMPL-dispatcher-base

- **MISSING**: common/dispatcher_base.py con DispatchProtocol Literal generic + validate signature

### DR-SIMPL-bench-split

- **MISSING**: pipeline_mne_bids/bench/{matrix.py,cli.py,reporter.py}

### DR-SIMPL-path-helpers

- **MISSING**: common/paths.py con DERIV/SUBJECTS_DIR/BIDS_ROOT singleton

### DR-SIMPL-config-validate

- **MISSING**: config/config_base.py importato da config_step1/step2

### BAND-5-9-T01

- **MISSING**: reports/DEEP_REVIEW_STEP3_band5-9.md
- **MISSING**: review concettuale + numerica

### BAND-5-9-T02

- **MISSING**: reports/DEEP_REVIEW_FC_band5-9.md
- **MISSING**: tests/test_fc_edge_extreme.py

### BAND-5-9-T03

- **MISSING**: crossref a vault [[060_PROJECTS/Vesta]] + [[060_PROJECTS/Claude-OPs]]

### DR-VAULT-ci-matrix

- **MISSING**: .github/workflows/test.yml ispirato a Vesta (deps su S-14 CI hooks done)

### DR-VAULT-heartbeat-pattern

- **MISSING**: cross-ref claude-ops/.planning/hb_lib.py se esiste, import / clone in .planning/hb_lib.py

### DR-VAULT-precommit

- **MISSING**: .pre-commit-config.yaml ispirato a Vesta CONTRIBUTING.md

### BAND-5-9-T04

- **MISSING**: tests/test_parcellation_edge.py (N edge cases)

### BAND-5-9-T05

- **MISSING**: .github/workflows/test.yml + .github/workflows/lint.yml

### S-24

- **MISSING**: .planning/research/CV_STRATEGY.md (~800 parole, paper-style)

### S-25

- **MISSING**: tabelle: BA, F1, MCC vs chance

### S-26

- **MISSING**: reports/figures/{fig01_pipeline,fig02_topo,fig03_fc,fig04_ml,fig05_results}.placeholder

### S-27

- **MISSING**: README.md aggiornato con: status table, run examples 7-step, badges placeholder

### S-28

- **MISSING**: scripts/verify_requirements.py: parse requirements.txt, check pip versions match installed

### DR-VAULT-eeg-ci-import

- **MISSING**: tests/test_e2e_smoke_minimal.py (single sub minimal pipeline)

### BAND-5-9-T06

- **MISSING**: fix per WARN identificati in BAND-T02

### S-29

- **MISSING**: data/mock_eceo/ (50 fake sub BIDS struct)

### S-30

- **MISSING**: README.md keywords section

### S-31

- **MISSING**: typing complete su parcellation/, connectivity/, ml_training/, features/

### S-32

- **MISSING**: .planning/research/ABSTRACT_v1.md (250 parole)

### S-33

- **MISSING**: common/logger.py JSON structured
- **MISSING**: reports/runs/ dir per run-id logs

### DR-REVIEW-tests-fc-consolidate

- **MISSING**: tests/test_fc_dispatcher.py merged sections
- **MISSING**: rimuovi 3 file ridondanti

### DR-REVIEW-package-init-explicit

- **MISSING**: __all__ in: parcellation, connectivity, features, ml_training, source_reconstruction, common, analysis, pipeline_mne_bids, dashboard

### DR-REVIEW-hb-queue-relocate

- **MISSING**: update import path nei moduli che li usano

### DR-REVIEW-pipeline-modular

- **MISSING**: pipeline_mne_bids/run_e2e/{exec,report}.py + wrapper compat

### DR-REVIEW-cwd-validation

- **MISSING**: scripts/verify_workers_cwd.sh check tmux sessions

### DR-SEC-band59-path-env-validate

- **MISSING**: common/paths.py: add Path.resolve() + check is_under expected base

### DR-SEC-band59-mock-dataset-validate

- **MISSING**: scripts/generate_mock_bids.py: --output-dir required, must be under data/, refuse system paths

### DR-SEC-band59-ci-secrets-audit

- **MISSING**: pin actions/* a SHA invece di tag (es. actions/checkout@v4 → @<sha>)

### DR-SEC-band59-sbom

- **MISSING**: reports/SBOM.json (cyclonedx format)
- **MISSING**: reports/PIP_AUDIT.md (pip-audit output, vulnerability check)

### DR-VAULT-schema-migration

- **MISSING**: .planning/migrations/002_heartbeat_v2.py
- **MISSING**: common/state_lib.py wrapper migrate-on-load

### DR-VAULT-makefile-tasks

- **MISSING**: Makefile (test, lint, format, run-step1, run-step2, bench)
- **MISSING**: tasks.py invoke style (alternativa)

### DR-VAULT-audit-rotation

- **MISSING**: .planning/audits/2026-05-01/ con DEEP_REVIEW_*.md, GAP_REVIEW_*.md, DECISION_LOG_*.md

### DR-VAULT-jsonschema-config

- **MISSING**: config/config.schema.json (JSON schema per parametri)
- **MISSING**: tests/test_config_schema.py validate config_step1/2 al boot

### DR-VAULT-broadcast-log

- **MISSING**: .planning/broadcast_log.jsonl append-only log eventi orch (sprint claim/done, worker status, decision)

### DR-RESEARCH-fc-leakage-2024

- **MISSING**: .planning/research/REFS_FC_LEAKAGE.md (5+ ref BibTeX)
- **MISSING**: aggiungi sezione "Spatial leakage" a METHODS_v1.md

### DR-RESEARCH-loso-cv-eeg

- **MISSING**: .planning/research/REFS_LOSO_CV.md (5+ ref BibTeX)
- **MISSING**: aggiunta a CV_STRATEGY.md

### DR-RESEARCH-parcellation-eeg

- **MISSING**: aggiunta a STEP3_PROPOSAL.md

### S-34

- **MISSING**: reports/COVERAGE.md (per-module coverage table)

### S-37

- **MISSING**: vault [[070_THESIS/RESULTS_BASELINE]] aggiornato con link reports/

### S-38

- **MISSING**: .planning/research/GLOSSARY.md (50+ entries: PLV, wPLI, dSPM, BIDS, ICA, ecc)

### DR-VAULT-broadcast-emit-hooks

- **MISSING**: queue_lib.append_sprint emit broadcast
- **MISSING**: queue_lib.update_status emit broadcast

### DR-RESEARCH-eeg-fc-pipelines-soa

- **MISSING**: .planning/research/EEG_PIPELINES_COMPARISON.md (table 5 col: pipeline, scope, parcellation, FC metrics, ML, status)
- **MISSING**: aggiunta sezione "Related work" a MAIN_v1.md con 3-5 paragrafi

### S-40

- **MISSING**: scripts/setup_scientific_dataset.py CLI: --dataset {ds005385|LEMON} --action {symlink|verify|prep}

### S-41

- **MISSING**: analysis/stats_utility.py (ci_bootstrap, cohen_d, statistical_power)

### S-42

- **MISSING**: common/reproducibility.py manifest builder

### DR-VAULT-test-coverage-import

- **MISSING**: .coveragerc + pyproject.toml [tool.coverage] threshold
- **MISSING**: aggiunta a .github/workflows/test.yml

### DR-RESEARCH-permutation-significance

- **MISSING**: .planning/research/REFS_PERMUTATION.md (5+ BibTeX)
- **MISSING**: aggiunta sezione "Permutation testing rationale" a CV_STRATEGY.md

### S-43

- **MISSING**: common/progress.py (tqdm wrapper con ETA)
- **MISSING**: integration in run_e2e_matchingpennies.py

### S-44

- **MISSING**: common/run_schema.py: schema definition

### S-45

- **MISSING**: cli/main.py entry-point unico
- **MISSING**: cli/{step1,step2,bench,e2e}.py subcommand

### S-46

- **MISSING**: tests/fixtures/synthetic.py (synthetic_epochs, synthetic_label_tc, synthetic_fc, synthetic_X_y_groups)

### DR-VAULT-makefile-completeness

- **MISSING**: reports/MAKEFILE_AUDIT.md (compare targets)

### S-48

- **MISSING**: reports/PERF_BENCHMARK.md (table tempi per step)

### S-49

- **MISSING**: docs/conf.py (sphinx) o pdoc CLI script

### DR-VAULT-bibtex-import

- **MISSING**: .planning/research/bibliografia.bib (master file)

### S-54

- **MISSING**: README.md aggiornato con badges shields.io

### S-55

- **MISSING**: docs/figures/architecture.svg (mermaid o draw.io export)
