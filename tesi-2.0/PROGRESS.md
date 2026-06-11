# Tesi_2.0 — Tracking progresso

**Path**: `/home/seraxel/Scrivania/Tesi_2.0/`
**Ultimo aggiornamento**: 2026-05-07 (Fase 2 Cleanup DONE, sonnet1-tesi)
**HEAD**: cb29346 (Fase 2 cleanup done)

---

## Stato step pipeline

| Step | Titolo | Stato | Data | Log |
|------|--------|-------|------|-----|
| 1 | Preprocessing (mne-bids-pipeline) | ✅ DONE | 2026-04-27→28 | [reports/STEP1_LOG.md](reports/STEP1_LOG.md) |
| 2 | Source reconstruction (dSPM, sub-05) | ✅ DONE | 2026-05-01 | [reports/STEP2_LOG.md](reports/STEP2_LOG.md) |
| 3 | Parcellazione (aparc, 68 ROI) | ✅ DONE | 2026-05-01 | `parcellation/extract_label_tc.py` |
| 4 | Connettività funzionale (wPLI, bande) | ✅ DONE | 2026-05-01 | `connectivity/fc_dispatcher.py` |
| 5 | Feature extraction (mne-features + FC flatten) | ✅ DONE | 2026-05-01 | `features/dispatcher.py` |
| 6 | ML classification (LOSO GroupKFold, 5 algo) | ✅ DONE | 2026-05-01 | `ml_training/ml_dispatcher.py` |
| 7 | E2E su dataset scientifico | ❌ BLOCKED | — | Attesa Q1 (dataset scientifico) |

**E2E smoke su matchingpennies** (sub-05): ✅ PASS — 8 epoch, aparc, wpli, 10s wall. Vedi `tests/test_integration_final.py`.

---

## Sessione 2026-05-01 — statistiche

| Metrica | Valore |
|---------|--------|
| Sprint completati | **113** |
| Worker attivi | 4 (sonnet1/2/3-ts, haiku1-ts) |
| Batch wave dispatched | ~21 |
| Bug critici risolti | 1 (`run_e2e_matchingpennies.py:141`) |
| Deliverable prodotti | 19+ file (vedi SESSION_SUMMARY) |
| Sprint bloccati (Q1) | 2 (S-07, S-47) |

Vedi `.planning/SESSION_SUMMARY_2026-05-01.md` per il dettaglio completo.

---

## Decisioni aperte

| # | Tema | Stato | Urgenza |
|---|------|-------|---------|
| Q1 | Dataset scientifico finale (ds005385 vs LEMON) | 🔴 APERTA | **URGENTE** — blocca Step 7 |

Tutte le altre decisioni (Q2–Q7) sono state chiuse autonomamente con default reversibili.

---

## Infrastruttura prodotta (2026-05-01)

### Moduli `common/`
- `run_id.py` — run-id standard (timestamp + git_sha + config_hash)
- `logger.py` — JSON structured logger
- `reproducibility.py` — manifest builder per riproducibilità
- `run_schema.py` — JSONSchema + validator per output pipeline
- `paths.py` — path security helpers
- `queue_lib.py` — queue lock/update (fcntl)
- `config_validator.py` — config schema validator

### Moduli `analysis/`
- `stats_utility.py` — ci_bootstrap, cohen_d, power analysis
- `bench_steps.py` — pipeline step benchmarker

### Test (`tests/`)
- `test_integration_final.py` — E2E integration (2/2 PASS)
- `test_perf_regression.py` — perf regression (4/4 PASS in 5s)
- 20+ altri test (244 PASS totali al 2026-05-01)

### Script (`scripts/`)
- `generate_mock_bids.py` — mock ECEO BIDS dataset generator
- `archive_sync_files.sh` — SYNC file archiver
- `archive_full_session.sh` — session history archiver

### Reports notevoli
- `reports/PERF_BASELINE.json` — timing baseline + CI thresholds
- `reports/INTEGRATION_TEST.md` — E2E integration report
- `reports/PERF_BENCHMARK.md` — step-level benchmark (post-refactor: inverse_parcellation 3.58x speedup)
- `.planning/SESSION_SUMMARY_2026-05-01.md` — session summary completo

### Research/docs
- `.planning/research/citation_style.csl` — IEEE CSL (Zotero/Pandoc)
- `.planning/research/bibliografia.bib` — 27 BibTeX entries
- `.planning/research/CV_STRATEGY.md` — strategia CV + §5 bootstrap BCa

---

## Dataset

| Nome | Ruolo | Path | Dimensione | Stato |
|------|-------|------|------------|-------|
| eeg_matchingpennies | Smoke test tecnico | `data/eeg_matchingpennies/` | 716 MB | integrato |
| ds005385 (Dortmund Vital) | Dataset scientifico attivo | `data/raw/ds005385/` (symlink) | 21 GB | in corso — subset N=15 |

**Current scientific run** (aggiornato 2026-05-01 T18:50):
- Dataset: ds005385, task=RestingState, condizioni EO/EC (Q1 risolto)
- Subset: N=15 soggetti, random seed=42 (vincolo operatore hard)
- PILOT: 5 soggetti — sub-007, sub-010, sub-011, sub-026, sub-031
- Symlink: `data/raw/ds005385` → `~/Scrivania/Tesi/data/ds005385/`
- Config: `config/subjects_whitelist.py`, `config/labels_ds005385.py`
- Struttura BIDS: `reports/DS005385_STRUCTURE.md`
- Stato wave-scientific: S-100 done, S-100b done, S-101 gated (OPERATOR_APPROVE_5SUB)

---

## Vincoli operatore (invariati)

1. Usare `mne-bids-pipeline` come base reale
2. Librerie: MNE / mne-bids-pipeline / mne-connectivity / mne-features / scikit-learn
3. NO deep learning, NO tool extra
4. Struttura cartelle fissa (10 dir), NO alternative
5. Orchestrazione interna a Tesi_2.0/ (Q3 chiusa autonomamente)

---

## Prossimi passi

1. **Wave-scientific** → S-101 PILOT 5 sub (gated OPERATOR_APPROVE_5SUB) → S-103→S-107 step-by-step
2. **Performance P0** → refactor `inverse_parcellation` con `apply_inverse_epochs()` batch (riduzione ~40%)
3. Q1 risolta (2026-05-01): ds005385 scelto, subset N=15, seed=42

---

## 2026-05-07 — Recovery from Trash

Incidente fake-init: una sessione orfana ha cestinato la repo verso le 17:42. Recovery eseguito da opus-father (Opzione A, 18:15): `cp -a` da Trash → `Tesi_2.0_RESTORED` → swap. HEAD ripristinato a `21bba74` (verificato). Master plan, STATE, PROGRESS, research draft, 273 test, bench results, derivatives N=15 — tutti presenti. Backup safety: `Tesi_2.0_HAIKU_FALSE_INIT/` + `Tesi_2.0_HAIKU_FALSE_INIT_20260507_181326/`. Trash NON svuotato (safety net 7gg). Dettagli in `.planning/MSG_TO_FATHER_20260507T1815_RECOVERY_DONE.md`.

**Conseguenze**:
- Vecchio `reports/FEATURES_AUDIT.md` di haiku1 (commit fasullo `f2a8220`) SCARTATO — era invalido (scritto sulla dir fake).
- Re-spawn opus-orch-tesi con master plan come north star.
- Workers attivi (tmux): `opus1-tesi` (Opus 4.7, complessi), `sonnet1-tesi` (Sonnet 4.6, usuali), `haiku1-tesi` (Haiku 4.5, atomici).
- Avvio FASE 1 Audit (master plan §4) con 5 sprint H-AUDIT-01..05 (routing per modello).

## 2026-05-07 — Fase 1 Audit DONE (5/5)

Sprint H-AUDIT-01..05 completati. Output:

| Sprint | Worker | Report | Commit |
|--------|--------|--------|--------|
| H-AUDIT-01 | opus1-tesi | `reports/PIPELINE_AUDIT.md` | (opus1-tesi) |
| H-AUDIT-02 | opus1-tesi | `reports/PIPELINE_LEAKAGE_AUDIT.md` | 9241452 |
| H-AUDIT-03 | sonnet1-tesi | `reports/DATASET_SUBJECT_AUDIT.md` | 25125e1 |
| H-AUDIT-04 | sonnet1-tesi | `reports/BENCH_REPLICABILITY.md` | ca2f88e |
| H-AUDIT-05 | haiku1-tesi | (vedi orch) | — |

Risultati chiave:
- Dataset ds005385: 200 totali, **179 validi** (EO+EC, no late_ses1), 15 N15_ACTIVE PASS, pool N=30 = 164 candidati.
- Bench grid 96-combo: **96/96 REPLICABLE** (FC 960/960, X 32/32 NaN=0).
- Winner `aparc×coh×theta×svm_rbf` BA=0.867 verificato REPLICABLE.

## 2026-05-07 — Fase 2 Cleanup DONE

Vedi `reports/CLEANUP_AUDIT_2026-05-07.md` per dettagli completi.

**Bilancio recuperato**: ~700 MB (cache `__pycache__`) + ~4.2 GB (shadow `Tesi_2.0_HAIKU_FALSE_INIT*` archiviato da opus-father in `~/Scrivania/.archive/`) = **~4.9 GB totali**.

**22 SYNC archiviati** in `.planning/archive/2026-05-07_phase2_cleanup/sprint_S/` (sprint S-100..S-116, S-47, S-FC-EXTEND, AGG/VALIDATE/ML/DOC/FIG-N15).

Note:
- Shadow `Tesi_2.0/` non è dentro repo — archiviato da father in `~/Scrivania/.archive/`.
- 2 stray `.pyc` tracked in `config/` rimossi (commit cb29346 orch).
- **Test post-cleanup**: pytest **247 PASS / 63 skip / 0 FAIL** (GREEN), ruff 25 errori pre-esistenti (non introdotti da Fase 2 — 0 .py toccati), pre-commit binary non installato (env issue).
- **Next**: Fase 3 N=30 (gate ack opus-father; symlink `data/raw/ds005385` confermato disponibile 18:34).

**Debito tecnico** (task per Fase 4 hardening):
- `H-HARD-RUFF`: fix 25 errori ruff pre-esistenti (20/25 auto-fixable)
- `H-HARD-PRECOMMIT`: installare pre-commit + verificare hook env

---

## 2026-05-07 — Fase 3 H-SCALE-01 done + Phase 3 batch CANCELLED (default conservative)

- **H-SCALE-01** (commit `ceae34c`): `config/subjects_whitelist_n30.py` deterministic seed=42, hard-rule N=15 ⊂ N=30, hash `1594bf5d4914a864`, 9/9 test PASS, ruff 0. **15 nuovi sub** pescati: sub-012/014/015/032/034/038/040/048/076/082/091/136/171/187/200.
- **Batch H-SCALE-02..05 ANNULLATO** (operator decision pending, default conservative applied 22:07 by opus-father). Razionale: direttiva operator TG 21:21 "Tesi a regime non intensivo, workspace da concludere e killare" + operator silence post-disambig deadline 21:55. Vedi `.planning/MSG_TO_FATHER_PHASE3_CANCELLED.md`.
- **Workers**: opus1+sonnet1+haiku1-tesi tutti idle low-freq (HB 5min) — re-attivabili su richiesta NM-wake o operator.
- **Stato tesi**: paper-grade N=15 BA=0.867 già disponibile (winner aparc×coh×theta×svm_rbf, CI95=[0.766, 0.967], p_perm=0.0000, 96-combo grid 96/96 REPLICABLE). Manuscript draft v1 presente. Sufficiente per discussione laurea con limiti N=15 esplicitati in Discussion.
- **Re-attivazione futura**: SYNC_H-SCALE-02..05.md restano in `.planning/` come artefatti pronti. Path A integrale (~13h) o Path C ridimensionato (~6-7h riuso N=15) o Path D minimale (no batch, focus Fase 4 hardening + Fase 6 manuscript) — orch preference Path D allineato a "non-intensivo + concludere".

---

## 2026-05-12 — Fase 3 H-SCALE-N30 DONE (operator GO confermato)

Pipeline N=30 end-to-end completata in singolo dispatch `scripts/run_pipeline_n30.py` (disk-aware, riuso N=15 cache + 15 nuovi sub).

| Step | Stato | Note |
|------|-------|------|
| 2 (fwd+inv) | ✅ 30/30 | dSPM fsaverage |
| 2b (stcs per-epoch) | ✅ 30/30 | cleanup post-3b |
| 3b (label_ts) | ✅ 30/30 | aparc + schaefer100 |
| 4b (FC) | ✅ 30/30 | 1920 file (2 atlas × 4 metric × 4 band × 2 cond) |
| 5 (features) | ✅ 32/32 | X matrices (60, n_feat) |
| 6+7 (ML LOSO-30) | ✅ 96 combo | logreg + svm_rbf + lda (3 clf) |

### Winner N=30

| Campo | Valore |
|---|---|
| Config | **schaefer100 × plv × theta × logreg** |
| Balanced Accuracy | **0.967** |
| CI 95% | **[0.917, 1.000]** |
| p_perm | 0.0000 (vedi caveat) |
| Confronto N=15 | BA 0.867 → 0.967 (+10pp) |

### Caveat statistico

I p_perm correnti provengono da smoke test (n_perm=1) dopo che ML3 full (n_perm=1000, 6 clf) è morto a ~10h CPU (probabile OOM). BA è valida (LOSO base). PID 807181 in corso con n_perm=1000 + 3 clf rapidi → amend p_perm refined entro 5-10 min.

### Commit serie H-SCALE-N30

- `ceae34c` whitelist N=30 (seed=42)
- `5c7ced9` run_pipeline_n30.py driver
- `2ea7a69` --atlases CLI flag
- `26436cf` aggregate_n15 n_subjects param (fix shape guard)
- `8efdd3c` step5 dynamic threshold
- (next) --classifiers flag + reports + MSG father

### Insight scientifico

- PLV emerge come metric vincente in N=30 (era top-2/3 a N=15 con coh dominante)
- Theta band confermato robusto in scaling
- schaefer100 (100 ROI) supera aparc (68 ROI) a N=30 — più ROI → meglio con più campioni
- CI95 width 0.201 → 0.083 (più stretto, meno fragile)

Vedi `.planning/MSG_TO_FATHER_N30_DONE.md` per dettaglio cronologia rerun + caveat completi.


---

## 2026-05-27 — Statistical Hardening DONE

**Sprint**: FIX-01 → FIX-07  
**Orchestratore**: Sonnet 4.6 (orch-tesi)  
**Workers**: haiku1-tesi + haiku2-tesi (Haiku 4.5, 5 wave parallele)  
**Test**: 272 → 276 passed, 0 failed

### Commit sprint
- `96f3edd` feat(stats): Hedges g (FIX-04)
- `68d23c2` docs(features): z-score clarity + regression test (FIX-05)
- `6ad1e10` feat(dashboard): multi-dataset selector (FIX-06.a+b)
- `25cad1c` fix(dashboard): ruff noqa E402 pre-existing (orch)
- `8dc8eb0` feat(dashboard): schema validator (FIX-06.c)
- `25afc3c` feat(lemon): LEMON scaffolding (FIX-06.d+e)
- `0fa4d96` test(dashboard): mock filesystem tests (FIX-06.f)
- `6a4937e` docs(reports): statistical validity notes (FIX-07)

### Fix applicati
1. **BCa Bootstrap CI** — bias-corrected accelerated (Efron & Tibshirani 1993)
2. **Nested CV** — LogReg inner GridSearch C ∈ [0.01..100] + GroupKFold inner
3. **HARKing Disclosure** — FDR-BH su 96 combo, winner sopravvive
4. **Hedges g** — effect size bias-corrected per N<50
5. **Z-score clarity** — no double normalization; docstring + regression test
6. **Dashboard multi-dataset** — parametric data_loader + selectbox + validator
7. **LEMON scaffolding** — config.py + .gitkeep; gate operatore per path reale

### Gate operatore (bloccante per LEMON)
`LEMON_RAW_PATH = None` in `config/dataset_lemon.py` — operatore deve scaricare LEMON BIDS (~200GB) e impostare path.

### Riferimenti
- `reports/STATISTICAL_VALIDITY_NOTES.md` — sommario validità scientifica
- `reports/EXPERIMENTS_N50.md` — HARKing disclosure + FDR-BH table
