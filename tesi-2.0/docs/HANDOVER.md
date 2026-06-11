# HANDOVER — Tesi_2.0

**Aggiornato**: 2026-05-01 T18:40 (sprint DR-FINAL-handover-docs, sonnet3-ts)
**Da leggere all'inizio di ogni nuova sessione** prima di qualsiasi altra azione.

---

## 1. Stato attuale (snapshot 2026-05-01 T18:40)

| Aspetto | Stato |
|---------|-------|
| Pipeline step 1-6 | Completi — smoke test matchingpennies PASS |
| Step 7 su dataset scientifico | Wave-scientific in corso (vedi sezione 2) |
| Dataset scientifico scelto | ds005385 (200 sub, 21 GB, EC/EO) — Q1 risolto |
| Subset attivo | N=15 soggetti, random seed=42, default A |
| Test suite | 244 PASS, 1 errore pre-esistente (fixture missing) |
| Ruff | 0 errori su tutti i moduli pubblici |
| Queue | 121 done, 2 blocked legacy, wave-scientific in corso |

---

## 2. Wave Scientific — stato in corso (leggere per primo)

Q1 risolta dall'operatore il 2026-05-01 T18:25: dataset scelto = **ds005385**.
Symlink attivo: `data/raw/ds005385 -> ~/Scrivania/Tesi/data/ds005385/` (200 sub, 21 GB).

### Sprint attivi al momento del handover

| ID | Descrizione | Owner | Stato |
|----|-------------|-------|-------|
| S-100 | BIDS analysis + mappatura EO/EC su ds005385 | sonnet1-ts | in corso |
| S-100b | Subset policy: N=15, seed=42 | sonnet2-ts | post S-100 |
| S-100c | Data audit + cleanup raccomandazioni | haiku1-ts | in corso |
| S-101 | STEP 2 source recon PILOT 5 sub | da assegnare | gated |

### Gate per S-101

S-101 e' bloccato su `OPERATOR_APPROVE_5SUB`. L'orch scrivera' un MSG_TO_FATHER
con il piano (lista 5 soggetti, tempo stimato, delta disco) una volta che S-100b e'
done. L'operatore deve approvare prima del dispatch.

### Sprint successivi (non dispatchare senza gate esplicito)

| ID | Descrizione | Gate |
|----|-------------|------|
| S-103 | Parcellazione aparc + schaefer100 su subset | post S-101 |
| S-104 | FC wPLI + coh banda alpha | post S-103 |
| S-105 | Feature extraction su subset | post S-104 |
| S-106 | ML LogReg + RF + GroupKFold(10) + permutation | post S-105 |
| S-107 | Esperimenti atlas x metric x algo | post S-106 |

Nota: S-102 (full 200-sub) e' deprecato da vincolo operatore — non dispatchare.

### File attesi (non ancora prodotti)

- `reports/STEP2_DS005385_PILOT.md` — prodotto da S-101 dopo approvazione
- Output S-100: report BIDS + config EO/EC mapping
- Output S-100b: policy file con lista soggetti subset N=15

---

## 3. Prima cosa da fare in ogni nuova sessione

```bash
cd /home/seraxel/Scrivania/Tesi_2.0

# 1. Heartbeat boot
python3 -c "
import json; from pathlib import Path; from datetime import datetime, timezone
p = Path('.planning/heartbeats_tesi/<session-name>.json')
p.write_text(json.dumps({'worker': '<session-name>', 'status': 'working',
  'timestamp': datetime.now(timezone.utc).isoformat(), 'sprint_id': None,
  'milestone': 'boot', 'last_capture': 'session start'}) + '\n')
"

# 2. Verifica queue (sprint claimabili con dipendenze soddisfatte)
python3 -c "
import json
rows = [json.loads(l) for l in open('.planning/DISPATCH_QUEUE_tesi.jsonl') if l.strip()]
done = {r['id'] for r in rows if r.get('status')=='done'}
cl = [r for r in rows if r.get('status')=='queued' and not r.get('owner')
      and all(d in done for d in r.get('deps',[]))]
print(f'Claimable: {len(cl)}')
for r in cl: print(f'  {r[\"id\"]}: {r[\"sprint\"][:70]}')
"

# 3. Leggi SYNC se esiste (.planning/SYNC_<id>.md), poi claim
```

---

## 4. File chiave

| File | Scopo |
|------|-------|
| `PROGRESS.md` | Stato step pipeline, asset, vincoli operatore |
| `.planning/SESSION_SUMMARY_2026-05-01.md` | Riepilogo completo sessione precedente (113 sprint) |
| `.planning/DISPATCH_QUEUE_tesi.jsonl` | Queue master — fonte autorevole di stato |
| `.planning/heartbeats_tesi/` | Heartbeat per worker (sonnet1/2/3-ts, haiku1-ts) |
| `.planning/MSG_TO_FATHER_*.md` | Messaggi orch -> father (leggere per context wave) |
| `docs/PIPELINE_OVERVIEW.md` | Diagramma ASCII pipeline 7 step |
| `reports/PERF_BENCHMARK.md` | Bottleneck: inverse_parcellation 55% del totale |
| `reports/STEP2_DS005385_PILOT.md` | Atteso da S-101 — non ancora prodotto |

---

## 5. Decisioni aperte

Q1 (dataset scientifico) risolta: **ds005385** scelto T18:25. Nessuna decisione
critica aperta. Blockers residui:

- S-101 gated su `OPERATOR_APPROVE_5SUB` — richiede approvazione esplicita operatore
- S-07 e S-47 (bloccati legacy su Q1) — verificare in queue se sbloccati automaticamente

---

## 6. Come riprendere il lavoro tecnico

```bash
# Smoke test E2E (10s, verifica che la pipeline non sia rotta)
python3 -m pytest tests/test_integration_final.py -q

# Performance regression (5s)
python3 -m pytest tests/test_perf_regression.py -q

# Ruff check su tutti i moduli
python3 -m ruff check common/ connectivity/ features/ ml_training/ pipeline_mne_bids/ analysis/
```

---

## 7. Ottimizzazioni raccomandate (non bloccanti)

1. **P0 — Performance**: refactor `inverse_parcellation` in `source_reconstruction/`
   con `mne.minimum_norm.apply_inverse_epochs()` batch API. Riduzione stimata ~40%
   sul tempo totale di pipeline (bottleneck attuale: 5.49s / 9.86s su 20 epoch).
2. **P1 — ML**: aggiungere `n_jobs=-1` agli stimatori sklearn in `ml_dispatcher.py`.

---

## 8. Struttura repo

```
Tesi_2.0/
├── common/             # Utilities: run_id, logger, queue_lib, paths, reproducibility
├── connectivity/       # FC dispatcher (wpli, plv, coh, imcoh, ciplv)
├── features/           # Feature extraction (mne-features + FC flatten)
├── ml_training/        # ML dispatcher + permutation test LOSO
├── parcellation/       # Source parcellation (extract_label_tc, neuromaps_helper)
├── pipeline_mne_bids/  # E2E orchestratore step 1-7
├── source_reconstruction/ # Inverse operator (finalize_inverse.py)
├── analysis/           # Stats utility, bench_steps, compare_runs
├── scripts/            # Utility shell/python (mock BIDS, archive, sanity)
├── tests/              # 244+ test PASS
├── config/             # Config step1 + step2 + JSON schema
├── data/               # Dataset (eeg_matchingpennies + ds005385 symlink)
├── reports/            # Report prodotti + manifests + figures
└── docs/               # Pipeline overview, architecture SVG, handover
```

---

## 9. Cross-reference

- Architettura: `docs/PIPELINE_OVERVIEW.md`, `docs/figures/architecture.svg`
- Regole worker: `~/.claude/CLAUDE.md` (globali), `docs/HANDOVER_GUIDE.md` (protocollo)
- Metodologia: `.planning/research/METHODS_v1.md`, `.planning/research/CV_STRATEGY.md`
- BibTeX: `.planning/research/bibliografia.bib` (27 entries), `citation_style.csl` (IEEE)
- Vault Obsidian: `~/Documenti/ObsidianVault/070_THESIS/INDEX.md`
