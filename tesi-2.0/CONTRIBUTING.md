# Contributing — Tesi_2.0

Guida per i worker (sessioni Claude) che contribuiscono al progetto.

---

## Developer setup

### Environment

```bash
conda activate base   # Python 3.13.12
```

Nessun `pip install` necessario: tutti i pacchetti sono gia' presenti.
Verifica imports: `python -c "import mne, mne_connectivity, mne_features, sklearn"`.

### Eseguire i test

```bash
pytest tests/ -v
```

Configurazione pytest in `pyproject.toml` (`[tool.pytest.ini_options]`).

---

## Workflow sprint

1. Leggi `.planning/SYNC_<sprint-id>.md` per istruzioni dettagliate.
2. Aggiorna `.planning/DISPATCH_QUEUE_tesi.jsonl` — campo `owner` + `status=claimed`.
3. Heartbeat ogni 5-7 min in `.planning/heartbeats_tesi/<worker>.json`:
   ```json
   {"worker": "<id>", "status": "working", "timestamp": "...", "sprint_id": "<sprint-id>", "last_capture": "..."}
   ```
4. Al completamento: heartbeat `status=done`, aggiorna queue `status=done`, scrivi
   `.planning/MSG_TO_ORCH_<worker>_<ts>.md` con verdict e artefatti prodotti.
5. Auto-claim sprint successivo se deps soddisfatti, altrimenti `status=blocked` + MSG_TO_ORCH.

---

## Naming conventions

### Branch

```
step<N>/<scope>       step3/parcellation, step4/connectivity
infra/<scope>         infra/tests, infra/run-id
fix/<area>            fix/fc-symmetrize, fix/extract-fwd-detect
cleanup/<area>        cleanup/fc-deadcode, cleanup/sync-archive
tests/<area>          tests/fc-edge
docs/<area>           docs/polish, docs/api-gen
analysis/<scope>      analysis/perf-benchmark
quality/<scope>       quality/lint, quality/mock-validator
security/<scope>      security/mock-dataset, security/path-validation
manuscript/<scope>    manuscript/bibtex, manuscript/methods
```

### Commit

Atomici per task. Messaggio in italiano o inglese coerente con il repository.
Mai `--no-verify`. Mai `--force-push` senza conferma operatore.

---

## Vincoli librerie (vincolo 2, PROGRESS.md)

Usare ESCLUSIVAMENTE:

- `mne` (incluso `mne.extract_label_time_course`, `mne.minimum_norm`)
- `mne-bids` / `mne-bids-pipeline`
- `mne-connectivity` (`spectral_connectivity_epochs`)
- `mne-features`
- `scikit-learn`
- `neuromaps` (per fetch annotazioni atlastica)
- Librerie standard Python + `numpy`, `scipy`, `pandas` (gia' dipendenze dei precedenti)

NO deep learning. NO librerie non in questa lista senza approvazione operatore.

---

## Struttura moduli

| Modulo | Responsabilita' |
|--------|-----------------|
| `parcellation/extract_label_tc.py` | STEP 3: STC -> label time courses, 4 atlasi |
| `parcellation/neuromaps_helper.py` | STEP 3b: fetch annotazioni neuromaps -> ROI |
| `connectivity/fc_dispatcher.py` | STEP 4: label_tc -> FC matrix per banda |
| `features/dispatcher.py` | STEP 5: FC matrix -> feature vector X |
| `ml_training/ml_dispatcher.py` | STEP 6: X, y, groups -> modelli + LOSO + permutation |
| `pipeline_mne_bids/run_e2e_matchingpennies.py` | STEP 7: orchestratore end-to-end |
| `source_reconstruction/finalize_inverse.py` | STEP 2: forward + inverse + STC |

Config files: `config/config_step<N>_<dataset>.py` (override mne-bids-pipeline).

---

## Code style

- Ruff per linting (configurazione in `pyproject.toml [tool.ruff]`).
- Type hints obbligatori su funzioni pubbliche.
- Docstring numpy-style su funzioni pubbliche (target sprint S-06).
- Nessun commento ovvio; commenta solo il "perche'" non ovvio.
- Nessun emoji nei file.

---

## Broadcast log eventi orchestratore

Per tracciare eventi di sprint/worker in modo append-only e concurrency-safe,
usa `common.broadcast.emit()`:

```python
from common.broadcast import emit

emit('sprint_claimed', sprint_id='S-XX', owner='sonnet1-ts')
emit('sprint_done', sprint_id='S-XX', verdict='PASS')
emit('worker_idle', worker='sonnet1-ts', age_min=12)
```

Il log viene scritto in `.planning/broadcast_log.jsonl` con timestamp ISO8601 UTC
e `fcntl.flock` esclusivo (via `common.queue_lib.queue_lock`). Ogni riga e' JSONL valido.

Per leggere il log: `from common.broadcast import read_log; events = read_log()`.

---

## Worker heartbeat write

Per evitare corruzioni da scrittura concorrente sui file `.planning/heartbeats_tesi/<worker>.json`,
usa sempre `.planning/hb_lib.write_heartbeat()` che scrive atomicamente via `tempfile + os.replace`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / ".planning"))
from hb_lib import write_heartbeat, read_heartbeat

hb_path = Path(".planning/heartbeats_tesi/sonnet2-ts.json")
write_heartbeat(hb_path, {"worker": "sonnet2-ts", "status": "working", ...})
```

**NON** usare `open(hb_path, 'w') + json.dump()` raw: il reader potrebbe vedere un file parzialmente scritto
nel mezzo di un crash o di una scrittura concorrente.

---

## Worker queue access

Per evitare race condition di append concorrente su `.planning/DISPATCH_QUEUE_tesi.jsonl`, usa sempre le utility di `.planning/queue_lib.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / ".planning"))
from queue_lib import append_sprint, update_status

# Append sicuro (flock esclusivo)
append_sprint(queue_path, {"id": "S-XX", "status": "queued", ...})

# Update atomico (read-modify-write con lock)
update_status(queue_path, "S-XX", status="done", owner="sonnet2-ts")
```

**NON** usare `open(queue_path, 'a') + write()` raw: due worker concorrenti possono interleave le righe JSONL corrompendo il file.

---

## Escalation blocchi

Se un task e' bloccato (dati mancanti, decisione operatore, dep non soddisfatta):

1. Aggiorna heartbeat: `status=blocked`, campo `blocker` con descrizione.
2. Scrivi `.planning/MSG_TO_ORCH_<worker>_<ts>.md` con dettagli.
3. Aggiorna `.planning/QUESTIONS_FOR_OPERATOR.md` se serve decisione operatore.
4. NON procedere con workaround non autorizzati.
