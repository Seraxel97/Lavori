# Heartbeat Pattern — Confronto Tesi_2.0 vs claude-ops

**Data**: 2026-05-01  
**Sprint**: DR-VAULT-heartbeat-pattern  
**Scope**: analisi del pattern heartbeat in `common/hb_lib.py` (Tesi_2.0) vs
implementazione in `~/Scrivania/claude-ops/` (sistema di orchestrazione principale).

---

## 1. Analisi claude-ops heartbeat pattern

Il sistema claude-ops implementa un pattern heartbeat significativamente più elaborato
rispetto a Tesi_2.0, articolato in più componenti:

| Componente | Path (claude-ops) | Funzione |
|-----------|-------------------|---------|
| `heartbeat_validator.py` | `scripts/heartbeat/` | Normalizza HB a schema v3; rinomina campi deprecated |
| `reconcile_hb.py` | `scripts/orchestrator/` | Marca task done nella queue se HB = done (BUG-2 fix) |
| `heartbeat_writer.sh` | `scripts/orchestrator/` | Shell wrapper per scrittura da bash |
| `mc_heartbeat_pusher.py` | `scripts/orchestrator/` | Push HB a monitoring channel (Telegram) |
| `tg_heartbeat_report.py` | `scripts/tg/` | Report periodico HB via Telegram |
| `backfill_heartbeats_v21.py` | `scripts/observability/` | Backfill storico HB per analisi gap |
| `hb_tg_watcher.py` | `ui/launcher_textual/watchers/` | Watcher live HB in UI Textual |

### Schema heartbeat claude-ops (v3 — da `heartbeat_validator.py`)

```json
{
    "session":          "sonnet2-ts",
    "state":            "in_progress",
    "role":             "worker",
    "last_heartbeat_ts": "2026-05-01T02:00:00Z",
    "sprint_id":        "S-01"
}
```

**Field mapping legacy → v3** (rinomina automatica):

| Campo vecchio | Campo v3 | Valore enum |
|--------------|----------|-------------|
| `worker` | `session` | qualsiasi stringa |
| `status` | `state` | `idle`, `in_progress`, `done`, `blocked`, `shutdown`, `paused` |
| `type` | `role` | `worker`, `orch`, `father`, `gap-review`, `gap-review-agent` |
| `ts` | `last_heartbeat_ts` | ISO-8601 string |

### Schema fixture claude-ops (formato reale da `tests/fixtures/heartbeats/`)

```json
{
    "ts": "2026-04-15T00:00:00Z",
    "sprint_id": "COMBO_V13_T6",
    "status": "done",
    "deliverables_done": ["combo_v13.py"],
    "worker": "sonnet2"
}
```

Le fixture usano ancora i campi legacy (`ts`, `status`, `worker`) — indicativo che il
migrazione a v3 è in corso ma non completata sulle fixture di test.

---

## 2. Schema heartbeat Tesi_2.0 (corrente)

Il pattern Tesi_2.0 è significativamente più semplice, progettato per un singolo progetto:

```json
{
    "worker":     "sonnet1-ts",
    "status":     "working",
    "timestamp":  "2026-05-01T03:00:00+02:00",
    "sprint_id":  "S-01",
    "milestone":  "tests done",
    "last_capture": "pytest 22/22 PASS",
    "next_action": "commit e claim S-02"
}
```

**API `common/hb_lib.py`** (post S-33 extension):

| Funzione | Firma | Note |
|---------|-------|------|
| `write_heartbeat(hb_path, data)` | `Path, dict → None` | Atomic tempfile+os.replace |
| `read_heartbeat(hb_path)` | `Path → dict \| None` | None se file mancante/corrotto |
| `read_all_heartbeats(hb_dir)` | `Path → list[dict]` | Scansiona dir, aggiunge `_source_file` |
| `is_stale(hb, threshold_min=10)` | `dict, float → bool` | Riconosce `ts` (float) e `timestamp` (ISO) |

---

## 3. Tabella confronto

| Dimensione | Tesi_2.0 `hb_lib` | claude-ops |
|-----------|-------------------|-----------|
| **Schema campi temporali** | `timestamp` (ISO string, locale) | `last_heartbeat_ts` (ISO UTC v3) o `ts` (ISO string, legacy) |
| **Campo worker/session** | `worker` | `session` (v3) / `worker` (legacy) |
| **Campo stato** | `status` (free text) | `state` (enum: idle/in_progress/done/blocked/shutdown/paused) |
| **Campo ruolo** | assente | `role` (enum: worker/orch/father/gap-review) |
| **Campi extra** | `sprint_id`, `milestone`, `last_capture`, `next_action` | `sprint_id`, `deliverables_done` |
| **Validazione schema** | nessuna (dict libero) | `heartbeat_validator.py` normalizza a v3 |
| **Staleness detection** | `is_stale()` con soglia configurabile | `reconcile_hb.py` + `hb_tg_watcher.py` |
| **Notifiche** | nessuna | Telegram via `mc_heartbeat_pusher.py` |
| **UI live** | nessuna | Watcher Textual (`hb_tg_watcher.py`) |
| **Storage** | File JSON in `.planning/heartbeats_tesi/` | File JSON multi-dir + Telegram channel |
| **Write strategy** | tempfile+os.replace (atomico) | `heartbeat_writer.sh` (non verificato atomico) |

---

## 4. Raccomandazione: divergenza motivata

### Convergenza non raccomandata (al momento)

Adottare lo schema v3 di claude-ops in Tesi_2.0 richiederebbe:
- Rinominare `worker` → `session` e `status` → `state` nei 20+ file di heartbeat esistenti
- Aggiungere il campo `role` (irrilevante per Tesi_2.0: tutti i worker hanno ruolo uniforme)
- Migrare `timestamp` → `last_heartbeat_ts` con formato UTC esplicito

Il costo di migrazione supera il beneficio per un singolo progetto di tesi senza
sistema di monitoring multi-repo.

### Convergenza parziale raccomandata

Due miglioramenti a basso costo che allineano i pattern senza breaking change:

1. **Timestamp UTC**: cambiare `timestamp` da `+02:00` (locale) a `.isoformat()` in
   formato UTC (es. `datetime.now(UTC).isoformat()`) per compatibilità con parser
   cross-timezone. Impatto: nessuna rottura, solo cambio di timezone nel valore.

2. **`status` come enum soft**: documentare i valori attesi di `status` come
   `working`, `blocked`, `done`, `idle` (allineati alla semantica claude-ops
   `in_progress`, `blocked`, `done`, `idle`) senza enforcing a runtime.

### Differenza tecnica chiave da preservare

La write strategy `tempfile+os.replace` di Tesi_2.0 è **superiore** all'approccio
`heartbeat_writer.sh` di claude-ops (non verificato atomico). Questa differenza
non va colmata verso claude-ops.

---

## 5. Action items

| Item | Priorità | Effort |
|------|----------|--------|
| Normalizza timestamp a UTC in `write_heartbeat` wrapper | LOW | 1 riga |
| Documenta enum soft per `status` nel docstring `hb_lib.py` | LOW | 2 righe |
| Considera `is_stale()` come candidato per import in claude-ops | INFO | — |
