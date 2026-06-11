# Handover Guide — Tesi_2.0 Worker Sessions

Protocollo per passaggio di contesto tra sessioni worker (claude-ops pattern).

---

## Quando compilare il template

Il worker deve generare un handover al termine di ogni sessione, **prima** di terminare:

1. Tutti gli sprint della sessione sono done (heartbeat status=done)
2. MSG_TO_ORCH scritti per ogni sprint completato
3. Queue aggiornata (nessun claimed rimasto aperto)

Il file di destinazione è `.planning/HANDOVER_<worker>_<timestamp>.md`  
(es. `.planning/HANDOVER_sonnet3-ts_20260501T1230.md`)

---

## Come compilare

Copia `.planning/HANDOVER_TEMPLATE.md` e sostituisci i placeholder `{...}`:

```bash
cp .planning/HANDOVER_TEMPLATE.md .planning/HANDOVER_{WORKER}_{YYYYMMDDTHHMMSS}.md
```

Poi compila le sezioni:

| Sezione | Fonte | Note |
|---------|-------|------|
| Sprint completati | MSG_TO_ORCH della sessione | Solo quelli di questa sessione |
| Sprint in corso | Queue claimed non done | Solo se sessione interrotta |
| Branch/push | `git branch -v` | Include sha e push status |
| Queue status | `python3 -c "..."` (vedi sotto) | Snapshot al momento del handover |
| Blockers | QUESTIONS_FOR_OPERATOR.md | Blockers aperti rilevanti |
| Next actions | Personale judgement | Max 3-5 item, ordinati per priorità |

### Snippet queue status

```python
import json
from collections import Counter
from pathlib import Path
rows = [json.loads(l) for l in Path('.planning/DISPATCH_QUEUE_tesi.jsonl').read_text().splitlines() if l.strip()]
print(Counter(r['status'] for r in rows))
print("claimable:", [r['id'] for r in rows if r['status']=='queued' and not r.get('owner')])
```

---

## Come leggere il handover (worker successivo)

Al boot di una nuova sessione:

1. Leggi l'handover più recente: `ls -t .planning/HANDOVER_*.md | head -1`
2. Controlla la sezione **Blockers** — sono ancora attivi?
3. Controlla la sezione **Sprint in corso** — c'è lavoro da riprendere?
4. Vai a **Next actions** e segui l'ordine suggerito
5. Se il handover è >2h, esegui un fresh `cat DISPATCH_QUEUE` per vedere se la queue è cambiata

---

## Regole generali

- **Un handover per sessione** — non accumulare in un file unico
- **No recap verbosi** — se è ovvio dal codice o dalla queue, non scriverlo
- **Timestamp UTC** — usa `datetime.now(UTC).isoformat()`
- **Commit il file** — il handover va committato sul branch della sessione prima del push

---

## Pattern di riferimento

Il progetto usa il ciclo: **claim → heartbeat working → lavoro → heartbeat done → MSG_TO_ORCH → auto-claim next**.

Il handover si inserisce tra l'ultimo `heartbeat done` e la fine della sessione.

```
... heartbeat done (ultimo sprint)
→ handover compilato
→ git commit + push branch
→ sessione terminata
```
