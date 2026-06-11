# Vault Skills Pattern — Valutazione per Tesi_2.0

**Sprint**: DR-VAULT-skill-init  
**Data**: 2026-05-01  
**Worker**: sonnet1-ts

---

## Contesto

Le *skill* Claude Code sono file di testo (`SKILL.md`) che istruiscono il modello
su pattern operativi ripetibili (bootstrap sessione, dispatch worker, monitoring).
Risiedono in `~/.claude/skills/` (globali) o in `.claude/skills/` (project-level).

Il catalogo globale (`claude-ops/skills/`) include 39 skill attualmente
disponibili tramite symlink:

| Categoria         | Skill rilevanti per Tesi_2.0 |
|-------------------|------------------------------|
| Worker/executor   | `worker`, `silent-worker`, `boot-minimal` |
| Orchestrazione    | `orchestrator`, `orchestrator-sonnet`, `dispatch`, `swarm-dispatch` |
| Queue management  | `queue_ops` (Python module), `orchestrator_select_batch`, `orchestrator_reorder_queue` |
| Monitoring        | `monitor`, `monitor-lite`, `gap-review` |
| Comunicazione     | `tmux-dispatch`, `bot-terminal-bridge`, `notify-vesta` |
| Review/qualità    | `gate-verify`, `dependency_gate`, `handover` |

---

## Analisi: serve una skill project-level per Tesi_2.0?

### Argomenti contro (raccomandato: NON introdurre)

1. **Copertura completa dal globale** — `worker`, `dispatch`, `monitor-lite`,
   `queue_ops` coprono già il ciclo completo claim→work→done→notify senza
   personalizzazione di dominio.

2. **Logica di dominio in Python** — `common/run_id.py`, `common/queue_lib.py`,
   `common/reproducibility.py` e `common/state_lib.py` incapsulano tutta la logica
   Tesi-specifica in moduli testabili. Duplicarla in un file `SKILL.md`
   introdurrebbe drift tra la skill e il codice effettivo.

3. **Session bootstrap già coperto** — Il blocco "Re-bootstrap post /clear" riportato
   nel prompt di avvio include già le istruzioni specifiche Tesi_2.0 (heartbeat path,
   queue path, naming convention MSG_TO_ORCH). Non è necessaria una skill separata.

4. **Manutenzione a carico zero** — Skills project-level richiedono aggiornamento
   sincronizzato con le API Python sottostanti. Oggi questo overhead non è giustificato.

### Quando potrebbe servire (futuro)

- Se il re-bootstrap post /clear dovesse superare ~200 token e fosse richiamato
  spesso da sessioni "fredde" senza SYNC disponibile.
- Se si introducesse un pattern "tesi-benchmark-run" sufficientemente distinto
  dal `worker` generico da giustificare un override.

---

## Raccomandazione

**Non introdurre skill project-level** in questa fase.  
Mantenere tutta la logica Tesi_2.0 in `common/` (Python testabile).  
Riesaminare se emerge un pattern ripetitivo non coperto dai globali.

---

## Verdict

`PASS` — valutazione completata, nessuna azione implementativa richiesta.
