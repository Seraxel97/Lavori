"""Append-only broadcast log per eventi orchestratore.

Registra eventi sprint/worker in `.planning/broadcast_log.jsonl`
con timestamp ISO e lock file per sicurezza concorrente.

Uso tipico:
    from common.broadcast import emit

    emit('sprint_claimed', sprint_id='S-XX', owner='sonnet1-ts')
    emit('sprint_done', sprint_id='S-XX', verdict='PASS')
    emit('worker_idle', worker='sonnet1-ts', age_min=12)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from common.queue_lib import queue_lock

_REPO_ROOT = Path(__file__).parent.parent
LOG_PATH = _REPO_ROOT / ".planning" / "broadcast_log.jsonl"


def emit(event_type: str, **fields) -> None:
    """Appende un evento al broadcast log con timestamp ISO8601 + JSONL.

    Parameters
    ----------
    event_type:
        Tipo evento (es. 'sprint_claimed', 'sprint_done', 'worker_idle').
    **fields:
        Campi aggiuntivi serializzabili in JSON (str, int, float, bool, None).
    """
    entry = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "event": event_type,
        **fields,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with queue_lock(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_log(log_path: Path | None = None) -> list[dict]:
    """Legge tutti gli eventi dal broadcast log.

    Parameters
    ----------
    log_path:
        Path opzionale al file log. Default: LOG_PATH.

    Returns
    -------
    list[dict]
        Lista di eventi in ordine di append. Lista vuota se file assente.
    """
    p = log_path or LOG_PATH
    if not p.exists():
        return []
    events = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events
