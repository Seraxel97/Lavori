"""Structured JSON logger per pipeline events — Tesi_2.0.

Ogni evento viene serializzato come riga JSONL nel file `<log_dir>/<run_id>.jsonl`.
Thread-safe: usa filelock via `threading.Lock` per append atomico.

Uso minimo:
    from common.logger import JsonLogger

    logger = JsonLogger(run_id="e2e_2026-05-01")
    logger.log("step_start", step="preprocessing")
    logger.log("step_done", step="preprocessing", duration_s=3.2)

Schema di ogni entry JSONL:
    {
        "ts":      <float>   Unix timestamp (time.time()),
        "iso":     <str>     ISO-8601 UTC,
        "event":   <str>     Nome evento (es. "step_start"),
        "run_id":  <str>     Identificatore della run,
        ...                  Campi extra passati come kwargs
    }
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JsonLogger:
    """Logger che scrive eventi strutturati in formato JSONL.

    Parameters
    ----------
    run_id:
        Identificatore univoco della run (es. "e2e_2026-05-01_aparc_wpli").
        Diventa il nome del file: `<log_dir>/<run_id>.jsonl`.
    log_dir:
        Directory dove salvare i log. Viene creata se non esiste.
        Default: `reports/runs/`.
    """

    def __init__(
        self,
        run_id: str,
        log_dir: Path | str = Path("reports/runs"),
    ) -> None:
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{run_id}.jsonl"
        self._lock = threading.Lock()

    def log(self, event: str, **fields: Any) -> None:
        """Emette un evento strutturato nel file JSONL.

        Parameters
        ----------
        event:
            Nome dell'evento (es. "step_start", "step_done", "error").
        **fields:
            Campi extra serializzabili in JSON (str, int, float, bool, None).

        Notes
        -----
        L'operazione di append è thread-safe grazie a un lock per istanza.
        Tutti i valori nei fields devono essere JSON-serializzabili.
        """
        entry: dict[str, Any] = {
            "ts": time.time(),
            "iso": datetime.now(UTC).isoformat(),
            "event": event,
            "run_id": self.run_id,
            **fields,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def read_events(self) -> list[dict[str, Any]]:
        """Legge tutti gli eventi dal file JSONL.

        Returns
        -------
        list[dict]
            Lista di eventi in ordine di scrittura.
            Ritorna lista vuota se il file non esiste.
        """
        if not self.path.exists():
            return []
        events = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    def __repr__(self) -> str:
        return f"JsonLogger(run_id={self.run_id!r}, path={self.path})"
