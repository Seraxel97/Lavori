"""Heartbeat atomic write helpers — tempfile+os.replace per evitare write race corruption.

Il pattern tempfile+os.replace garantisce che il reader veda sempre un JSON valido:
o la versione precedente o quella nuova, mai un file parzialmente scritto.

API pubblica:
    write_heartbeat(worker, status, **fields)     — scrive HB atomico
    read_heartbeat(hb_path)                       — legge singolo HB
    read_all_heartbeats(hb_dir) -> list[dict]     — legge tutti gli HB in una dir
    is_stale(hb_dict, threshold_min=10) -> bool   — verifica se l'HB è scaduto
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


def write_heartbeat(hb_path: Path, data: dict) -> None:
    """Scrive il heartbeat in modo atomico (tempfile → os.replace).

    Parameters
    ----------
    hb_path:
        Path del file JSON heartbeat di destinazione.
    data:
        Dict da serializzare come JSON.
    """
    hb_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=hb_path.parent, suffix=".tmp")
    try:
        os.write(fd, content.encode())
        os.close(fd)
        os.replace(tmp, hb_path)
    except Exception:
        os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_heartbeat(hb_path: Path) -> dict | None:
    """Legge il heartbeat; ritorna None se il file non esiste o e' corrotto."""
    try:
        return json.loads(hb_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_all_heartbeats(hb_dir: Path) -> list[dict]:
    """Legge tutti i file JSON heartbeat presenti in una directory.

    Ignora file con estensione .tmp e file non parseable (corrotti o in
    scrittura). Non è ricorsiva: legge solo il livello diretto di hb_dir.

    Parameters
    ----------
    hb_dir:
        Directory da scansionare (es. `.planning/heartbeats_tesi/`).

    Returns
    -------
    list[dict]
        Lista di dict heartbeat, uno per file valido. Campo ``_source_file``
        aggiunto automaticamente con il nome del file (senza directory).
        Ritorna lista vuota se la directory non esiste.
    """
    if not hb_dir.is_dir():
        return []
    results: list[dict] = []
    for fpath in sorted(hb_dir.glob("*.json")):
        if fpath.suffix == ".tmp":
            continue
        hb = read_heartbeat(fpath)
        if hb is not None:
            hb["_source_file"] = fpath.name
            results.append(hb)
    return results


def is_stale(hb: dict, threshold_min: float = 10.0) -> bool:
    """Verifica se un heartbeat è scaduto (last update > threshold_min minuti fa).

    Strategia di rilevamento timestamp (priorità decrescente):
    1. Campo ``ts`` come Unix timestamp float (time.time()).
    2. Campo ``timestamp`` come stringa ISO-8601.
    3. mtime del file (se ``_source_file`` e la dir è nota) — non implementato
       qui; usare ``os.path.getmtime`` direttamente se necessario.

    Se nessun campo temporale è rilevabile, ritorna True (considera stale
    per sicurezza: meglio un falso allarme che ignorare un worker bloccato).

    Parameters
    ----------
    hb:
        Dict heartbeat come ritornato da read_heartbeat.
    threshold_min:
        Soglia di staleness in minuti (default: 10).

    Returns
    -------
    bool
        True se l'heartbeat è considerato scaduto.
    """
    now = time.time()
    threshold_s = threshold_min * 60.0

    # Prova campo ts (float Unix timestamp)
    if "ts" in hb:
        try:
            ts = float(hb["ts"])
            return (now - ts) > threshold_s
        except (TypeError, ValueError):
            pass

    # Prova campo timestamp (ISO-8601 string)
    if "timestamp" in hb:
        try:
            from datetime import datetime
            ts_str: str = hb["timestamp"]
            # Gestisci sia +HH:MM che Z
            if ts_str.endswith("Z"):
                ts_str = ts_str[:-1] + "+00:00"
            dt = datetime.fromisoformat(ts_str)
            ts = dt.timestamp()
            return (now - ts) > threshold_s
        except (TypeError, ValueError, AttributeError):
            pass

    # Nessun timestamp rilevabile → considera stale per sicurezza
    return True
