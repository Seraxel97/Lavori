"""Validazione JSON schema per moduli config Python della pipeline.

Estrae gli attributi rilevanti da un modulo config (via vars/inspect)
e li valida contro `config/config.schema.json`.

Uso:
    import importlib
    from common.config_validator import validate_config

    cfg = importlib.import_module("config.config_step2_source_matchingpennies")
    errors = validate_config(cfg)
    if errors:
        for e in errors:
            print("ERRORE:", e)
    else:
        print("Config valida")
"""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import jsonschema

_SCHEMA_PATH = Path(__file__).parent.parent / "config" / "config.schema.json"

_RELEVANT_ATTRS = frozenset(
    [
        "subjects",
        "task",
        "ch_types",
        "conditions",
        "l_freq",
        "h_freq",
        "reject",
        "inverse_method",
        "spacing",
        "loose",
        "depth",
        "noise_cov",
    ]
)


def _extract_config_dict(module: ModuleType) -> dict:
    """Estrae gli attributi rilevanti dal modulo config come dizionario."""
    raw = vars(module)
    return {k: v for k, v in raw.items() if k in _RELEVANT_ATTRS}


def validate_config(module: ModuleType, schema_path: Path | None = None) -> list[str]:
    """Valida un modulo config Python contro il JSON schema pipeline.

    Parameters
    ----------
    module:
        Modulo Python importato (es. config.config_step1_matchingpennies).
    schema_path:
        Path opzionale al file schema JSON. Default: config/config.schema.json.

    Returns
    -------
    list[str]
        Lista di messaggi di errore. Lista vuota se il config e' valido.
    """
    schema_file = schema_path or _SCHEMA_PATH
    schema = json.loads(schema_file.read_text())
    instance = _extract_config_dict(module)

    errors: list[str] = []
    validator = jsonschema.Draft7Validator(schema)
    for err in sorted(validator.iter_errors(instance), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in err.path) if err.path else "<root>"
        errors.append(f"{path}: {err.message}")
    return errors
