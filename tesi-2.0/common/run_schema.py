"""Run output schema standardization — JSONSchema + validator per pipeline run.

Definisce lo schema standard per l'output di ogni run pipeline (E2E, bench, ecc.)
e fornisce una funzione di validazione per verificare la conformita'.

Schema fields:
  run_id       — identificatore univoco (required)
  timestamp    — ISO8601 UTC (required)
  manifest     — dizionario riproducibilita' da common.reproducibility (required)
  steps        — lista di step result (optional, partial runs allowed)
  errors       — lista di errori/warning (optional, non invalida il run)
  summary      — metriche riassuntive finali (optional)

Uso:
    from common.run_schema import validate_run_schema, RUN_SCHEMA

    errors = validate_run_schema(my_run_output)
    if errors:
        print("Schema invalid:", errors)
"""

from __future__ import annotations

import json

import jsonschema

_STEP_RESULT_SCHEMA: dict = {
    "type": "object",
    "required": ["step_name", "status"],
    "additionalProperties": True,
    "properties": {
        "step_name": {"type": "string", "minLength": 1},
        "status": {"type": "string", "enum": ["ok", "error", "skipped", "partial"]},
        "elapsed_s": {"type": "number", "minimum": 0},
        "output": {"type": "object", "additionalProperties": True},
        "error_msg": {"type": ["string", "null"]},
    },
}

_MANIFEST_SCHEMA: dict = {
    "type": "object",
    "required": ["run_id", "timestamp", "python_version"],
    "additionalProperties": True,
    "properties": {
        "run_id": {"type": "string"},
        "timestamp": {"type": "string"},
        "python_version": {"type": "string"},
        "git_sha": {"type": ["string", "null"]},
        "config_hash": {"type": ["string", "null"]},
        "seeds": {"type": "object"},
        "env_pkgs": {"type": "object"},
    },
}

RUN_SCHEMA: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://tesi.local/run_output.schema.json",
    "title": "Pipeline run output",
    "description": "Schema per l'output standardizzato di ogni run pipeline Tesi_2.0.",
    "type": "object",
    "required": ["run_id", "timestamp", "manifest"],
    "additionalProperties": True,
    "properties": {
        "run_id": {
            "type": "string",
            "minLength": 1,
            "description": "Identificatore univoco del run.",
        },
        "timestamp": {
            "type": "string",
            "description": "Timestamp ISO8601 UTC di inizio run.",
        },
        "manifest": {
            **_MANIFEST_SCHEMA,
            "description": "Manifest di riproducibilita' (da common.reproducibility).",
        },
        "steps": {
            "type": "array",
            "items": _STEP_RESULT_SCHEMA,
            "description": "Lista di step result (ammessa lista parziale per run interrotti).",
        },
        "errors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Lista errori/warning — presenza non invalida il run.",
        },
        "summary": {
            "type": "object",
            "additionalProperties": True,
            "description": "Metriche riassuntive finali (ba_mean, n_features, ecc.).",
        },
    },
}


def validate_run_schema(json_data: dict | str) -> list[str]:
    """Valida un run output contro RUN_SCHEMA.

    Parameters
    ----------
    json_data:
        Dizionario o stringa JSON del run output.

    Returns
    -------
    list[str]
        Lista di messaggi di errore (vuota = valido).
    """
    if isinstance(json_data, str):
        try:
            json_data = json.loads(json_data)
        except json.JSONDecodeError as exc:
            return [f"JSON non parsabile: {exc}"]

    errors: list[str] = []
    validator = jsonschema.Draft7Validator(RUN_SCHEMA)
    for err in sorted(validator.iter_errors(json_data), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in err.path) if err.path else "<root>"
        errors.append(f"{path}: {err.message}")
    return errors


def build_run_output(
    run_id: str,
    manifest: dict,
    *,
    steps: list[dict] | None = None,
    errors: list[str] | None = None,
    summary: dict | None = None,
) -> dict:
    """Costruisce un run output conforme allo schema.

    Parameters
    ----------
    run_id:
        ID del run.
    manifest:
        Manifest da build_manifest() (common.reproducibility).
    steps, errors, summary:
        Campi opzionali.

    Returns
    -------
    dict
        Run output validabile con validate_run_schema.
    """
    from datetime import UTC, datetime
    return {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "manifest": manifest,
        "steps": steps or [],
        "errors": errors or [],
        "summary": summary or {},
    }
