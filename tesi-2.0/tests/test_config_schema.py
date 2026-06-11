"""DR-VAULT-jsonschema-config: validazione JSON schema per config pipeline."""

from __future__ import annotations

import importlib
import types

from common.config_validator import validate_config

# ── Test 1: config_step1 valido ───────────────────────────────────────────────


def test_config_step1_valid() -> None:
    """config_step1_matchingpennies supera la validazione schema."""
    cfg = importlib.import_module("config.config_step1_matchingpennies")
    errors = validate_config(cfg)
    assert errors == [], "config_step1 non valido:\n" + "\n".join(errors)


# ── Test 2: config_step2 valido ───────────────────────────────────────────────


def test_config_step2_valid() -> None:
    """config_step2_source_matchingpennies supera la validazione schema."""
    cfg = importlib.import_module("config.config_step2_source_matchingpennies")
    errors = validate_config(cfg)
    assert errors == [], "config_step2 non valido:\n" + "\n".join(errors)


# ── Test 3: config invalida → errori rilevati ─────────────────────────────────


def test_invalid_config_detected() -> None:
    """Config con spacing invalido e subjects vuoto → errori schema rilevati."""
    fake = types.ModuleType("fake_config")
    fake.subjects = []                  # minItems=1 violato
    fake.task = "restingstate"
    fake.ch_types = ["eeg"]
    fake.conditions = ["EC", "EO"]
    fake.spacing = "invalid_spacing"    # enum violato
    fake.loose = 1.5                    # maximum=1 violato

    errors = validate_config(fake)
    assert errors, "Attesi errori di validazione, trovati 0"

    joined = "\n".join(errors)
    # Almeno uno dei tre errori deve essere segnalato
    assert any(
        kw in joined.lower()
        for kw in ("spacing", "subjects", "loose", "minitems", "maximum", "enum")
    ), f"Errori attesi non trovati in:\n{joined}"
