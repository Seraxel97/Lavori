"""S-42: test reproducibility manifest builder."""

from __future__ import annotations

from pathlib import Path

from common.reproducibility import build_manifest, save_manifest

_REQUIRED_FIELDS = {
    "run_id", "timestamp", "python_version", "platform",
    "mne_version", "numpy_version", "sklearn_version",
    "git_sha", "config_path", "config_hash", "seeds", "env_pkgs",
}


def test_build_manifest_fields() -> None:
    """build_manifest ritorna tutti i campi richiesti con valori coerenti."""
    config = Path("config/config_step2_source_matchingpennies.py")
    manifest = build_manifest("test_run_001", config)

    # Tutti i campi obbligatori presenti
    missing = _REQUIRED_FIELDS - set(manifest.keys())
    assert not missing, f"Campi mancanti nel manifest: {missing}"

    # Campi con valore non-null attesi
    assert manifest["run_id"] == "test_run_001"
    assert manifest["python_version"], "python_version vuoto"
    assert manifest["mne_version"], "mne_version vuoto (mne dovrebbe essere installato)"
    assert manifest["numpy_version"], "numpy_version vuoto"
    assert manifest["seeds"]["random_state"] == 42
    assert manifest["seeds"]["numpy_seed"] == 42
    assert isinstance(manifest["env_pkgs"], dict)
    assert len(manifest["env_pkgs"]) > 0, "env_pkgs vuoto"

    # config_hash calcolato se config esiste
    if config.exists():
        assert manifest["config_hash"] is not None, "config_hash None con config esistente"
        assert len(manifest["config_hash"]) == 64, "SHA256 atteso a 64 char hex"

    # timestamp ISO8601 con 'T'
    assert "T" in manifest["timestamp"], f"timestamp non ISO8601: {manifest['timestamp']}"

    # save + verifica file prodotto
    out = save_manifest(manifest, Path("/tmp/test_manifests_s42"))
    assert out.exists(), f"File manifest non creato: {out}"
    assert out.stat().st_size > 0
    import json
    loaded = json.loads(out.read_text())
    assert loaded["run_id"] == "test_run_001"


def test_config_hash_stable() -> None:
    """Stesso config → stesso SHA256 (determinismo hash)."""
    config = Path("config/config_step1_matchingpennies.py")
    if not config.exists():
        import pytest
        pytest.skip("config_step1 non trovato")

    m1 = build_manifest("hash_run_A", config)
    m2 = build_manifest("hash_run_B", config)

    assert m1["config_hash"] is not None
    assert m1["config_hash"] == m2["config_hash"], (
        f"Hash instabile per stesso file: {m1['config_hash']} != {m2['config_hash']}"
    )

    # Hash diverso per config diverso
    config2 = Path("config/config_step2_source_matchingpennies.py")
    if config2.exists():
        m3 = build_manifest("hash_run_C", config2)
        assert m3["config_hash"] != m1["config_hash"], (
            "Hash identico per config diversi — SHA256 non differenzia i file"
        )
