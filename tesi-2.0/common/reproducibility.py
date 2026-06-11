"""Reproducibility harness — manifest builder per run pipeline.

Genera un manifest JSON con metadati di runtime per garantire
riproducibilita' dei risultati: versioni pacchetti, hash config,
SHA git, seeds.

Uso:
    from common.reproducibility import build_manifest, save_manifest

    manifest = build_manifest("bench_run_001", "config/config_step2_source_matchingpennies.py")
    out = save_manifest(manifest, "reports/")
    print(f"Manifest: {out}")
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_CORE_PKGS = frozenset(
    ["mne", "mne-bids", "mne-connectivity", "mne-features", "scikit-learn",
     "numpy", "scipy", "pandas", "matplotlib", "neuromaps"]
)


def _git_sha() -> str | None:
    """Ritorna lo SHA HEAD del repo git, None se non in repo git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=str(_REPO_ROOT), timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _config_hash(config_path: Path) -> str:
    """SHA256 del file config (hex digest)."""
    sha = hashlib.sha256()
    sha.update(config_path.read_bytes())
    return sha.hexdigest()


def _pkg_version(name: str) -> str | None:
    """Versione installata di un pacchetto via importlib.metadata."""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _env_pkgs(requirements_path: Path | None = None) -> dict[str, str | None]:
    """Versioni dei core pkg da requirements.txt (presenti nell'env)."""
    req_path = requirements_path or (_REPO_ROOT / "requirements.txt")
    if not req_path.exists():
        return {pkg: _pkg_version(pkg) for pkg in sorted(_CORE_PKGS)}

    installed: dict[str, str | None] = {}
    for line in req_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # "pkg==1.2.3" o "pkg>=1.0" — prende solo il nome
        pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].strip()
        if pkg_name.lower() in _CORE_PKGS:
            installed[pkg_name] = _pkg_version(pkg_name)
    return installed


def build_manifest(
    run_id: str,
    config_path: str | Path | None = None,
    *,
    random_seed: int = 42,
    requirements_path: Path | None = None,
) -> dict:
    """Costruisce il manifest di riproducibilita' per un run.

    Parameters
    ----------
    run_id:
        Identificatore univoco del run (es. 'bench_run_001').
    config_path:
        Path al file config Python (opzionale). Se fornito, aggiunge config_hash.
    random_seed:
        Seed casuale documentato nel manifest (default 42).
    requirements_path:
        Override path requirements.txt. Default: repo_root/requirements.txt.

    Returns
    -------
    dict
        Manifest con campi: run_id, timestamp, python_version, mne_version,
        numpy_version, sklearn_version, git_sha, config_hash, seeds, env_pkgs.
    """
    manifest: dict = {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "mne_version": _pkg_version("mne"),
        "numpy_version": _pkg_version("numpy"),
        "sklearn_version": _pkg_version("scikit-learn"),
        "git_sha": _git_sha(),
        "config_path": str(config_path) if config_path else None,
        "config_hash": None,
        "seeds": {
            "random_state": random_seed,
            "numpy_seed": random_seed,
        },
        "env_pkgs": _env_pkgs(requirements_path),
    }

    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            manifest["config_hash"] = _config_hash(p)

    return manifest


def save_manifest(manifest: dict, output_dir: str | Path | None = None) -> Path:
    """Salva il manifest su `reports/manifests/<run_id>.json`.

    Parameters
    ----------
    manifest:
        Dizionario prodotto da build_manifest.
    output_dir:
        Directory radice per i manifesti. Default: repo_root/reports/manifests/.

    Returns
    -------
    Path
        Path del file JSON salvato.
    """
    base = Path(output_dir) if output_dir else (_REPO_ROOT / "reports" / "manifests")
    base.mkdir(parents=True, exist_ok=True)
    run_id = manifest.get("run_id", "unknown")
    out = base / f"{run_id}.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return out
