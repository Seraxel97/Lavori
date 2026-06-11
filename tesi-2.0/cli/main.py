"""CLI unificato Tesi_2.0 — typer app con comandi step1/step2/bench/e2e.

Uso:
    python -m cli.main --help
    python -m cli.main step1
    python -m cli.main step2
    python -m cli.main bench --output reports/BENCH_MATRIX_RESULTS.json
    python -m cli.main e2e --subject sub-05 --atlas aparc --metric wpli
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

app = typer.Typer(
    name="tesi",
    help="Pipeline EEG source-level FC — Tesi_2.0",
    add_completion=False,
)


@app.command()
def step1(
    config: str = typer.Option(
        "config/config_step1_matchingpennies.py",
        "--config",
        "-c",
        help="Path al file di configurazione MNE-BIDS-Pipeline STEP 1.",
    ),
) -> None:
    """Esegui STEP 1: preprocessing (mne-bids-pipeline)."""
    typer.echo(f"[STEP 1] preprocessing — config: {config}")
    result = subprocess.run(
        ["mne_bids_pipeline", "--config", config, "--steps", "preprocessing"],
        check=False,
    )
    if result.returncode != 0:
        typer.echo(f"[STEP 1] FAIL (exit code {result.returncode})", err=True)
        raise typer.Exit(result.returncode)
    typer.echo("[STEP 1] DONE")


@app.command()
def step2(
    config: str = typer.Option(
        "config/config_step2_source_matchingpennies.py",
        "--config",
        "-c",
        help="Path al file di configurazione MNE-BIDS-Pipeline STEP 2.",
    ),
    subject: str = typer.Option("05", "--subject", "-s", help="Subject ID (es. '05')."),
    task: str = typer.Option("matchingpennies", "--task", "-t", help="Task name."),
    method: str = typer.Option("dSPM", "--method", help="Metodo inverso."),
    loose: float = typer.Option(0.2, "--loose", help="Parametro loose."),
    depth: float = typer.Option(0.8, "--depth", help="Parametro depth."),
    lambda2: float = typer.Option(1.0 / 9.0, "--lambda2", help="Regolarizzazione Tikhonov."),
) -> None:
    """Esegui STEP 2: source reconstruction (pipeline + finalize_inverse)."""
    typer.echo(f"[STEP 2] source recon — config: {config}, subject: {subject}")

    r1 = subprocess.run(
        ["mne_bids_pipeline", "--config", config, "--steps", "preprocessing,sensor,source"],
        check=False,
    )
    if r1.returncode != 0:
        typer.echo(f"[STEP 2a] FAIL (exit code {r1.returncode})", err=True)
        raise typer.Exit(r1.returncode)

    r2 = subprocess.run(
        [
            sys.executable, "source_reconstruction/finalize_inverse.py",
            "--subject", subject,
            "--task", task,
            "--method", method,
            "--loose", str(loose),
            "--depth", str(depth),
            "--lambda2", str(lambda2),
        ],
        check=False,
    )
    if r2.returncode != 0:
        typer.echo(f"[STEP 2b] finalize_inverse FAIL (exit code {r2.returncode})", err=True)
        raise typer.Exit(r2.returncode)

    typer.echo("[STEP 2] DONE")


@app.command()
def bench(
    subject: str = typer.Option("sub-05", "--subject", "-s", help="Subject ID BIDS."),
    deriv: str = typer.Option(
        "data/derivatives/mne-bids-pipeline",
        "--deriv",
        help="Root directory derivati.",
    ),
    sfreq_target: float = typer.Option(500.0, "--sfreq-target", help="Sfreq dopo decimazione."),
    n_epochs_max: int = typer.Option(0, "--n-epochs-max", help="Max epoch (0=tutte)."),
    output: str = typer.Option(
        "reports/BENCH_MATRIX_RESULTS.json",
        "--output",
        "-o",
        help="Path output JSON.",
    ),
) -> None:
    """Esegui benchmark matrix (700 run: 7 metriche × 4 atlanti × 5 bande × 5 algo)."""
    typer.echo(f"[BENCH] avvio — subject={subject}, output={output}")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "pipeline_mne_bids.run_bench_matrix",
            "--subject", subject,
            "--deriv", deriv,
            "--sfreq-target", str(sfreq_target),
            "--n-epochs-max", str(n_epochs_max),
            "--out-json", output,
        ],
        check=False,
    )
    if result.returncode != 0:
        typer.echo(f"[BENCH] FAIL (exit code {result.returncode})", err=True)
        raise typer.Exit(result.returncode)
    typer.echo(f"[BENCH] DONE → {output}")


@app.command()
def e2e(
    subject: str = typer.Option("sub-05", "--subject", "-s", help="Subject ID BIDS."),
    deriv: str = typer.Option(
        "data/derivatives/mne-bids-pipeline",
        "--deriv",
        help="Root directory derivati.",
    ),
    atlas: str = typer.Option("aparc", "--atlas", "-a", help="Atlante parcellazione."),
    metric: str = typer.Option("wpli", "--metric", "-m", help="Metrica FC."),
    n_epochs_max: int = typer.Option(0, "--n-epochs-max", help="Max epoch (0=tutte)."),
    output: str = typer.Option(
        "reports/E2E_MATCHINGPENNIES.md",
        "--output",
        "-o",
        help="Path report MD output.",
    ),
) -> None:
    """Esegui pipeline E2E completa (STEP 1-6): epochs → inverse → FC → features → ML."""
    typer.echo(f"[E2E] avvio — subject={subject}, atlas={atlas}, metric={metric}")

    result = subprocess.run(
        [
            sys.executable, "-m", "pipeline_mne_bids.run_e2e_matchingpennies",
            "--subject", subject,
            "--deriv", deriv,
            "--atlas", atlas,
            "--metric", metric,
            "--n-epochs-max", str(n_epochs_max),
            "--out", output,
        ],
        check=False,
    )
    if result.returncode != 0:
        typer.echo(f"[E2E] FAIL (exit code {result.returncode})", err=True)
        raise typer.Exit(result.returncode)
    typer.echo(f"[E2E] DONE → {output}")


if __name__ == "__main__":
    app()
