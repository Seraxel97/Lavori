"""Test CLI unificato — typer app callable senza errori (--help e subcomandi)."""

from __future__ import annotations

from typer.testing import CliRunner

from cli.main import app

runner = CliRunner()


def test_main_help() -> None:
    """App principale risponde a --help senza errori."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Pipeline EEG" in result.output
    assert "step1" in result.output
    assert "step2" in result.output
    assert "bench" in result.output
    assert "e2e" in result.output


def test_step1_help() -> None:
    """Subcomando step1 --help è accessibile."""
    result = runner.invoke(app, ["step1", "--help"])
    assert result.exit_code == 0
    assert "STEP 1" in result.output
    assert "--config" in result.output


def test_step2_help() -> None:
    """Subcomando step2 --help è accessibile."""
    result = runner.invoke(app, ["step2", "--help"])
    assert result.exit_code == 0
    assert "STEP 2" in result.output
    assert "--config" in result.output
    assert "--subject" in result.output
    assert "--method" in result.output


def test_bench_help() -> None:
    """Subcomando bench --help è accessibile."""
    result = runner.invoke(app, ["bench", "--help"])
    assert result.exit_code == 0
    assert "700" in result.output or "benchmark" in result.output.lower()
    assert "--output" in result.output


def test_e2e_help() -> None:
    """Subcomando e2e --help è accessibile."""
    result = runner.invoke(app, ["e2e", "--help"])
    assert result.exit_code == 0
    assert "E2E" in result.output
    assert "--atlas" in result.output
    assert "--metric" in result.output


def test_unknown_command() -> None:
    """Comando sconosciuto produce exit code != 0."""
    result = runner.invoke(app, ["nonexistent"])
    assert result.exit_code != 0
