from __future__ import annotations

from typer.testing import CliRunner

from tailsafe.cli import app

runner = CliRunner()


def test_version() -> None:
    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert "tailsafe" in res.stdout


def test_params_check() -> None:
    res = runner.invoke(app, ["params", "check"])
    assert res.exit_code == 0, res.stdout
    assert "ASSUMPTION" in res.stdout
