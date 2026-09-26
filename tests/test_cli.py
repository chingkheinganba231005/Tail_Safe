from __future__ import annotations

from pathlib import Path

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


def test_building_generate_validate_info_render(tmp_path: Path) -> None:
    out = tmp_path / "b.json"
    res = runner.invoke(
        app,
        [
            "building",
            "generate",
            "slab",
            "--storeys",
            "6",
            "--set",
            "flats_per_side=4",
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.stdout
    res = runner.invoke(app, ["building", "validate", str(out)])
    assert res.exit_code == 0 and "0 error(s)" in res.stdout
    res = runner.invoke(app, ["building", "info", str(out)])
    assert res.exit_code == 0 and '"units": 40' in res.stdout
    png = tmp_path / "b.png"
    res = runner.invoke(app, ["building", "render", str(out), "--out", str(png), "--dpi", "40"])
    assert res.exit_code == 0 and png.exists()


def test_building_templates_lists_options() -> None:
    res = runner.invoke(app, ["building", "templates"])
    assert res.exit_code == 0
    assert "cruciform" in res.stdout and "wing_end_stairs" in res.stdout


def test_schema_export(tmp_path: Path) -> None:
    res = runner.invoke(app, ["schema", "export", "--out-dir", str(tmp_path)])
    assert res.exit_code == 0
    assert (tmp_path / "building.schema.json").exists()
