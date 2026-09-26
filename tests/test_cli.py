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


def test_sim_run_and_validate(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    png = tmp_path / "run.png"
    res = runner.invoke(
        app,
        [
            "sim",
            "run",
            "slab",
            "--storeys",
            "6",
            "--slot",
            "weekend_night",
            "--block-stair",
            "A@60",
            "--out",
            str(out),
            "--plot",
            str(png),
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert out.exists() and png.exists()
    import json

    data = json.loads(out.read_text())
    assert data["results"]["not_evacuated"] == 0
    res = runner.invoke(app, ["validate"])
    assert res.exit_code == 0 and "FAIL" not in res.stdout


def test_sim_run_with_fire(tmp_path: Path) -> None:
    out = tmp_path / "fire.json"
    png = tmp_path / "fire.png"
    res = runner.invoke(
        app,
        [
            "sim",
            "run",
            "slab",
            "--storeys",
            "6",
            "--fire",
            "L03.unit.02",
            "--fire-door-open",
            "--out",
            str(out),
            "--plot",
            str(png),
        ],
    )
    assert res.exit_code == 0, res.stdout
    import json

    data = json.loads(out.read_text())
    assert data["tenability"] is not None
    assert (tmp_path / "fire_smoke.png").exists()


def test_micro_run_compare_and_fd(tmp_path: Path) -> None:
    import json

    png = tmp_path / "floor.png"
    out = tmp_path / "micro.json"
    base = ["cruciform", "--storeys", "6"]
    res = runner.invoke(
        app,
        [
            "micro",
            "run",
            *base,
            "--index",
            "1",
            "--plot",
            str(png),
            "--level",
            "3",
            "--out",
            str(out),
        ],
    )
    assert res.exit_code == 0, res.stdout
    data = json.loads(out.read_text())
    assert data["micro"]["not_out"] == 0 and png.exists()

    agree = tmp_path / "agree.json"
    res = runner.invoke(app, ["micro", "compare", *base, "--runs", "2", "--out", str(agree)])
    assert res.exit_code == 0, res.stdout
    assert "| Time the last walker gets out |" in res.stdout
    assert json.loads(agree.read_text())["runs"] == 2

    res = runner.invoke(app, ["micro", "fd"])
    assert res.exit_code == 0 and "| Density" in res.stdout
