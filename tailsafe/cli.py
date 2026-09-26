"""Command-line interface: ``tailsafe --help``."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from tailsafe import DISCLAIMER, __version__
from tailsafe.config import Params

app = typer.Typer(
    help="TailSafe — tail-risk evacuation stress-testing for high-rise Hong Kong.",
    no_args_is_help=True,
    add_completion=False,
)
params_app = typer.Typer(help="Inspect the parameter registry (config/params.yaml).")
app.add_typer(params_app, name="params")
building_app = typer.Typer(help="Generate, validate, inspect and render buildings.")
app.add_typer(building_app, name="building")
schema_app = typer.Typer(help="JSON schemas generated from the data models.")
app.add_typer(schema_app, name="schema")


@app.command()
def version() -> None:
    """Print the TailSafe version and the responsible-use notice."""
    typer.echo(f"tailsafe {__version__}")
    typer.echo(DISCLAIMER)


@params_app.command("check")
def params_check(
    path: Annotated[Path | None, typer.Option(help="Registry file to check.")] = None,
) -> None:
    """Validate the registry and summarise how many values still need citations."""
    params = Params.load(path)
    leaves = list(params.leaves())
    pending = params.assumption_report()
    typer.echo(f"{params.origin}: {len(leaves)} parameters, all with a source field.")
    typer.echo(f"{len(pending)} are ASSUMPTIONs that still need a citation.")


@params_app.command("list")
def params_list(
    assumptions: Annotated[
        bool, typer.Option("--assumptions", help="Only values that still need a citation.")
    ] = False,
    path: Annotated[Path | None, typer.Option(help="Registry file to read.")] = None,
) -> None:
    """List parameters with their units and sources."""
    params = Params.load(path)
    rows = params.assumption_report() if assumptions else list(params.leaves())
    for p in rows:
        unit = f" [{p.unit}]" if p.unit else ""
        typer.echo(f"{p.path}{unit}  ({p.dist})\n    source: {p.source}")


def _parse_sets(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"expected key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw
    return out


@building_app.command("templates")
def building_templates() -> None:
    """List procedural templates and their options."""
    from tailsafe.building.templates import TEMPLATES

    for name, fn in TEMPLATES.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        typer.echo(f"{name}: {doc}")
        for pname, par in inspect.signature(fn).parameters.items():
            if pname != "params":
                typer.echo(f"    {pname} = {par.default!r}")


@building_app.command("generate")
def building_generate(
    template: Annotated[str, typer.Argument(help="cruciform, slab, twin_core or care_home")],
    out: Annotated[Path, typer.Option(help="Output JSON path.")],
    storeys: Annotated[int | None, typer.Option(help="Storeys including G/F.")] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Template option as key=value (repeatable)."),
    ] = None,
    pretty: Annotated[bool, typer.Option(help="Indent the JSON.")] = False,
) -> None:
    """Generate a building from a procedural Hong Kong template."""
    from tailsafe.building.io import save_building
    from tailsafe.building.templates import generate

    kwargs = _parse_sets(set_ or [])
    if storeys is not None:
        kwargs["storeys"] = storeys
    b = generate(template, **kwargs)
    save_building(b, out, pretty=pretty)
    typer.echo(f"Wrote {out} ({len(b.nodes)} nodes, {len(b.edges)} edges, {len(b.units)} units)")


@building_app.command("validate")
def building_validate(path: Annotated[Path, typer.Argument(help="Building JSON.")]) -> None:
    """Validate a building file against the JSON schema and semantic rules."""
    from tailsafe.building.io import validate_against_schema
    from tailsafe.building.model import Building
    from tailsafe.building.validate import validate_building

    data = json.loads(path.read_text(encoding="utf-8"))
    validate_against_schema(data)
    issues = validate_building(Building.model_validate(data))
    for issue in issues:
        typer.echo(str(issue))
    n_err = sum(1 for i in issues if i.severity == "error")
    typer.echo(f"{path}: {n_err} error(s), {len(issues) - n_err} warning(s)")
    if n_err:
        raise typer.Exit(code=1)


@building_app.command("info")
def building_info(path: Annotated[Path, typer.Argument(help="Building JSON.")]) -> None:
    """Print headline facts about a building."""
    from tailsafe.building.graph import summary
    from tailsafe.building.io import load_building

    typer.echo(json.dumps(summary(load_building(path)), indent=2, ensure_ascii=False))


@building_app.command("render")
def building_render(
    path: Annotated[Path, typer.Argument(help="Building JSON.")],
    out: Annotated[Path, typer.Option(help="Image path (.png, .svg or .pdf).")],
    level: Annotated[int | None, typer.Option(help="Level to draw in plan.")] = None,
    dpi: Annotated[int, typer.Option(help="Resolution.")] = 150,
) -> None:
    """Render a plan of one storey and the 3D stack."""
    from tailsafe.building.io import load_building
    from tailsafe.building.render import save_render

    save_render(load_building(path), out, level=level, dpi=dpi)
    typer.echo(f"Wrote {out}")


@schema_app.command("export")
def schema_export(
    out_dir: Annotated[Path | None, typer.Option(help="Directory (default: schemas/).")] = None,
) -> None:
    """Regenerate the JSON schema files from the Pydantic models."""
    from tailsafe.building.io import export_schemas

    for path in export_schemas(out_dir):
        typer.echo(f"Wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
