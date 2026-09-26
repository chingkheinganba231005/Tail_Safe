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
sim_app = typer.Typer(help="Run the evacuation simulator.")
app.add_typer(sim_app, name="sim")


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


def _load_or_generate(source: str, storeys: int | None) -> Any:
    from tailsafe.building.io import load_building
    from tailsafe.building.templates import TEMPLATES, generate

    if source in TEMPLATES:
        return generate(source, **({"storeys": storeys} if storeys else {}))
    return load_building(Path(source))


@sim_app.command("run")
def sim_run(
    building: Annotated[
        str, typer.Argument(help="Building JSON path or a template name (e.g. cruciform).")
    ],
    storeys: Annotated[int | None, typer.Option(help="Storeys, when using a template.")] = None,
    slot: Annotated[
        str, typer.Option(help="weekday_day, weekday_night, weekend_day, weekend_night.")
    ] = "weekday_night",
    share_65: Annotated[
        float | None, typer.Option("--share-65", help="Share of residents aged 65+.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    block_stair: Annotated[
        list[str] | None,
        typer.Option(help="Block a staircase as STAIR@SECONDS, e.g. A@240 (repeatable)."),
    ] = None,
    evac_lifts: Annotated[
        bool, typer.Option(help="Use all non-firefighting lifts for evacuation.")
    ] = False,
    lift_out: Annotated[
        list[str] | None, typer.Option(help="Lift out of service as LIFT@SECONDS (repeatable).")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Write the summary JSON here.")] = None,
    plot: Annotated[Path | None, typer.Option(help="Write an evacuation plot here.")] = None,
) -> None:
    """Simulate one evacuation scenario and print the headline results."""
    from tailsafe.population.synth import PopulationConfig, TimeSlot, sample_population
    from tailsafe.sim.meso import SimConfig, SimScenario, run_meso, stair_blockage
    from tailsafe.sim.plot import save_run_plot

    b = _load_or_generate(building, storeys)
    lifts = tuple(lf.id for lf in b.lifts if not lf.firefighting) if evac_lifts else ()
    pop = sample_population(
        b,
        PopulationConfig(
            time_slot=TimeSlot(slot), share_65_plus=share_65, evacuation_lifts=bool(lifts)
        ),
        seed=seed,
    )
    blockages: tuple[Any, ...] = ()
    for spec in block_stair or []:
        sid, _, when = spec.partition("@")
        blockages += stair_blockage(b, sid, float(when or 0.0))
    outages = tuple(
        (lid, float(t or 0.0)) for lid, _, t in (x.partition("@") for x in lift_out or [])
    )
    scenario = SimScenario(blockages=blockages, evacuation_lifts=lifts, lift_outages=outages)
    res = run_meso(b, pop, scenario, SimConfig(record_series=plot is not None))
    summary = {
        "building": b.name,
        "scenario": {
            "time_slot": slot,
            "share_65_plus": share_65,
            "seed": seed,
            "blocked_stairs": block_stair or [],
            "evacuation_lifts": list(lifts),
            "lift_outages": lift_out or [],
        },
        "population": pop.summary(),
        "results": res.summary(),
        "floor_clearance_s": {b.level_label(k): v for k, v in res.floor_clearance().items()},
        "top_queues": res.top_queues(8),
        "disclaimer": DISCLAIMER,
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False, default=float)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    typer.echo(text)
    if plot:
        save_run_plot(res, plot, title=f"{b.name} — {slot}, seed {seed}")
        typer.echo(f"Wrote {plot}")


@app.command()
def validate(
    markdown: Annotated[bool, typer.Option(help="Print a Markdown table.")] = False,
) -> None:
    """Run the analytical validation cases of the simulator."""
    from tailsafe.sim.validation import markdown_table, run_checks

    results = run_checks()
    if markdown:
        typer.echo(markdown_table(results))
    else:
        for r in results:
            mark = "PASS" if r.passed else "FAIL"
            typer.echo(f"[{mark}] {r.name}: reference {r.expected:.2f}, meso {r.simulated:.2f}")
    if not all(r.passed for r in results):
        raise typer.Exit(code=1)


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
