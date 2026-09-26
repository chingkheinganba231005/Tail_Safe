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
stress_app = typer.Typer(help="Monte Carlo stress tests and tail-risk metrics.")
app.add_typer(stress_app, name="stress")
micro_app = typer.Typer(help="Microscopic replays and meso–micro cross-checks.")
app.add_typer(micro_app, name="micro")


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
    fire: Annotated[
        str | None, typer.Option(help="Start a fire in this flat (node id), e.g. L14.unit.N3.")
    ] = None,
    fire_door_open: Annotated[
        bool, typer.Option(help="The fire flat's door is left open.")
    ] = False,
    out: Annotated[Path | None, typer.Option(help="Write the summary JSON here.")] = None,
    plot: Annotated[Path | None, typer.Option(help="Write an evacuation plot here.")] = None,
) -> None:
    """Simulate one evacuation scenario and print the headline results."""
    from tailsafe.config import get_params
    from tailsafe.hazard.model import FireSpec, HazardModel
    from tailsafe.hazard.plot import save_hazard_plot
    from tailsafe.population.synth import PopulationConfig, TimeSlot, sample_population
    from tailsafe.risk.tenability import rset_aset
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
    hazard = None
    if fire:
        pr = get_params()
        fire_spec = FireSpec(
            node=fire,
            growth=float(pr["hazard.fire.growth_coefficient"].ppf([0.5])[0]),
            peak=float(pr["hazard.fire.peak_hrr"].ppf([0.5])[0]),
            door_open=fire_door_open,
        )
        hazard = HazardModel(b, pr).run(fire_spec, keep_fields=plot is not None)
    scenario = SimScenario(
        blockages=blockages, evacuation_lifts=lifts, lift_outages=outages, hazard=hazard
    )
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
        "tenability": _tenability_json(rset_aset(res), b) if hazard is not None else None,
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
        if hazard is not None:
            hz_plot = plot.with_name(plot.stem + "_smoke" + plot.suffix)
            save_hazard_plot(b, hazard, hz_plot)
            typer.echo(f"Wrote {hz_plot}")


def _tenability_json(ra: Any, b: Any) -> dict[str, Any]:
    fails = [
        {"floor": b.level_label(int(lv)), "rset_s": float(r), "aset_s": float(a)}
        for lv, r, a in zip(ra.floor_levels, ra.floor_rset, ra.floor_aset, strict=True)
        if r > a
    ]
    return {
        "rset_exceeds_aset": ra.fails,
        "failing_floors": fails,
        "occupants_over_fed_limit": ra.occupants_over_fed_limit,
        "occupants_incapacitated": ra.occupants_incapacitated,
        "max_fed": ra.max_fed,
    }


def _load_spec(spec: str) -> Any:
    import yaml

    from tailsafe.scenarios.spec import ScenarioSpec, demo_spec

    if spec == "demo":
        return demo_spec()
    path = Path(spec)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ScenarioSpec.model_validate(data)


def _fmt_minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f}"


def _print_risk(result: Any) -> None:
    from tailsafe.scenarios.montecarlo import LOSSES

    typer.echo(
        f"\n{result.n} scenarios in {result.elapsed:.1f} s — {result.building_name}, "
        f"'{result.spec.name}' (minutes; [95% CI])"
    )
    header = f"{'loss':<24}{'mean':>8}{'median':>8}{'P95':>8}{'P99':>8}{'CVaR95':>10}  CVaR95 CI"
    typer.echo(header)
    for name in LOSSES:
        r = result.risk(name)
        typer.echo(
            f"{name:<24}{_fmt_minutes(r.mean.value):>8}{_fmt_minutes(r.median.value):>8}"
            f"{_fmt_minutes(r.p95.value):>8}{_fmt_minutes(r.p99.value):>8}"
            f"{_fmt_minutes(r.cvar.value):>10}  "
            f"[{_fmt_minutes(r.cvar.lo)}, {_fmt_minutes(r.cvar.hi)}]"
        )
    if result.spec.hazard is not None and result.spec.hazard.enabled:
        ten = result.tenability_summary()
        p = ten["p_rset_exceeds_aset"]
        inc = ten["p_any_incapacitated"]
        typer.echo(
            f"\nP(RSET > ASET) = {p['value']:.3f} [{p['lo']:.3f}, {p['hi']:.3f}]   "
            f"P(anyone incapacitated) = {inc['value']:.3f} [{inc['lo']:.3f}, {inc['hi']:.3f}]   "
            f"mean incapacitated = {ten['mean_incapacitated']:.2f}"
        )
        for row in ten["worst_floors"]:
            typer.echo(
                f"  floor {row['level']:>3}: P(RSET_floor > ASET_floor) = {row['value']:.3f}"
            )


@stress_app.command("spec")
def stress_spec() -> None:
    """Print the demo scenario specification (a starting point for your own)."""
    from tailsafe.scenarios.spec import demo_spec

    typer.echo(demo_spec().model_dump_json(indent=2, exclude_none=True))


@stress_app.command("run")
def stress_run(
    building: Annotated[str, typer.Argument(help="Building JSON path or a template name.")],
    spec: Annotated[str, typer.Option(help="'demo' or a JSON/YAML ScenarioSpec file.")] = "demo",
    storeys: Annotated[int | None, typer.Option(help="Storeys, when using a template.")] = None,
    runs: Annotated[int, typer.Option(help="Number of scenarios.")] = 1000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    workers: Annotated[int | None, typer.Option(help="Processes (default: all CPUs).")] = None,
    lhs: Annotated[bool, typer.Option(help="Latin Hypercube scenario design.")] = True,
    target_halfwidth: Annotated[
        float | None,
        typer.Option(help="Stop early when the CVaR95 CI half-width (s) is below this."),
    ] = None,
    loss: Annotated[str, typer.Option(help="Loss for the breakdown and plot.")] = "total_time",
    out: Annotated[Path | None, typer.Option(help="Directory for results and plots.")] = None,
) -> None:
    """Run a Monte Carlo stress test and report tail-risk metrics."""
    from tailsafe.risk.breakdown import tail_breakdown
    from tailsafe.risk.plot import save_stress_plot
    from tailsafe.scenarios.montecarlo import MCConfig, run_monte_carlo

    b = _load_or_generate(building, storeys)
    sc = _load_spec(spec)
    cfg = MCConfig(
        n_runs=runs, seed=seed, workers=workers, lhs=lhs, target_halfwidth=target_halfwidth
    )

    def progress(done: int, total: int) -> None:
        if done % max(total // 10, 1) == 0 or done == total:
            typer.echo(f"  {done}/{total} scenarios", err=True)

    result = run_monte_carlo(b, sc, cfg, progress=progress)
    _print_risk(result)
    breakdowns = {
        name: tail_breakdown(result, name) for name in ("total_time", "self_evacuation_time")
    }
    for name, bdn in breakdowns.items():
        if bdn["headline"]:
            typer.echo(f"\n[{name}] {bdn['headline']}")
    bd = breakdowns.get(loss) or tail_breakdown(result, loss)
    if out:
        result.save(out)
        report = {**result.summary(), "breakdown": breakdowns, "disclaimer": DISCLAIMER}
        (out / "metrics.json").write_text(
            json.dumps(report, indent=2, default=float, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        save_stress_plot(result, out / "distribution.png", bd, loss)
        typer.echo(f"\nWrote {out}/result.json, arrays.npz, metrics.json, distribution.png")


@stress_app.command("report")
def stress_report(
    directory: Annotated[Path, typer.Argument(help="Directory written by `stress run --out`.")],
    loss: Annotated[str, typer.Option(help="Loss for the breakdown.")] = "total_time",
) -> None:
    """Re-print metrics and the tail breakdown of a saved stress test."""
    from tailsafe.risk.breakdown import tail_breakdown
    from tailsafe.scenarios.montecarlo import MCResult

    result = MCResult.load(directory)
    _print_risk(result)
    bd = tail_breakdown(result, loss)
    typer.echo("")
    for row in bd["profiles"]:
        typer.echo(
            f"  {row['category']:<28} occupants {100 * row['occupant_share']:5.1f}%   "
            f"tail stragglers {100 * row['straggler_share']:5.1f}%   ratio {row['risk_ratio']:.1f}"
        )
    if bd["headline"]:
        typer.echo("\n" + bd["headline"])


@stress_app.command("bottlenecks")
def stress_bottlenecks(
    directory: Annotated[Path, typer.Argument(help="Directory written by `stress run --out`.")],
    loss: Annotated[str, typer.Option(help="Loss whose CVaR is attributed.")] = "p95_occupant_time",
    factor: Annotated[
        float, typer.Option(help="Capacity multiplier for what-if relaxations.")
    ] = 1.5,
    rerun_fraction: Annotated[
        float, typer.Option(help="Initial share of worst scenarios to re-run.")
    ] = 0.2,
    workers: Annotated[int | None, typer.Option(help="Processes (default: all CPUs).")] = None,
) -> None:
    """Rank bottlenecks by counterfactual change in CVaR (re-runs the tail scenarios)."""
    from tailsafe.analysis.bottlenecks import attribute_bottlenecks
    from tailsafe.analysis.plot import save_bottleneck_plot
    from tailsafe.scenarios.montecarlo import MCResult

    result = MCResult.load(directory)
    table = attribute_bottlenecks(
        result, loss=loss, factor=factor, rerun_fraction=rerun_fraction, workers=workers
    )
    typer.echo(
        f"Max egress flow {table['max_flow_persons_per_s']:.2f} persons/s; "
        f"min cut: {', '.join(table['min_cut'])}"
    )
    typer.echo(f"\nWorst queues in the tail ({loss}):")
    for q in table["queues"][:6]:
        typer.echo(
            f"  {q['where']:<45} in {100 * q['recurrence']:5.1f}% of tail scenarios, "
            f"{q['tail_person_seconds'] / 60:7.0f} person-min"
        )
    typer.echo(f"\nCounterfactual ranking (ΔCVaR{int(100 * table['alpha'])}, min, 95% CI):")
    for r in table["ranking"]:
        d = r["delta_cvar"]
        typer.echo(
            f"  {r['rank']:>2}. {r['label']:<45} {d['value'] / 60:7.2f}  "
            f"[{d['lo'] / 60:6.2f}, {d['hi'] / 60:6.2f}]"
        )
    if table["headline"]:
        typer.echo("\n" + table["headline"])
    (directory / "bottlenecks.json").write_text(
        json.dumps(table, indent=2, default=float, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    save_bottleneck_plot(table, directory / "bottlenecks.png")
    typer.echo(f"\nWrote {directory}/bottlenecks.json and bottlenecks.png")


@app.command("optimize")
def optimize_cmd(
    building: Annotated[str, typer.Argument(help="Building JSON path or a template name.")],
    spec: Annotated[str, typer.Option(help="'demo' or a JSON/YAML ScenarioSpec file.")] = "demo",
    storeys: Annotated[int | None, typer.Option(help="Storeys, when using a template.")] = None,
    objective: Annotated[str, typer.Option(help="cvar, p_rset or weighted.")] = "cvar",
    loss: Annotated[str, typer.Option(help="Loss for cvar / weighted objectives.")] = "total_time",
    scenarios: Annotated[int, typer.Option(help="Scenarios per evaluation (SAA sample).")] = 100,
    confirm: Annotated[int, typer.Option(help="Fresh scenarios for confirmation.")] = 400,
    max_wardens: Annotated[int, typer.Option(help="Maximum number of floor wardens.")] = 2,
    levers: Annotated[
        str, typer.Option(help="Comma-separated: lifts,hold_open,stair_assignment,phasing,wardens")
    ] = "lifts,hold_open,stair_assignment,phasing,wardens",
    cmaes_iterations: Annotated[int, typer.Option(help="CMA-ES iterations for phasing.")] = 4,
    seed: Annotated[int, typer.Option(help="Seed of the optimisation sample.")] = 1,
    workers: Annotated[int | None, typer.Option(help="Processes (default: all CPUs).")] = None,
    out: Annotated[Path | None, typer.Option(help="Directory for the plan and plots.")] = None,
) -> None:
    """Search for the operational plan that shrinks the tail most, then confirm it."""
    from tailsafe.optimize.plot import save_before_after
    from tailsafe.optimize.search import Objective, OptimizeConfig, optimize

    b = _load_or_generate(building, storeys)
    sc = _load_spec(spec)
    obj = Objective.model_validate({"kind": objective, "loss": loss})
    cfg = OptimizeConfig(
        n_scenarios=scenarios,
        confirm_scenarios=confirm,
        max_wardens=max_wardens,
        levers=tuple(x.strip() for x in levers.split(",") if x.strip()),
        cmaes_iterations=cmaes_iterations,
        seed=seed,
        workers=workers,
    )
    typer.echo(f"Minimising {obj.label()} over {scenarios} scenarios per plan…", err=True)
    result = optimize(b, sc, obj, cfg, log=lambda m: typer.echo(m, err=True))
    summary = result.summary(b)
    typer.echo(f"\n{summary['evaluations']} plans evaluated. Best plan:")
    for line in summary["plan_description"]:
        typer.echo(f"  • {line}")
    conf = result.confirmation
    typer.echo(f"\nConfirmed on {conf['scenarios']} fresh scenarios (seed {conf['seed']}):")
    for name, row in conf["losses"].items():
        d = row["delta_cvar"]
        before, after = row["before_cvar"] / 60, row["after_cvar"] / 60
        typer.echo(
            f"  CVaR95 {name:<22} {before:7.1f} → {after:7.1f} min"
            f"   Δ {d['value'] / 60:+.1f} [{d['lo'] / 60:+.1f}, {d['hi'] / 60:+.1f}]"
            f"{'  (significant)' if row['significant'] else ''}"
        )
    pr = conf["p_rset_exceeds_aset"]
    typer.echo(
        f"  P(RSET > ASET)              {pr['before']:.3f} → {pr['after']:.3f}"
        f"   Δ {pr['delta']['value']:+.3f} [{pr['delta']['lo']:+.3f}, {pr['delta']['hi']:+.3f}]"
    )
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "optimization.json").write_text(
            json.dumps(summary, indent=2, default=float, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (out / "plan.json").write_text(result.best.plan.model_dump_json(indent=2) + "\n")
        save_before_after(
            result.confirm_baseline,
            result.confirm_best,
            summary["plan_description"],
            out / "before_after.png",
        )
        result.confirm_baseline.save(out / "confirm_baseline")
        result.confirm_best.save(out / "confirm_plan")
        typer.echo(f"\nWrote {out}/optimization.json, plan.json, before_after.png")


@app.command()
def pitch(
    stress: Annotated[Path, typer.Option(help="Directory from `stress run --out`.")] = Path(
        "out/demo1000"
    ),
    optimization: Annotated[
        Path | None, typer.Option(help="Directory from `optimize --out`.")
    ] = Path("out/opt-demo"),
    out: Annotated[Path, typer.Option(help="Markdown file to write.")] = Path(
        "docs/pitch_metrics.md"
    ),
) -> None:
    """Regenerate docs/pitch_metrics.md from the latest saved results."""
    from tailsafe.report.pitch import load_json, pitch_markdown

    metrics = load_json(stress / "metrics.json")
    if metrics is None:
        raise typer.BadParameter(f"{stress}/metrics.json not found; run `stress run --out` first")
    bottlenecks = load_json(stress / "bottlenecks.json")
    opt = load_json(optimization / "optimization.json") if optimization else None
    sources = {"stress test": str(stress / "metrics.json")}
    if bottlenecks:
        sources["bottlenecks"] = str(stress / "bottlenecks.json")
    if opt and optimization:
        sources["optimisation"] = str(optimization / "optimization.json")
    out.write_text(pitch_markdown(metrics, bottlenecks, opt, sources=sources), encoding="utf-8")
    typer.echo(f"Wrote {out}")


@micro_app.command("run")
def micro_run(
    building: Annotated[str, typer.Argument(help="Building JSON path or a template name.")],
    spec: Annotated[str, typer.Option(help="'demo' or a JSON/YAML ScenarioSpec file.")] = "demo",
    storeys: Annotated[int | None, typer.Option(help="Storeys, when using a template.")] = None,
    index: Annotated[int, typer.Option(help="Scenario index (as in `stress run`).")] = 0,
    seed: Annotated[int, typer.Option(help="Seed of the stress test.")] = 0,
    plot: Annotated[Path | None, typer.Option(help="PNG snapshot of one floor.")] = None,
    level: Annotated[int, typer.Option(help="Floor for --plot.")] = 1,
    time: Annotated[float, typer.Option(help="Time (s) for --plot.")] = 300.0,
    out: Annotated[Path | None, typer.Option(help="Write the summary JSON here.")] = None,
) -> None:
    """Replay one scenario person by person and compare it with the meso engine."""
    from tailsafe.scenarios.sampler import ScenarioSampler, scenario_uniforms
    from tailsafe.sim.meso import run_meso
    from tailsafe.sim.micro import run_micro
    from tailsafe.sim.network import compile_network
    from tailsafe.sim.plot import save_micro_frame

    b = _load_or_generate(building, storeys)
    sc_spec = _load_spec(spec)
    u = scenario_uniforms(seed, index, 1)[0]
    sc = ScenarioSampler(b, sc_spec).sample(seed, index, u)
    net = compile_network(b)
    me = run_meso(net, sc.population, sc.sim)
    mi = run_micro(net, sc.population, sc.sim, meso=me, seed=seed, index=index)
    report = {"scenario": index, "seed": seed, "meso": me.summary(), "micro": mi.summary()}
    typer.echo(json.dumps(report, indent=2, default=float))
    if plot:
        save_micro_frame(mi, level, time, plot)
        typer.echo(f"Wrote {plot}", err=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")


@micro_app.command("compare")
def micro_compare(
    building: Annotated[str, typer.Argument(help="Building JSON path or a template name.")],
    spec: Annotated[str, typer.Option(help="'demo' or a JSON/YAML ScenarioSpec file.")] = "demo",
    storeys: Annotated[int | None, typer.Option(help="Storeys, when using a template.")] = None,
    runs: Annotated[int, typer.Option(help="Scenarios to compare.")] = 20,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    out: Annotated[Path | None, typer.Option(help="Write the full result JSON here.")] = None,
) -> None:
    """Meso–micro agreement on the same scenarios (bias, correlation, RMSE)."""
    from tailsafe.analysis.agreement import agreement_markdown, meso_micro_agreement

    b = _load_or_generate(building, storeys)

    def progress(done: int, total: int) -> None:
        typer.echo(f"  {done}/{total} scenarios", err=True)

    res = meso_micro_agreement(b, _load_spec(spec), runs, seed=seed, progress=progress)
    typer.echo(agreement_markdown(res))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")


@micro_app.command("fd")
def micro_fd() -> None:
    """Speed–density relation of the micro model in a periodic corridor."""
    from tailsafe.sim.micro import fundamental_diagram

    typer.echo(
        "| Density (persons/m²) | Micro speed (m/s) | Hydraulic speed (m/s) "
        "| Micro flow (persons/m/s) | Hydraulic flow (persons/m/s) |"
    )
    typer.echo("|---:|---:|---:|---:|---:|")
    for r in fundamental_diagram():
        typer.echo(
            f"| {r['density']:.2f} | {r['speed']:.2f} | {r['hydraulic_speed']:.2f} "
            f"| {r['flow']:.2f} | {r['hydraulic_flow']:.2f} |"
        )


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
