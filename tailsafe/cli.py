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
vision_app = typer.Typer(help="Read floor-plan images into buildings.")
app.add_typer(vision_app, name="vision")
surrogate_app = typer.Typer(help="Graph surrogate: training data, training, evaluation.")
app.add_typer(surrogate_app, name="surrogate")


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

    from tailsafe.scenarios.spec import ScenarioSpec, reference_spec

    if spec == "reference":
        return reference_spec()
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
    """Print the reference scenario specification (a starting point for your own)."""
    from tailsafe.scenarios.spec import reference_spec

    typer.echo(reference_spec().model_dump_json(indent=2, exclude_none=True))


@stress_app.command("run")
def stress_run(
    building: Annotated[str, typer.Argument(help="Building JSON path or a template name.")],
    spec: Annotated[
        str, typer.Option(help="'reference' or a JSON/YAML ScenarioSpec file.")
    ] = "reference",
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
    spec: Annotated[
        str, typer.Option(help="'reference' or a JSON/YAML ScenarioSpec file.")
    ] = "reference",
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


def _briefing_distributions(stress: Path, optimization: Path | None) -> dict[str, Any]:
    from tailsafe.scenarios.montecarlo import MCResult

    if optimization and (optimization / "confirm_plan").exists():
        return {
            "Baseline": MCResult.load(optimization / "confirm_baseline").loss("total_time"),
            "With plan": MCResult.load(optimization / "confirm_plan").loss("total_time"),
        }
    if (stress / "result.json").exists():
        return {"Baseline": MCResult.load(stress).loss("total_time")}
    return {}


@app.command()
def brief(
    stress: Annotated[Path, typer.Option(help="Directory from `stress run --out`.")],
    optimization: Annotated[
        Path | None, typer.Option(help="Directory from `optimize --out` (optional).")
    ] = None,
    writer: Annotated[
        str, typer.Option(help="auto (LLM when configured), template or llm.")
    ] = "auto",
    out: Annotated[Path, typer.Option(help="Markdown file.")] = Path("out/briefing.md"),
    pdf: Annotated[Path | None, typer.Option(help="One-page PDF (or .png).")] = Path(
        "out/briefing.pdf"
    ),
) -> None:
    """One-page briefing for the building manager; every number checked against the results."""
    from tailsafe.report.briefing import briefing_facts, briefing_pdf, make_briefing

    if writer not in ("auto", "template", "llm"):
        raise typer.BadParameter("--writer must be auto, template or llm")

    def load_json(path: Path) -> dict[str, Any] | None:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    metrics = load_json(stress / "metrics.json")
    if metrics is None:
        raise typer.BadParameter(f"{stress}/metrics.json not found; run `stress run --out` first")
    opt = load_json(optimization / "optimization.json") if optimization else None
    facts = briefing_facts(metrics, load_json(stress / "bottlenecks.json"), opt)
    result = make_briefing(facts, use_llm={"auto": None, "template": False, "llm": True}[writer])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.markdown, encoding="utf-8")
    out.with_suffix(".facts.json").write_text(
        json.dumps(facts, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    typer.echo(result.markdown)
    typer.echo(f"Source: {result.source}. Wrote {out}", err=True)
    if result.note:
        typer.echo(f"Note: {result.note}", err=True)
    if pdf:
        briefing_pdf(result.markdown, pdf, _briefing_distributions(stress, optimization) or None)
        typer.echo(f"Wrote {pdf}", err=True)


@app.command("export-site")
def export_site(
    out: Annotated[Path, typer.Option(help="Directory the static web build serves.")] = Path(
        "web/public/data"
    ),
    runs: Annotated[int, typer.Option(help="Stress-test scenarios per building.")] = 300,
    template: Annotated[
        list[str] | None, typer.Option(help="Building types to include (default: all four).")
    ] = None,
    micro: Annotated[bool, typer.Option(help="Include the person-by-person replays.")] = True,
) -> None:
    """Record the data for the browser version (static hosting, no server)."""
    from tailsafe.api.static_site import export_static_site

    manifest = export_static_site(
        out, templates=template, runs=runs, micro=micro, log=lambda m: typer.echo(m, err=True)
    )
    n = len(manifest["buildings"])
    typer.echo(f"Wrote {manifest['responses']} responses for {n} buildings to {out}")


@micro_app.command("run")
def micro_run(
    building: Annotated[str, typer.Argument(help="Building JSON path or a template name.")],
    spec: Annotated[
        str, typer.Option(help="'reference' or a JSON/YAML ScenarioSpec file.")
    ] = "reference",
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
    spec: Annotated[
        str, typer.Option(help="'reference' or a JSON/YAML ScenarioSpec file.")
    ] = "reference",
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


def _parse_scale(scale: str | None) -> Any:
    from tailsafe.vision.detect import Scale

    if not scale:
        return None
    try:
        x1, y1, x2, y2, metres = (float(v) for v in scale.split(","))
    except ValueError as exc:
        raise typer.BadParameter("--scale is x1,y1,x2,y2,metres (pixels and metres)") from exc
    return Scale(x1=x1, y1=y1, x2=x2, y2=y2, metres=metres)


@vision_app.command("synth")
def vision_synth(
    template: Annotated[str, typer.Argument(help="Template to render a floor of.")] = "cruciform",
    level: Annotated[int, typer.Option(help="Floor to render.")] = 1,
    px_per_m: Annotated[float, typer.Option(help="Resolution.")] = 20.0,
    noise: Annotated[float, typer.Option(help="Grey noise (0-1).")] = 0.03,
    out: Annotated[Path, typer.Option(help="PNG to write.")] = Path("out/plan.png"),
) -> None:
    """Render a synthetic floor plan (with known answer) to try the reader on."""
    from tailsafe.building.templates import generate
    from tailsafe.vision.synth import render_plan, to_png_bytes

    b = generate(template)
    img, truth = render_plan(b, level, px_per_m=px_per_m, noise=noise, blur=0.5 if noise else 0.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(to_png_bytes(img))
    (out.with_suffix(".truth.json")).write_text(
        json.dumps(truth.as_dict(), indent=1), encoding="utf-8"
    )
    typer.echo(
        f"Wrote {out} ({img.shape[1]}×{img.shape[0]} px, {1 / px_per_m:.3f} m/px) and its truth"
    )


@vision_app.command("detect")
def vision_detect(
    image: Annotated[Path, typer.Argument(help="PNG, JPEG or PDF floor plan.")],
    scale: Annotated[
        str | None, typer.Option(help="Reference line: x1,y1,x2,y2,metres (image pixels).")
    ] = None,
    out: Annotated[Path | None, typer.Option(help="Detection JSON to write.")] = None,
    overlay: Annotated[Path | None, typer.Option(help="PNG with the detection drawn.")] = None,
) -> None:
    """Find walls, doorways, rooms and stairs in a plan image."""
    from tailsafe.vision.detect import detect_plan
    from tailsafe.vision.graph import detection_summary
    from tailsafe.vision.overlay import save_overlay
    from tailsafe.vision.raster import load_image

    img = load_image(image)
    det = detect_plan(img, _parse_scale(scale))
    typer.echo(json.dumps(detection_summary(det), indent=2))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(det.model_dump_json(indent=1), encoding="utf-8")
    if overlay:
        save_overlay(img, det, overlay)
        typer.echo(f"Wrote {overlay}", err=True)


@vision_app.command("build")
def vision_build(
    detection: Annotated[Path, typer.Argument(help="Detection JSON (possibly corrected).")],
    storeys: Annotated[int, typer.Option(help="Storeys including G/F.")] = 10,
    name: Annotated[str, typer.Option(help="Building name.")] = "Building from floor plan",
    out: Annotated[Path, typer.Option(help="Building JSON to write.")] = Path(
        "out/plan_building.json"
    ),
) -> None:
    """Stack a detected floor into a building JSON (validated)."""
    from tailsafe.building.graph import summary
    from tailsafe.building.io import save_building
    from tailsafe.vision.detect import PlanDetection
    from tailsafe.vision.graph import plan_to_building

    det = PlanDetection.model_validate_json(detection.read_text(encoding="utf-8"))
    b = plan_to_building(det, storeys=storeys, name=name)
    save_building(b, out)
    typer.echo(json.dumps(summary(b), indent=2, ensure_ascii=False))


@vision_app.command("eval")
def vision_eval(
    reference_scale: Annotated[
        bool, typer.Option(help="Give the reader the true scale (as a reference line).")
    ] = True,
) -> None:
    """Precision and recall on synthetic plans rendered from the templates."""
    from tailsafe.vision.evaluate import evaluate, evaluation_markdown

    typer.echo(evaluation_markdown(evaluate(reference_scale=reference_scale)))


@surrogate_app.command("data")
def surrogate_data(
    cases: Annotated[int, typer.Option(help="Number of (building, scenario) cases.")] = 800,
    runs: Annotated[int, typer.Option(help="Monte Carlo runs per case.")] = 64,
    seed: Annotated[int, typer.Option(help="Seed of the case draw and the runs.")] = 0,
    start: Annotated[int, typer.Option(help="First case index (to extend a data set).")] = 0,
    workers: Annotated[int | None, typer.Option(help="Processes (default: all CPUs).")] = None,
    out: Annotated[Path, typer.Option(help="JSON-lines file.")] = Path("out/surrogate/cases.jsonl"),
) -> None:
    """Simulate random buildings and scenarios as training data."""
    from tailsafe.surrogate.data import generate_dataset, save_dataset

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            typer.echo(f"  {done}/{total} cases", err=True)

    recs = generate_dataset(
        cases, n_runs=runs, seed=seed, start=start, workers=workers, progress=progress
    )
    save_dataset(recs, out)
    typer.echo(f"Wrote {len(recs)} cases to {out}")


@surrogate_app.command("eval")
def surrogate_eval(
    data: Annotated[Path, typer.Option(help="Data set from `surrogate data`.")] = Path(
        "out/surrogate/cases.jsonl"
    ),
    epochs: Annotated[int, typer.Option(help="Training epochs per split.")] = 120,
    out: Annotated[Path | None, typer.Option(help="Write the full result JSON here.")] = Path(
        "out/surrogate/eval.json"
    ),
) -> None:
    """Random split and leave-one-typology-out evaluation (accuracy, calibration, speed)."""
    from tailsafe.surrogate.data import load_dataset
    from tailsafe.surrogate.evaluate import evaluate, evaluation_markdown
    from tailsafe.surrogate.model import TrainConfig

    res = evaluate(
        load_dataset(data),
        tcfg=TrainConfig(epochs=epochs),
        log=lambda m: typer.echo(m, err=True),
    )
    typer.echo(evaluation_markdown(res))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=float) + "\n", encoding="utf-8")


@surrogate_app.command("train")
def surrogate_train(
    data: Annotated[Path, typer.Option(help="Data set from `surrogate data`.")] = Path(
        "out/surrogate/cases.jsonl"
    ),
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 120,
    evaluation: Annotated[
        Path | None, typer.Option(help="Result of `surrogate eval`, stored with the weights.")
    ] = Path("out/surrogate/eval.json"),
    out: Annotated[
        Path | None, typer.Option(help="Weights file (default: the shipped model).")
    ] = None,
) -> None:
    """Train on every case and save the weights used by the API and the what-if screen."""
    from collections import Counter
    from datetime import UTC, datetime

    import numpy as np

    from tailsafe.surrogate.data import load_dataset
    from tailsafe.surrogate.evaluate import prepare
    from tailsafe.surrogate.model import ModelConfig, TrainConfig, save_model, train
    from tailsafe.surrogate.predictor import DEFAULT_WEIGHTS

    records = load_dataset(data)
    graphs, targets, meta = prepare(records)
    order = np.random.default_rng(0).permutation(len(graphs))
    n_val = max(1, len(order) // 10)
    val, tr = order[:n_val], order[n_val:]
    mcfg = ModelConfig()
    net, stats, hist = train(
        [graphs[i] for i in tr],
        [targets[i] for i in tr],
        [graphs[i] for i in val],
        [targets[i] for i in val],
        mcfg=mcfg,
        tcfg=TrainConfig(epochs=epochs),
        log=lambda m: typer.echo(m, err=True),
    )
    summary = None
    if evaluation and evaluation.exists():
        ev = json.loads(evaluation.read_text(encoding="utf-8"))
        summary = {
            "random_p95_relative_error": {
                k: v["relative_error_p95"] for k, v in ev["random"]["losses"].items()
            },
            "holdout_p95_relative_error": {
                typ: {k: v["relative_error_p95"] for k, v in m["losses"].items()}
                for typ, m in ev["holdout"].items()
            },
            "speedup_single_core": ev.get("speed", {}).get("speedup_single_core"),
        }
    target = out or DEFAULT_WEIGHTS
    save_model(
        target,
        net,
        stats,
        mcfg,
        {
            "trained_on": dict(Counter(m["template"] for m in meta)),
            "cases": len(records),
            "runs_per_case": int(records[0].get("n_runs", 64)) if records else 0,
            "epochs": len(hist),
            "created": datetime.now(UTC).strftime("%Y-%m-%d"),
            "evaluation": summary,
        },
    )
    typer.echo(f"Wrote {target} ({target.stat().st_size / 1024:.0f} KiB)")


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
