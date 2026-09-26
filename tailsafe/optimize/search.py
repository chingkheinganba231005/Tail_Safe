"""Search for the operational plan that shrinks the tail most.

Sample-average approximation with common random numbers: every candidate plan
is evaluated on the *same* ``n_scenarios`` sampled scenarios, so differences
between plans are not swamped by scenario noise. The search:

1. **Screen** each lever on its own (lift dispatch rules, door hold-open, stair
   assignments by floor band, phased-release presets, single wardens).
2. **Combine** greedily: add the best setting of each improving lever while the
   objective keeps improving.
3. **Wardens**: add wardens one at a time (greedy, up to ``max_wardens``) on the
   most promising floors.
4. **Refine** the phased-release delays with a small CMA-ES.
5. **Confirm** the final plan against the baseline on *fresh* scenarios (a
   different seed), reporting paired-bootstrap confidence intervals. In-sample
   improvements are optimistically biased (winner's curse); the confirmation is
   the number to quote.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from tailsafe.building.model import Building, LevelKind
from tailsafe.config import Params, get_params
from tailsafe.optimize.cmaes import cmaes
from tailsafe.optimize.plan import InterventionPlan
from tailsafe.risk.metrics import Estimate, cvar, paired_difference
from tailsafe.scenarios.montecarlo import MCConfig, MCResult, MonteCarloPool, run_monte_carlo
from tailsafe.scenarios.spec import ScenarioSpec
from tailsafe.sim.meso import SimConfig

Log = Callable[[str], None]
LEVERS = ("lifts", "hold_open", "stair_assignment", "phasing", "wardens")


class Objective(BaseModel):
    """What to minimise."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["cvar", "p_rset", "weighted"] = "cvar"
    loss: str = "total_time"
    alpha: float = 0.95
    mean_weight: float = 0.5

    def value(self, res: MCResult) -> float:
        """Objective value of a Monte Carlo result (lower is better)."""
        x = res.loss(self.loss)
        c = cvar(x, self.alpha)
        if self.kind == "cvar":
            return c
        if self.kind == "p_rset":
            # Tie-break equal probabilities by the tail of the loss.
            return res.p_rset_exceeds_aset().value + 1e-9 * c
        return self.mean_weight * float(np.mean(x)) + (1 - self.mean_weight) * c

    def label(self) -> str:
        """Human-readable objective."""
        if self.kind == "p_rset":
            return "P(RSET > ASET)"
        name = self.loss.replace("_", " ")
        if self.kind == "cvar":
            return f"CVaR{int(100 * self.alpha)} of {name}"
        w = self.mean_weight
        return f"{w:g}·mean + {1 - w:g}·CVaR{int(100 * self.alpha)} of {name}"


@dataclass(frozen=True)
class OptimizeConfig:
    """Search budget and allowed levers."""

    n_scenarios: int = 100
    seed: int = 1
    workers: int | None = None
    levers: tuple[str, ...] = LEVERS
    max_wardens: int = 2
    phasing_bands: int = 4
    max_delay: float = 900.0
    cmaes_iterations: int = 4
    confirm_scenarios: int = 400
    confirm_seed: int = 10_000
    tolerance: float = 1e-6
    sim: SimConfig = field(default_factory=SimConfig)


@dataclass
class Evaluation:
    """One evaluated plan."""

    plan: InterventionPlan
    value: float
    stage: str
    cvar: float
    mean: float
    p_rset: float


@dataclass
class OptimizationResult:
    """Outcome of :func:`optimize`."""

    objective: Objective
    baseline: Evaluation
    best: Evaluation
    history: list[Evaluation]
    confirmation: dict[str, Any]
    confirm_baseline: MCResult
    confirm_best: MCResult

    def summary(self, building: Building | None = None) -> dict[str, Any]:
        """JSON-friendly summary."""

        def ev(e: Evaluation) -> dict[str, Any]:
            return {
                "stage": e.stage,
                "objective": e.value,
                "cvar": e.cvar,
                "mean": e.mean,
                "p_rset_exceeds_aset": e.p_rset,
                "plan": e.plan.model_dump(),
                "levers": e.plan.levers(),
            }

        return {
            "objective": self.objective.model_dump() | {"label": self.objective.label()},
            "baseline": ev(self.baseline),
            "best": ev(self.best),
            "plan_description": self.best.plan.describe(building),
            "evaluations": len(self.history),
            "history": [ev(e) for e in self.history],
            "confirmation": self.confirmation,
        }


class Evaluator:
    """Evaluates plans on a fixed scenario sample (cached by plan)."""

    def __init__(
        self,
        building: Building,
        spec: ScenarioSpec,
        objective: Objective,
        pool: MonteCarloPool,
        cfg: MCConfig,
        log: Log | None = None,
    ) -> None:
        self.building = building
        self.spec = spec
        self.objective = objective
        self.pool = pool
        self.cfg = cfg
        self.log = log
        self.cache: dict[str, Evaluation] = {}
        self.results: dict[str, MCResult] = {}
        self.history: list[Evaluation] = []

    def __call__(self, plan: InterventionPlan, stage: str) -> Evaluation:
        """Evaluate ``plan`` (from cache if seen before)."""
        key = plan.model_dump_json()
        if key in self.cache:
            return self.cache[key]
        res = run_monte_carlo(
            self.building, plan.apply(self.spec, self.building), self.cfg, pool=self.pool
        )
        x = res.loss(self.objective.loss)
        e = Evaluation(
            plan=plan,
            value=self.objective.value(res),
            stage=stage,
            cvar=cvar(x, self.objective.alpha),
            mean=float(np.mean(x)),
            p_rset=res.p_rset_exceeds_aset().value,
        )
        self.cache[key] = e
        self.results[key] = res
        self.history.append(e)
        if self.log:
            self.log(f"  [{stage}] {'; '.join(plan.describe())} -> {e.value:.4g}")
        return e


def _residential_levels(building: Building) -> list[int]:
    return sorted(
        {n.level for n in building.units}
        - {lv.index for lv in building.levels if lv.kind == LevelKind.REFUGE}
        - {0}
    )


def _bands(levels: list[int], k: int) -> list[int]:
    """Lowest level of bands 2..k splitting ``levels`` into ``k`` near-equal groups."""
    if len(levels) < k or k < 2:
        return []
    cuts = [levels[round(i * len(levels) / k)] for i in range(1, k)]
    return sorted(set(cuts))


def merge(a: InterventionPlan, b: InterventionPlan) -> InterventionPlan:
    """``a`` with every lever that ``b`` sets taken from ``b``."""
    upd: dict[str, Any] = {}
    if b.evacuation_lifts:
        upd.update(evacuation_lifts=True, lift_priority=b.lift_priority)
    if b.hold_open_stair_doors:
        upd["hold_open_stair_doors"] = True
    if b.stair_split_level is not None:
        upd.update(
            stair_split_level=b.stair_split_level,
            upper_stair=b.upper_stair,
            lower_stair=b.lower_stair,
        )
    if any(d > 0 for d in b.band_delays):
        upd.update(band_edges=b.band_edges, band_delays=b.band_delays)
    if b.warden_levels:
        upd["warden_levels"] = sorted(set(a.warden_levels) | set(b.warden_levels))
    return a.model_copy(update=upd)


def screening_plans(
    building: Building, spec: ScenarioSpec, cfg: OptimizeConfig
) -> dict[str, list[InterventionPlan]]:
    """Single-lever plans to screen."""
    out: dict[str, list[InterventionPlan]] = {}
    levels = _residential_levels(building)
    if "lifts" in cfg.levers and any(not lf.firefighting for lf in building.lifts):
        out["lifts"] = [
            InterventionPlan(evacuation_lifts=True, lift_priority=r)
            for r in ("top_down", "nearest", "bottom_up")
        ]
    if "hold_open" in cfg.levers and spec.hazard is not None:
        out["hold_open"] = [InterventionPlan(hold_open_stair_doors=True)]
    stairs = [s.id for s in building.stairs]
    if "stair_assignment" in cfg.levers and len(stairs) >= 2 and levels:
        plans = []
        splits = sorted({levels[len(levels) * q // 4] for q in (1, 2, 3)} | {levels[0]})
        for up, low in ((stairs[0], stairs[1]), (stairs[1], stairs[0])):
            for sp in splits:
                plans.append(
                    InterventionPlan(stair_split_level=sp, upper_stair=up, lower_stair=low)
                )
        out["stair_assignment"] = plans
    if "phasing" in cfg.levers and levels:
        edges = _bands(levels, cfg.phasing_bands)
        k = len(edges) + 1
        step = cfg.max_delay / max(k, 2)
        presets = [
            [step * (k - 1 - b) for b in range(k)],  # upper floors first
            [step * b for b in range(k)],  # lower floors first
        ]
        if spec.fire_level is not None:
            fire_band = sum(1 for e in edges if spec.fire_level >= e)
            presets.append([0.0 if b in (fire_band, fire_band + 1) else step for b in range(k)])
        out["phasing"] = [InterventionPlan(band_edges=edges, band_delays=d) for d in presets]
    return out


def warden_candidates(
    building: Building, spec: ScenarioSpec, base: MCResult, limit: int = 6
) -> list[int]:
    """Floors most worth a warden: the fire floor, floors whose residents most
    often wait for rescue, and the slowest-clearing floors."""
    levels = _residential_levels(building)
    scores: dict[int, float] = {lv: 0.0 for lv in levels}
    for r in base.runs:
        if r.groups is not None:
            resc = r.groups["rescued"].astype(bool)
            for lv, size in zip(r.groups["level"][resc], r.groups["size"][resc], strict=True):
                scores[int(lv)] = scores.get(int(lv), 0.0) + float(size)
        for lv, t in zip(r.floor_levels, r.floor_rset, strict=True):
            if lv in scores and np.isfinite(t):
                scores[lv] += t / 3600.0
    ranked = sorted(scores, key=lambda lv: -scores[lv])
    out: list[int] = []
    if spec.fire_level is not None and spec.fire_level in scores:
        out.append(spec.fire_level)
    for lv in ranked:
        if lv not in out:
            out.append(lv)
        if len(out) >= limit:
            break
    return out


def optimize(
    building: Building,
    spec: ScenarioSpec,
    objective: Objective | None = None,
    config: OptimizeConfig | None = None,
    *,
    params: Params | None = None,
    log: Log | None = None,
) -> OptimizationResult:
    """Find a plan that reduces the objective, then confirm it on fresh scenarios."""
    obj = objective or Objective()
    cfg = config or OptimizeConfig()
    p = params or get_params()
    mc = MCConfig(
        n_runs=cfg.n_scenarios,
        seed=cfg.seed,
        batch_size=cfg.n_scenarios,
        workers=cfg.workers,
        keep_groups=True,
        sim=cfg.sim,
    )
    say = log or (lambda _msg: None)
    with MonteCarloPool(building, p, cfg.workers) as pool:
        ev = Evaluator(building, spec, obj, pool, mc, log)
        say("Baseline")
        base = ev(InterventionPlan(), "baseline")
        base_res = ev.results[InterventionPlan().model_dump_json()]

        # 1. screening
        say("Screening single levers")
        best_by_lever: dict[str, Evaluation] = {}
        for lever, plans in screening_plans(building, spec, cfg).items():
            evals = [ev(pl, f"screen:{lever}") for pl in plans]
            best_by_lever[lever] = min(evals, key=lambda e: e.value)
        if "wardens" in cfg.levers and cfg.max_wardens > 0:
            cands = warden_candidates(building, spec, base_res)
            evals = [ev(InterventionPlan(warden_levels=[lv]), "screen:wardens") for lv in cands]
            if evals:
                best_by_lever["wardens"] = min(evals, key=lambda e: e.value)

        # 2. greedy combination
        say("Combining improving levers")
        improving = sorted(
            (e for e in best_by_lever.values() if e.value < base.value - cfg.tolerance),
            key=lambda e: e.value,
        )
        current = improving[0] if improving else base
        for e in improving[1:]:
            cand = ev(merge(current.plan, e.plan), "combine")
            if cand.value < current.value - cfg.tolerance:
                current = cand

        # 3. more wardens, greedily
        if "wardens" in cfg.levers and cfg.max_wardens > 1:
            say("Adding wardens")
            cands = warden_candidates(building, spec, base_res)
            while len(current.plan.warden_levels) < cfg.max_wardens:
                trials = [
                    ev(
                        current.plan.model_copy(
                            update={"warden_levels": sorted({*current.plan.warden_levels, lv})}
                        ),
                        "wardens",
                    )
                    for lv in cands
                    if lv not in current.plan.warden_levels
                ]
                if not trials:
                    break
                best = min(trials, key=lambda e: e.value)
                if best.value < current.value - cfg.tolerance:
                    current = best
                else:
                    break

        # 4. refine phasing delays
        levels = _residential_levels(building)
        if "phasing" in cfg.levers and cfg.cmaes_iterations > 0 and levels:
            say("Refining phased release (CMA-ES)")
            edges = current.plan.band_edges or _bands(levels, cfg.phasing_bands)
            k = len(edges) + 1
            x0 = np.array(
                current.plan.band_delays if current.plan.band_delays else [0.0] * k, dtype=float
            )

            def f(x: np.ndarray) -> float:
                delays = [float(round(v / 30.0) * 30.0) for v in x]  # 30 s steps: cache hits
                plan = current.plan.model_copy(update={"band_edges": edges, "band_delays": delays})
                return ev(plan, "cmaes").value

            res = cmaes(
                f,
                x0,
                np.zeros(k),
                np.full(k, cfg.max_delay),
                sigma0=0.25,
                iterations=cfg.cmaes_iterations,
                seed=cfg.seed,
            )
            if res.f < current.value - cfg.tolerance:
                delays = [float(round(v / 30.0) * 30.0) for v in res.x]
                current = ev(
                    current.plan.model_copy(update={"band_edges": edges, "band_delays": delays}),
                    "cmaes",
                )

        best_eval = min([current, base], key=lambda e: e.value)

        # 5. confirmation on fresh scenarios
        say(f"Confirming on {cfg.confirm_scenarios} fresh scenarios")
        cc = MCConfig(
            n_runs=cfg.confirm_scenarios,
            seed=cfg.confirm_seed,
            batch_size=min(100, cfg.confirm_scenarios),
            workers=cfg.workers,
            keep_groups=True,
            sim=cfg.sim,
        )
        before = run_monte_carlo(building, spec, cc, pool=pool)
        after = run_monte_carlo(building, best_eval.plan.apply(spec, building), cc, pool=pool)
    confirmation = confirm(before, after, obj)
    return OptimizationResult(
        objective=obj,
        baseline=base,
        best=best_eval,
        history=ev.history,
        confirmation=confirmation,
        confirm_baseline=before,
        confirm_best=after,
    )


def confirm(before: MCResult, after: MCResult, objective: Objective) -> dict[str, Any]:
    """Paired comparison of two results on the same scenarios."""
    out: dict[str, Any] = {"scenarios": before.n, "seed": before.config.seed, "losses": {}}
    for loss in ("total_time", "self_evacuation_time", "p95_occupant_time"):
        b, a = before.loss(loss), after.loss(loss)
        d = paired_difference(b, a, alpha=objective.alpha, seed=before.config.seed)
        out["losses"][loss] = {
            "before_cvar": cvar(b, objective.alpha),
            "after_cvar": cvar(a, objective.alpha),
            "delta_cvar": asdict(d),
            "relative": d.value / cvar(b, objective.alpha) if cvar(b, objective.alpha) else 0.0,
            "before_mean": float(b.mean()),
            "after_mean": float(a.mean()),
            "significant": bool(d.hi < 0),
        }
    pb = np.array([r.rset_exceeds_aset for r in before.runs], dtype=float)
    pa = np.array([r.rset_exceeds_aset for r in after.runs], dtype=float)
    rng = np.random.default_rng(before.config.seed)
    idx = rng.integers(0, pb.size, size=(1000, pb.size))
    diffs = pa[idx].mean(axis=1) - pb[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    out["p_rset_exceeds_aset"] = {
        "before": float(pb.mean()),
        "after": float(pa.mean()),
        "delta": asdict(Estimate(float(pa.mean() - pb.mean()), float(lo), float(hi))),
    }
    target = out["losses"].get(objective.loss, {})
    if objective.kind == "p_rset":
        out["significant"] = bool(hi < 0)
    else:
        out["significant"] = bool(target.get("significant", False))
    return out
