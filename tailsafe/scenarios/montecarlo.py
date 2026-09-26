"""Monte Carlo runner: many sampled scenarios through the meso engine, in parallel.

Scenario ``i`` of a run with seed ``s`` is fully determined by ``(s, i)`` and
the batch layout, whichever worker computes it; results are identical for any
number of workers. Runs proceed in batches; with ``target_halfwidth`` set, the
runner stops once the bootstrap CI of CVaR₉₅ is narrow enough (after
``min_runs``).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tailsafe.building.model import Building
from tailsafe.config import Params, get_params
from tailsafe.risk.metrics import Estimate, RiskSummary, cvar_halfwidth, summarize
from tailsafe.risk.tenability import rset_aset, wilson_interval
from tailsafe.scenarios.sampler import ScenarioSampler, scenario_uniforms
from tailsafe.scenarios.spec import ScenarioSpec
from tailsafe.sim.meso import MesoResult, SimConfig, run_meso
from tailsafe.sim.network import SimNetwork, compile_network
from tailsafe.sim.routing import Router

LOSSES = ("total_time", "self_evacuation_time", "p95_occupant_time")
_GROUP_FIELDS = ("level", "profile", "size", "exit", "left", "rescued", "fed")

Progress = Callable[[int, int], None]


@dataclass(frozen=True)
class MCConfig:
    """How many scenarios to run and how."""

    n_runs: int = 1000
    seed: int = 0
    lhs: bool = True
    batch_size: int = 100
    workers: int | None = None
    chunk_size: int = 10
    target_halfwidth: float | None = None
    min_runs: int = 200
    alpha: float = 0.95
    confidence: float = 0.95
    n_boot: int = 1000
    keep_groups: bool = True
    sim: SimConfig = field(default_factory=SimConfig)

    def resolved_workers(self) -> int:
        """Worker processes to use (``None`` = all CPUs)."""
        return max(1, self.workers if self.workers is not None else (os.cpu_count() or 1))


@dataclass
class RunOutput:
    """Compact, picklable result of one scenario."""

    index: int
    total_time: float
    self_evacuation_time: float
    p50_occupant_time: float
    p95_occupant_time: float
    n_agents: int
    n_rescued: int
    n_not_evacuated: int
    lift_trips: int
    forced_entries: int
    steps: int
    info: dict[str, Any]
    groups: dict[str, NDArray[Any]] | None
    arc_maxq: NDArray[np.float32]
    arc_qint: NDArray[np.float32]
    rset_exceeds_aset: bool = False
    n_over_fed_limit: int = 0
    n_incapacitated: int = 0
    max_fed: float = 0.0
    floor_levels: list[int] = field(default_factory=list)
    floor_rset: list[float] = field(default_factory=list)
    floor_aset: list[float] = field(default_factory=list)


def compress(
    index: int,
    res: MesoResult,
    info: dict[str, Any],
    keep_groups: bool,
    params: Params | None = None,
) -> RunOutput:
    """Reduce a :class:`MesoResult` to what the Monte Carlo analysis needs."""
    pop = res.population
    ra = rset_aset(res, params)
    groups: dict[str, NDArray[Any]] | None = None
    if keep_groups:
        groups = {
            "level": pop.group_level.astype(np.int16),
            "profile": pop.group_key_profile.astype(np.int8),
            "size": pop.group_size.astype(np.int16),
            "exit": res.group_exit.astype(np.float32),
            "left": res.group_left_floor.astype(np.float32),
            "rescued": res.group_rescued.astype(bool),
            "fed": res.group_fed.astype(np.float32),
        }
    return RunOutput(
        index=index,
        total_time=res.total_time,
        self_evacuation_time=res.self_evacuation_time,
        p50_occupant_time=res.exit_time_quantile(0.5),
        p95_occupant_time=res.exit_time_quantile(0.95),
        n_agents=pop.n_agents,
        n_rescued=int(pop.group_size[res.group_rescued].sum()),
        n_not_evacuated=res.n_not_evacuated,
        lift_trips=res.lift_trips,
        forced_entries=res.forced_entries,
        steps=res.steps,
        info=info,
        groups=groups,
        arc_maxq=res.arc_max_queue.astype(np.float32),
        arc_qint=res.arc_queue_integral.astype(np.float32),
        rset_exceeds_aset=ra.fails,
        n_over_fed_limit=ra.occupants_over_fed_limit,
        n_incapacitated=ra.occupants_incapacitated,
        max_fed=ra.max_fed,
        floor_levels=[int(x) for x in ra.floor_levels],
        floor_rset=[float(x) for x in ra.floor_rset],
        floor_aset=[float(x) for x in ra.floor_aset],
    )


# ============================================================================ workers
_WORKER: dict[str, Any] = {}


def _init_worker(
    building_json: str,
    spec_json: str,
    params_tree: dict[str, Any],
    sim_config: SimConfig,
    keep_groups: bool,
) -> None:
    params = Params(params_tree)
    building = Building.model_validate_json(building_json)
    net = compile_network(building, params)
    _WORKER.clear()
    _WORKER.update(
        params=params,
        net=net,
        router=Router(net, params),
        sampler=ScenarioSampler(building, ScenarioSpec.model_validate_json(spec_json), params),
        sim_config=sim_config,
        keep_groups=keep_groups,
    )


def _simulate_chunk(
    seed: int, indices: list[int], uniforms: NDArray[np.float64]
) -> list[RunOutput]:
    w = _WORKER
    out = []
    for i, u in zip(indices, uniforms, strict=True):
        sc = w["sampler"].sample(seed, i, u)
        res = run_meso(
            w["net"], sc.population, sc.sim, w["sim_config"], params=w["params"], router=w["router"]
        )
        out.append(compress(i, res, sc.info, w["keep_groups"], w["params"]))
    return out


def _warm_up() -> None:
    """Compile (or load) the Numba kernel before forking workers."""
    from tailsafe.sim.cases import stair_tower, uniform_population

    b = stair_tower(floors=1)
    run_meso(b, uniform_population(b, 1))


# ============================================================================ result
@dataclass
class MCResult:
    """All scenarios of a Monte Carlo stress test."""

    building_id: str
    building_name: str
    building_digest: str
    params_digest: str
    spec: ScenarioSpec
    config: MCConfig
    runs: list[RunOutput]
    arc_labels: list[str]
    arc_edge_ids: list[str]
    elapsed: float
    converged: bool | None = None

    @property
    def n(self) -> int:
        """Number of scenarios."""
        return len(self.runs)

    def array(self, name: str) -> NDArray[np.float64]:
        """Per-scenario values of a scalar field (e.g. ``total_time``)."""
        return np.array([getattr(r, name) for r in self.runs], dtype=np.float64)

    def loss(self, name: str = "total_time") -> NDArray[np.float64]:
        """Per-scenario loss (one of :data:`LOSSES`)."""
        if name not in LOSSES:
            raise ValueError(f"unknown loss {name!r}; choose from {LOSSES}")
        return self.array(name)

    def risk(self, name: str = "total_time", alpha: float | None = None) -> RiskSummary:
        """Tail statistics of a loss with bootstrap confidence intervals."""
        return summarize(
            self.loss(name),
            alpha=alpha or self.config.alpha,
            confidence=self.config.confidence,
            n_boot=self.config.n_boot,
            seed=self.config.seed,
        )

    @property
    def arc_maxq(self) -> NDArray[np.float32]:
        """Max queue per scenario and arc ``[n, M]``."""
        return np.stack([r.arc_maxq for r in self.runs])

    @property
    def arc_qint(self) -> NDArray[np.float32]:
        """Time-integrated queue (person-seconds) per scenario and arc ``[n, M]``."""
        return np.stack([r.arc_qint for r in self.runs])

    # ------------------------------------------------------------------ tenability
    def p_rset_exceeds_aset(self) -> Estimate:
        """P(RSET > ASET) across scenarios with a Wilson 95% interval."""
        k = sum(r.rset_exceeds_aset for r in self.runs)
        lo, hi = wilson_interval(k, self.n)
        return Estimate(k / self.n if self.n else 0.0, lo, hi)

    def floor_exceedance(self) -> dict[int, Estimate]:
        """Per floor: P(RSET_floor > ASET_floor) with Wilson intervals."""
        hits: dict[int, int] = {}
        seen: dict[int, int] = {}
        for r in self.runs:
            for lv, rs, a in zip(r.floor_levels, r.floor_rset, r.floor_aset, strict=True):
                seen[lv] = seen.get(lv, 0) + 1
                hits[lv] = hits.get(lv, 0) + int(rs > a)
        out = {}
        for lv in sorted(seen):
            lo, hi = wilson_interval(hits[lv], seen[lv])
            out[lv] = Estimate(hits[lv] / seen[lv], lo, hi)
        return out

    def tenability_summary(self) -> dict[str, Any]:
        """P(RSET > ASET), exposure and incapacitation statistics."""
        p = self.p_rset_exceeds_aset()
        inc = self.array("n_incapacitated")
        over = self.array("n_over_fed_limit")
        any_inc = int((inc > 0).sum())
        lo, hi = wilson_interval(any_inc, self.n)
        floors = self.floor_exceedance()
        worst = sorted(floors.items(), key=lambda kv: -kv[1].value)[:5]
        return {
            "p_rset_exceeds_aset": asdict(p),
            "p_any_incapacitated": asdict(Estimate(any_inc / max(self.n, 1), lo, hi)),
            "mean_incapacitated": float(inc.mean()) if self.n else 0.0,
            "mean_over_fed_limit": float(over.mean()) if self.n else 0.0,
            "cvar_incapacitated": float(np.sort(inc)[-max(1, self.n // 20) :].mean())
            if self.n
            else 0.0,
            "worst_floors": [{"level": lv, **asdict(est)} for lv, est in worst if est.value > 0],
        }

    def summary(self) -> dict[str, Any]:
        """Headline numbers (JSON-friendly)."""
        return {
            "building": {"id": self.building_id, "name": self.building_name},
            "scenario": self.spec.model_dump(mode="json"),
            "runs": self.n,
            "seed": self.config.seed,
            "elapsed_s": round(self.elapsed, 2),
            "converged": self.converged,
            "occupants_mean": float(self.array("n_agents").mean()) if self.runs else 0.0,
            "rescued_occupants_mean": float(self.array("n_rescued").mean()) if self.runs else 0.0,
            "risk": {name: self.risk(name).as_dict() for name in LOSSES},
            "tenability": self.tenability_summary() if self.runs else {},
        }

    # ------------------------------------------------------------------ persistence
    def save(self, directory: Path) -> Path:
        """Write ``result.json`` (metadata, per-run scalars) and ``arrays.npz``."""
        directory.mkdir(parents=True, exist_ok=True)
        scalars = {
            f: [getattr(r, f) for r in self.runs]
            for f in RunOutput.__dataclass_fields__
            if f not in ("info", "groups", "arc_maxq", "arc_qint")
        }
        meta = {
            "building_id": self.building_id,
            "building_name": self.building_name,
            "building_digest": self.building_digest,
            "params_digest": self.params_digest,
            "spec": self.spec.model_dump(mode="json"),
            "config": {**asdict(self.config), "sim": asdict(self.config.sim)},
            "elapsed": self.elapsed,
            "converged": self.converged,
            "arc_labels": self.arc_labels,
            "arc_edge_ids": self.arc_edge_ids,
            "scalars": scalars,
            "info": [r.info for r in self.runs],
        }
        (directory / "result.json").write_text(
            json.dumps(meta, default=_json_default) + "\n", encoding="utf-8"
        )
        arrays: dict[str, NDArray[Any]] = {
            "arc_maxq": self.arc_maxq,
            "arc_qint": self.arc_qint,
        }
        if self.runs and self.runs[0].groups is not None:
            sizes = np.array([len(r.groups["exit"]) for r in self.runs if r.groups is not None])
            arrays["group_offsets"] = np.concatenate([[0], np.cumsum(sizes)])
            for f in _GROUP_FIELDS:
                arrays[f"group_{f}"] = np.concatenate(
                    [r.groups[f] for r in self.runs if r.groups is not None]
                )
        savez: Any = np.savez_compressed
        savez(directory / "arrays.npz", **arrays)
        return directory

    @classmethod
    def load(cls, directory: Path) -> MCResult:
        """Read a result written by :meth:`save`."""
        meta = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        with np.load(directory / "arrays.npz") as npz:
            # Read each array once: indexing an NpzFile decompresses on every access.
            data = {k: npz[k] for k in npz.files}
        cfg = dict(meta["config"])
        cfg["sim"] = SimConfig(**cfg["sim"])
        scalars = meta["scalars"]
        n = len(scalars["index"])
        runs = []
        for k in range(n):
            groups = None
            if "group_offsets" in data:
                a, b = int(data["group_offsets"][k]), int(data["group_offsets"][k + 1])
                groups = {f: data[f"group_{f}"][a:b] for f in _GROUP_FIELDS}
            runs.append(
                RunOutput(
                    **{f: scalars[f][k] for f in scalars},
                    info=meta["info"][k],
                    groups=groups,
                    arc_maxq=data["arc_maxq"][k],
                    arc_qint=data["arc_qint"][k],
                )
            )
        return cls(
            building_id=meta["building_id"],
            building_name=meta["building_name"],
            building_digest=meta["building_digest"],
            params_digest=meta["params_digest"],
            spec=ScenarioSpec.model_validate(meta["spec"]),
            config=MCConfig(**cfg),
            runs=runs,
            arc_labels=meta["arc_labels"],
            arc_edge_ids=meta["arc_edge_ids"],
            elapsed=meta["elapsed"],
            converged=meta["converged"],
        )


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(f"not JSON serialisable: {type(o)}")


# ============================================================================ driver
def run_monte_carlo(
    building: Building,
    spec: ScenarioSpec,
    config: MCConfig | None = None,
    *,
    params: Params | None = None,
    progress: Progress | None = None,
) -> MCResult:
    """Run the stress test and return every scenario's outcome."""
    cfg = config or MCConfig()
    p = params or get_params()
    ScenarioSampler(building, spec, p)  # validate the spec against the building early
    net: SimNetwork = compile_network(building, p)
    init_args = (
        building.model_dump_json(),
        spec.model_dump_json(),
        p.as_dict(),
        cfg.sim,
        cfg.keep_groups,
    )
    workers = cfg.resolved_workers()
    t0 = time.perf_counter()
    runs: list[RunOutput] = []
    converged: bool | None = None if cfg.target_halfwidth is None else False

    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        _warm_up()
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context(method),
            initializer=_init_worker,
            initargs=init_args,
        )
    else:
        _init_worker(*init_args)
    try:
        done = 0
        while done < cfg.n_runs:
            n = min(cfg.batch_size, cfg.n_runs - done)
            U = scenario_uniforms(cfg.seed, done, n, lhs=cfg.lhs, batch_size=cfg.batch_size)
            idx = list(range(done, done + n))
            chunks = [
                (idx[k : k + cfg.chunk_size], U[k : k + cfg.chunk_size])
                for k in range(0, n, cfg.chunk_size)
            ]
            batch: list[RunOutput] = []
            if executor is not None:
                futures: list[Future[list[RunOutput]]] = [
                    executor.submit(_simulate_chunk, cfg.seed, ci, cu) for ci, cu in chunks
                ]
                for fut in futures:
                    batch.extend(fut.result())
                    if progress:
                        progress(done + len(batch), cfg.n_runs)
            else:
                for ci, cu in chunks:
                    batch.extend(_simulate_chunk(cfg.seed, ci, cu))
                    if progress:
                        progress(done + len(batch), cfg.n_runs)
            runs.extend(sorted(batch, key=lambda r: r.index))
            done += n
            if cfg.target_halfwidth is not None and done >= cfg.min_runs:
                losses = np.array([r.total_time for r in runs])
                hw = cvar_halfwidth(losses, alpha=cfg.alpha, seed=cfg.seed)
                if hw <= cfg.target_halfwidth:
                    converged = True
                    break
    finally:
        if executor is not None:
            executor.shutdown()
    return MCResult(
        building_id=building.id,
        building_name=building.name,
        building_digest=building.digest(),
        params_digest=p.digest,
        spec=spec,
        config=cfg,
        runs=runs,
        arc_labels=[net.arc_label(a) for a in range(net.n_arcs)],
        arc_edge_ids=[net.edge_ids[int(e)] for e in net.arc_edge],
        elapsed=time.perf_counter() - t0,
        converged=converged,
    )
