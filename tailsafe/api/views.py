"""Compact JSON views of results for the web frontend."""

from __future__ import annotations

from typing import Any

import numpy as np

from tailsafe import DISCLAIMER
from tailsafe.building.graph import summary as building_summary
from tailsafe.building.model import Building, NodeType
from tailsafe.hazard.model import HazardResult
from tailsafe.risk.breakdown import tail_breakdown
from tailsafe.scenarios.montecarlo import LOSSES, MCResult
from tailsafe.sim.meso import MesoResult
from tailsafe.sim.network import ARC_STAIR_DOWN

_CIRC = {NodeType.CORRIDOR, NodeType.LOBBY, NodeType.LIFT_LOBBY, NodeType.PROTECTED_LOBBY}


def building_view(b: Building) -> dict[str, Any]:
    """The building (schema JSON) plus headline facts."""
    return {
        "id": b.digest()[:16],
        "summary": building_summary(b),
        "building": b.model_dump(mode="json", exclude_none=True),
    }


def stress_view(res: MCResult) -> dict[str, Any]:
    """Distributions, tail metrics, tenability, breakdowns and stair congestion."""
    b = res.building
    losses = {name: [float(x) for x in res.loss(name)] for name in LOSSES}
    congestion: dict[str, dict[int, float]] = {}
    if b is not None:
        stairs = {s.id for s in b.stairs}
        edges = b.edge_by_id
        x = res.loss("p95_occupant_time")
        tail = x >= np.quantile(x, 0.95)
        qint = res.arc_qint
        tail_mean = qint[tail].mean(axis=0) if tail.any() else qint.mean(axis=0)
        for a, eid in enumerate(res.arc_edge_ids):
            e = edges[eid]
            if e.stair in stairs and e.kind == "stair":
                lv = b.node_by_id[e.source].level
                congestion.setdefault(e.stair, {})
                congestion[e.stair][lv] = congestion[e.stair].get(lv, 0.0) + float(tail_mean[a])
    floors = res.floor_exceedance()
    order = np.argsort(-res.loss("total_time"))
    return {
        "runs": res.n,
        "seed": res.config.seed,
        "elapsed_s": res.elapsed,
        "spec": res.spec.model_dump(mode="json"),
        "risk": {name: res.risk(name).as_dict() for name in LOSSES},
        "losses": losses,
        "occupants_mean": float(res.array("n_agents").mean()) if res.n else 0.0,
        "rescued_mean": float(res.array("n_rescued").mean()) if res.n else 0.0,
        "tenability": res.tenability_summary(),
        "floor_exceedance": [
            {"level": lv, "p": est.value, "lo": est.lo, "hi": est.hi} for lv, est in floors.items()
        ],
        "breakdown": {
            name: tail_breakdown(res, name) for name in ("total_time", "self_evacuation_time")
        }
        if res.runs and res.runs[0].groups is not None
        else {},
        "stair_congestion": {
            s: [{"level": lv, "person_seconds": v} for lv, v in sorted(d.items())]
            for s, d in congestion.items()
        },
        "worst_scenarios": [int(res.runs[int(i)].index) for i in order[:10]],
        "median_scenario": int(res.runs[int(order[len(order) // 2])].index) if res.n else 0,
        "disclaimer": DISCLAIMER,
    }


def replay_view(res: MesoResult, info: dict[str, Any], every: float = 20.0) -> dict[str, Any]:
    """Time series of one scenario for the 3D stack: queues, smoke, people per floor."""
    net = res.net
    b = net.building
    levels = [lv.index for lv in b.levels]
    assert res.series is not None and res.series_times is not None
    rec_dt = float(res.series_times[1] - res.series_times[0]) if res.series_times.size > 1 else 10.0
    finite_exit = res.group_exit[np.isfinite(res.group_exit)]
    t_end = float(finite_exit.max()) if finite_exit.size else float(res.series_times[-1])
    # Frames cover the whole evacuation, including rescues after the kernel stopped
    # stepping (queues are empty then); at most ~720 frames.
    every = max(every, t_end / 720.0)
    times = np.arange(0.0, t_end + every, every)
    rows = np.round(times / rec_dt).astype(int)
    recorded = rows < res.series.shape[1]
    t_idx = np.where(recorded, rows, 0)
    stair_queues: dict[str, list[list[float]]] = {}
    for s_i, sid in enumerate(net.stair_ids):
        flights = np.flatnonzero((net.arc_stair == s_i) & (net.arc_kind == ARC_STAIR_DOWN))
        grid = np.zeros((times.size, len(levels)))
        for a in flights:
            lv = int(net.node_level[net.arc_src[a]])
            grid[:, levels.index(lv)] += np.where(recorded, res.series[a, t_idx], 0.0)
        stair_queues[sid] = np.round(grid, 1).tolist()

    pop = res.population
    left = np.where(res.group_left_floor >= 0, res.group_left_floor, np.inf)
    remaining = np.zeros((times.size, len(levels)))
    for lv, t_left, size in zip(pop.group_level, left, pop.group_size, strict=True):
        remaining[:, levels.index(int(lv))] += (times < t_left) * size
    out_curve = [int(pop.group_size[res.group_exit <= t].sum()) for t in times]

    visibility = None
    hz = res.scenario.hazard
    if isinstance(hz, HazardResult) and hz.visibility is not None:
        circ = [i for i, n in enumerate(b.nodes) if n.type in _CIRC]
        hz_rows = np.minimum(np.round(times / hz.dt).astype(int), hz.times.size - 1)
        vis = np.full((times.size, len(levels)), 100.0)
        for i in circ:
            k = levels.index(b.nodes[i].level)
            vis[:, k] = np.minimum(vis[:, k], hz.visibility[hz_rows, i])
        visibility = np.round(vis, 1).tolist()
    return {
        "times": times.tolist(),
        "levels": levels,
        "stair_queues": stair_queues,
        "remaining": remaining.astype(int).tolist(),
        "evacuated": out_curve,
        "visibility": visibility,
        "info": info,
        "summary": res.summary(),
        "disclaimer": DISCLAIMER,
    }
