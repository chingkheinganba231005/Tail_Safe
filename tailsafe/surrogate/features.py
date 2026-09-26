"""Graph features for the surrogate: the building's circulation network.

Flats are folded into the corridor or lobby they open onto (as counts and
expected occupants), leaving the circulation network — corridors, lobbies,
landings, refuges, exits — as graph nodes and its walkways, doors and stair
flights as edges (both directions). Scenario settings enter as node and edge
flags (fire floor, wardens, blocked stairs) and as global features. Nothing
here uses simulation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tailsafe.building.model import Building, EdgeKind, LevelKind, NodeType
from tailsafe.config import Params, get_params
from tailsafe.scenarios.spec import ScenarioSpec
from tailsafe.sim.network import ARC_STAIR_DOWN, compile_network

NODE_TYPES = (
    NodeType.CORRIDOR,
    NodeType.LOBBY,
    NodeType.LIFT_LOBBY,
    NodeType.PROTECTED_LOBBY,
    NodeType.STAIR_LANDING,
    NodeType.REFUGE,
    NodeType.OPEN_AREA,
    NodeType.EXIT,
)
N_NODE = len(NODE_TYPES) + 13
N_EDGE = 11
N_GLOBAL = 24
SLOTS = ("weekday_day", "weekday_night", "weekend_day", "weekend_night")
PRIORITY = ("top_down", "nearest", "bottom_up")


@dataclass
class StaticGraph:
    """Building part of the features (reused across scenarios)."""

    node_ids: list[str]
    node_level: NDArray[np.int32]
    node_base: NDArray[np.float32]  # [N, N_NODE] with scenario columns zero
    edge_src: NDArray[np.int32]  # directed, both ways
    edge_dst: NDArray[np.int32]
    edge_base: NDArray[np.float32]
    edge_ids: list[str]  # physical edge per directed edge
    edge_forward: NDArray[np.bool_]  # the copy in the edge's own orientation
    edge_stair: list[str | None]
    landing_stair: list[str | None]
    lift_lobby: NDArray[np.bool_]
    n_levels: int
    n_units: int
    occupants: float
    care_share: float
    n_lifts: int
    n_ff_lifts: int
    n_stairs: int
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphData:
    """Everything the model needs for one (building, scenario) pair."""

    node_x: NDArray[np.float32]
    edge_src: NDArray[np.int32]
    edge_dst: NDArray[np.int32]
    edge_x: NDArray[np.float32]
    global_x: NDArray[np.float32]
    edge_ids: list[str]
    edge_forward: NDArray[np.bool_]

    @property
    def n_nodes(self) -> int:
        """Number of circulation nodes."""
        return int(self.node_x.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of directed edges."""
        return int(self.edge_src.size)


def _household_means(p: Params) -> dict[str, float]:
    out = {}
    for key in ("flat_small", "flat_medium", "flat_large", "care_room"):
        outcomes, probs = p[f"population.household_size.{key}"].categories
        out[key] = float(np.dot(np.asarray(outcomes, dtype=np.float64), probs))
    return out


def static_graph(building: Building, params: Params | None = None) -> StaticGraph:
    """Circulation graph with building features (no scenario)."""
    p = params or get_params()
    b = building
    circ = [n for n in b.nodes if n.type != NodeType.UNIT]
    index = {n.id: i for i, n in enumerate(circ)}
    n_levels = len(b.levels)
    hh = _household_means(p)
    units = np.zeros(len(circ))
    occ = np.zeros(len(circ))
    by = b.node_by_id
    n_units = 0
    care = 0
    total_occ = 0.0
    for e in b.edges:
        s, t = by[e.source], by[e.target]
        for u, c in ((s, t), (t, s)):
            if u.type == NodeType.UNIT and c.type != NodeType.UNIT:
                k = index[c.id]
                units[k] += 1
                expected = hh.get(u.unit_type or "flat_medium", hh["flat_medium"])
                occ[k] += expected
    for n in b.nodes:
        if n.type == NodeType.UNIT:
            n_units += 1
            care += n.unit_type == "care_room"
            total_occ += hh.get(n.unit_type or "flat_medium", hh["flat_medium"])
    refuge_levels = {lv.index for lv in b.levels if lv.kind == LevelKind.REFUGE}
    lift_nodes = {st.node for lf in b.lifts if not lf.firefighting for st in lf.stops}

    nx = np.zeros((len(circ), N_NODE), dtype=np.float32)
    for i, n in enumerate(circ):
        nx[i, NODE_TYPES.index(n.type) if n.type in NODE_TYPES else 0] = 1.0
        o = len(NODE_TYPES)
        nx[i, o + 0] = n.level / 50.0
        nx[i, o + 1] = float(n.level == 0)
        nx[i, o + 2] = np.log1p(n.area or 0.0) / 5.0
        nx[i, o + 3] = units[i] / 10.0
        nx[i, o + 4] = occ[i] / 20.0
        nx[i, o + 5] = n.level / max(n_levels - 1, 1)
        nx[i, o + 6] = float(n.level in refuge_levels)
        # o+7 fire floor, o+8 warden, o+9 phased delay, o+10 blocked, o+11 block time,
        # o+12 evacuation-lift lobby: scenario columns filled in `graph_features`

    net = compile_network(b, p)
    arc_of: dict[tuple[str, str], int] = {}
    for a in range(net.n_arcs):
        arc_of[(net.node_ids[int(net.arc_src[a])], net.node_ids[int(net.arc_dst[a])])] = a
    src: list[int] = []
    dst: list[int] = []
    feats: list[list[float]] = []
    eids: list[str] = []
    fwd: list[bool] = []
    estair: list[str | None] = []
    for e in b.edges:
        if e.source not in index or e.target not in index:
            continue
        for from_id, to_id, forward in ((e.source, e.target, True), (e.target, e.source, False)):
            arc = arc_of.get((from_id, to_id))
            cap = float(net.arc_cap[arc]) if arc is not None else 0.0
            down = 0.0
            if e.kind == EdgeKind.STAIR:
                down = 1.0 if arc is not None and net.arc_kind[arc] == ARC_STAIR_DOWN else -1.0
            f = [
                float(e.kind == EdgeKind.FLAT),
                float(e.kind == EdgeKind.DOOR),
                float(e.kind == EdgeKind.STAIR),
                down,
                e.width,
                e.length / 10.0,
                cap,
                float(e.fire_rated),
                float(e.self_closing),
                0.0,  # blocked (scenario)
                0.0,  # blockage time (scenario)
            ]
            src.append(index[from_id])
            dst.append(index[to_id])
            feats.append(f)
            eids.append(e.id)
            fwd.append(forward)
            estair.append(e.stair if e.kind == EdgeKind.STAIR else None)
    return StaticGraph(
        node_ids=[n.id for n in circ],
        node_level=np.array([n.level for n in circ], dtype=np.int32),
        node_base=nx,
        edge_src=np.array(src, dtype=np.int32),
        edge_dst=np.array(dst, dtype=np.int32),
        edge_base=np.array(feats, dtype=np.float32).reshape(-1, N_EDGE),
        edge_ids=eids,
        edge_forward=np.array(fwd, dtype=bool),
        edge_stair=estair,
        landing_stair=[n.stair if n.type == NodeType.STAIR_LANDING else None for n in circ],
        lift_lobby=np.array([n.id in lift_nodes for n in circ], dtype=bool),
        n_levels=n_levels,
        n_units=n_units,
        occupants=total_occ,
        care_share=care / max(n_units, 1),
        n_lifts=sum(not lf.firefighting for lf in b.lifts),
        n_ff_lifts=sum(lf.firefighting for lf in b.lifts),
        n_stairs=len(b.stairs),
    )


def graph_features(g: StaticGraph, spec: ScenarioSpec, params: Params | None = None) -> GraphData:
    """Add the scenario to a static graph."""
    p = params or get_params()
    o = len(NODE_TYPES)
    nx = g.node_base.copy()
    lv = g.node_level
    if spec.fire_level is not None:
        nx[:, o + 7] = (lv == spec.fire_level).astype(np.float32)
    if spec.warden_levels:
        nx[:, o + 8] = np.isin(lv, spec.warden_levels).astype(np.float32)
    for level, delay in spec.phased_release.items():
        nx[lv == int(level), o + 9] = float(delay) / 600.0
    blocked = {
        bk.stair: float(bk.time.value or 0.0)
        for bk in spec.stair_blockages
        if bk.time.value is not None
    }
    for bk in spec.stair_blockages:
        if bk.stair not in blocked:  # a random time: use its median
            blocked[bk.stair] = float(bk.time.ppf([0.5])[0])
    for i, sid in enumerate(g.landing_stair):
        if sid is not None and sid in blocked:
            nx[i, o + 10] = 1.0
            nx[i, o + 11] = blocked[sid] / 900.0
    if spec.evacuation_lifts:
        nx[:, o + 12] = g.lift_lobby.astype(np.float32)
    ex = g.edge_base.copy()
    for k, sid in enumerate(g.edge_stair):
        if sid is not None and sid in blocked:
            ex[k, 9] = 1.0
            ex[k, 10] = blocked[sid] / 900.0

    gx = np.zeros(N_GLOBAL, dtype=np.float32)
    gx[SLOTS.index(spec.time_slot.value)] = 1.0
    mix = p["population.age_mix.residential"].categories
    older = float(sum(pr for k, pr in zip(mix[0], mix[1], strict=True) if k in ("older", "frail")))
    gx[4] = spec.share_65_plus if spec.share_65_plus is not None else older
    gx[5] = (
        spec.share_80_plus_of_65_plus
        if spec.share_80_plus_of_65_plus is not None
        else p.scalar("population.frail_share_of_65_plus")
    )
    gx[6] = (
        spec.vacancy_rate if spec.vacancy_rate is not None else p.scalar("population.vacancy_rate")
    )
    gx[7] = g.n_levels / 50.0
    gx[8] = g.n_units / 500.0
    gx[9] = g.occupants / 2000.0
    gx[10] = g.care_share
    gx[11] = g.n_lifts / 3.0
    gx[12] = spec.lifts_out_of_service / max(g.n_lifts + g.n_ff_lifts, 1)
    gx[13] = float(spec.evacuation_lifts)
    gx[14 + PRIORITY.index(spec.lift_priority)] = float(spec.evacuation_lifts)
    gx[17] = float(spec.evacuation_lifts and spec.lift_eligibility == "wheelchair_users")
    gx[18] = float(spec.hazard is not None and spec.hazard.enabled)
    gx[19] = (spec.fire_level or 0) / 50.0
    gx[20] = len(spec.warden_levels) / 5.0
    gx[21] = g.n_stairs / 2.0
    gx[22] = len(blocked) / max(g.n_stairs, 1)
    gx[23] = min(blocked.values()) / 900.0 if blocked else 0.0
    return GraphData(
        node_x=nx,
        edge_src=g.edge_src,
        edge_dst=g.edge_dst,
        edge_x=ex,
        global_x=gx,
        edge_ids=g.edge_ids,
        edge_forward=g.edge_forward,
    )
