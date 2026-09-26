"""Mesoscopic queue-network evacuation simulator.

Model
-----
The building is a network of directed arcs (see :mod:`tailsafe.sim.network`).
Households move as groups at the pace of their slowest member.

* **Walking.** A group on an arc advances at its profile speed for the arc type
  (horizontal, stair down with fatigue, stair up) times a density factor from
  the hydraulic speed–density relation ``(1 - aD) / (1 - aD0)`` for ``D > D0``.
  ``D`` counts the people *walking* on the passage in both directions
  (counter-flow slows everyone); people standing in the end-of-arc queue are
  excluded because their effect is already the queue itself. ``D`` is capped at
  the flow-maximising density ``1 / (2a)`` (≈ 1.88 p/m²): denser conditions are
  queueing, which the queue already represents (otherwise slow walkers would
  trigger a runaway density–speed collapse). The hazard speed multiplier applies
  when a hazard field is given.
* **Queues and capacity.** Arriving at the end of an arc, a group joins a FIFO
  queue and enters the next arc when that arc has inflow capacity left this
  time step (capacity = max specific flow × effective width; groups consume
  capacity equal to their size) and storage room (jam density × area). Groups
  that are blocked by storage for longer than ``stuck_time`` squeeze in anyway,
  which prevents gridlock in counter-flow (counted as *forced entries*).
  Counter-flow also reduces an arc's capacity by ``capacity_penalty`` scaled by
  the opposite direction's share of occupants.
* **Merging.** Where several queues compete for the same arc, a smooth weighted
  round-robin shares capacity. At stair landings people entering from the floor
  get weight ``floor_deference_ratio`` and people already in the stair the rest;
  unused share passes to the other stream.
* **Routing.** Next-hop tables by estimated travel time, a logit choice of
  staircase per group (or an assigned staircase per floor), waypoints for
  counter-flow visits, refuge-floor rest stops and lift lobbies. Blockages are
  discovered when a group reaches the blocked arc.
* **Lifts** in evacuation service fetch waiting groups (top-down by default)
  and carry them to the discharge level. **Rescue**: households that cannot
  self-evacuate wait for fire-service teams, which serve the lowest floors first.
* **Hazard coupling** (optional): per-node speed multipliers and FED rates on a
  time grid; groups accumulate FED and are incapacitated at the threshold.

Times are in seconds from the alarm. Everything is deterministic given the
inputs; all randomness lives in the population and scenario samplers.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from tailsafe.building.model import Building, EdgeKind, NodeType
from tailsafe.config import Params, get_params
from tailsafe.population.synth import Mode, Population
from tailsafe.sim import _kernel as K
from tailsafe.sim.network import ARC_FLAT, SimNetwork, compile_network
from tailsafe.sim.routing import Router, choose_stair_class

LIFT_RULES = {"top_down": K.LIFT_TOP_DOWN, "nearest": K.LIFT_NEAREST, "bottom_up": K.LIFT_BOTTOM_UP}


class HazardField(Protocol):
    """Time-gridded hazard fields on nodes (produced by the hazard model, M4)."""

    dt: float
    speed_multiplier: NDArray[np.float32]  # [H, N]
    fed_rate: NDArray[np.float32]  # [H, N] per second


@dataclass(frozen=True)
class Blockage:
    """An edge that becomes impassable (both directions) from ``time`` seconds."""

    edge_id: str
    time: float = 0.0


def stair_blockage(building: Building, stair_id: str, time: float = 0.0) -> tuple[Blockage, ...]:
    """Block every flight of a staircase from ``time`` (e.g. a smoke-logged stair)."""
    if stair_id not in building.stair_by_id:
        raise ValueError(f"unknown stair {stair_id!r}; building has {sorted(building.stair_by_id)}")
    return tuple(
        Blockage(e.id, time)
        for e in building.edges
        if e.kind == EdgeKind.STAIR and e.stair == stair_id
    )


@dataclass(frozen=True)
class SimScenario:
    """Everything about one evacuation besides the building and the occupants."""

    blockages: tuple[Blockage, ...] = ()
    evacuation_lifts: tuple[str, ...] = ()
    lift_outages: tuple[tuple[str, float], ...] = ()
    lift_priority: str = "top_down"
    phased_release: Mapping[int, float] = field(default_factory=dict)
    stair_assignment: Mapping[int, str] = field(default_factory=dict)
    rescue_start: float | None = None
    rescue_teams: int | None = None
    hazard: HazardField | None = None
    held_open_doors: tuple[str, ...] = ()
    capacity_multipliers: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class SimConfig:
    """Numerical settings (not physical parameters)."""

    dt: float = 0.5
    t_max: float = 4 * 3600.0
    record_series: bool = False
    record_interval: float = 5.0
    stuck_time: float = 60.0
    speed_floor: float = 0.05
    fed_threshold: float | None = None  # default: hazard.tenability.fed_incapacitation


@dataclass
class MesoResult:
    """Outputs of one simulation run."""

    net: SimNetwork
    population: Population
    group_exit: NDArray[np.float64]
    group_left_floor: NDArray[np.float64]
    group_fed: NDArray[np.float64]
    group_rescued: NDArray[np.bool_]
    group_state: NDArray[np.int8]
    group_class: NDArray[np.int32]
    arc_max_queue: NDArray[np.float64]
    arc_queue_integral: NDArray[np.float64]
    arc_entries: NDArray[np.float64]
    series: NDArray[np.float32] | None
    series_times: NDArray[np.float64] | None
    steps: int
    forced_entries: int
    lift_trips: int
    wall_time: float
    config: SimConfig
    scenario: SimScenario = field(default_factory=lambda: SimScenario())

    # ------------------------------------------------------------------ agents
    @property
    def agent_exit(self) -> NDArray[np.float64]:
        """Exit time of each occupant (inf if not evacuated)."""
        out: NDArray[np.float64] = self.group_exit[self.population.agent_group]
        return out

    @property
    def agent_fed(self) -> NDArray[np.float64]:
        """Accumulated FED of each occupant."""
        out: NDArray[np.float64] = self.group_fed[self.population.agent_group]
        return out

    # ------------------------------------------------------------------ headline
    @property
    def evacuated(self) -> NDArray[np.bool_]:
        """Groups that reached an exit (on foot, by lift or by rescue)."""
        out: NDArray[np.bool_] = np.isfinite(self.group_exit)
        return out

    @property
    def n_not_evacuated(self) -> int:
        """Occupants not out of the building by ``t_max`` (or incapacitated)."""
        return int(self.population.group_size[~self.evacuated].sum())

    @property
    def incapacitated(self) -> NDArray[np.bool_]:
        """Groups incapacitated by smoke (FED reached the threshold)."""
        out: NDArray[np.bool_] = self.group_state == K.ST_INCAPACITATED
        return out

    @property
    def n_incapacitated(self) -> int:
        """Occupants incapacitated by smoke."""
        return int(self.population.group_size[self.incapacitated].sum())

    @property
    def total_time(self) -> float:
        """Time the last occupant who gets out is out (including rescue).

        Incapacitated occupants are excluded (see :attr:`n_incapacitated`);
        ``inf`` only if someone neither got out nor was incapacitated by ``t_max``.
        """
        if self.population.n_groups == 0:
            return 0.0
        pending = ~self.evacuated & ~self.incapacitated
        if pending.any():
            return float("inf")
        done = self.group_exit[self.evacuated]
        return float(done.max()) if done.size else 0.0

    @property
    def self_evacuation_time(self) -> float:
        """Time the last occupant who left without fire-service rescue is out."""
        mask = ~self.group_rescued & ~self.incapacitated
        if not mask.any():
            return 0.0
        return float(self.group_exit[mask].max())

    def exit_time_quantile(self, q: float) -> float:
        """Occupant-weighted quantile of exit times (``inf`` counts as not out)."""
        t = np.sort(self.agent_exit)
        if t.size == 0:
            return 0.0
        return float(t[min(int(np.ceil(q * t.size)) - 1, t.size - 1)] if q > 0 else t[0])

    def floor_clearance(self) -> dict[int, float]:
        """Time the last household of each floor left that floor."""
        out: dict[int, float] = {}
        for lv, t in zip(self.population.group_level, self.group_left_floor, strict=True):
            val = float(t) if t >= 0 else np.inf
            out[int(lv)] = max(out.get(int(lv), 0.0), val)
        return dict(sorted(out.items()))

    def top_queues(self, n: int = 10) -> list[dict[str, Any]]:
        """Arcs with the largest time-integrated queue (person-seconds)."""
        order = np.argsort(-self.arc_queue_integral)[:n]
        return [
            {
                "arc": int(a),
                "label": self.net.arc_label(int(a)),
                "max_queue": float(self.arc_max_queue[a]),
                "queue_person_seconds": float(self.arc_queue_integral[a]),
            }
            for a in order
            if self.arc_queue_integral[a] > 0
        ]

    def summary(self) -> dict[str, Any]:
        """Headline numbers of the run."""
        return {
            "occupants": self.population.n_agents,
            "groups": self.population.n_groups,
            "total_time_s": self.total_time,
            "self_evacuation_time_s": self.self_evacuation_time,
            "p50_exit_s": self.exit_time_quantile(0.5),
            "p95_exit_s": self.exit_time_quantile(0.95),
            "rescued_occupants": int(self.population.group_size[self.group_rescued].sum()),
            "incapacitated_occupants": self.n_incapacitated,
            "max_fed": float(self.group_fed.max()) if self.group_fed.size else 0.0,
            "not_evacuated": self.n_not_evacuated,
            "lift_trips": self.lift_trips,
            "forced_entries": self.forced_entries,
            "steps": self.steps,
            "wall_time_s": self.wall_time,
        }


@dataclass
class _Prepared:
    args: tuple[Any, ...]
    group_class: NDArray[np.int32]
    n_rec: int
    # Decisions and inputs by name, shared with the micro engine so both
    # engines route, time and slow people identically.
    named: dict[str, Any]


def _waypoints(
    net: SimNetwork,
    pop: Population,
    scenario: SimScenario,
    group_class: NDArray[np.int32],
    home: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.float64], NDArray[np.int8], NDArray[np.int8], list[int]]:
    """Per-group waypoint nodes (counter-flow visit, refuge rest or lift lobby)."""
    G = pop.n_groups
    wp_node = np.full((G, 2), -1, dtype=np.int32)
    wp_dwell = np.zeros((G, 2))
    wp_kind = np.zeros((G, 2), dtype=np.int8)
    mode = pop.group_mode.copy()
    b = net.building

    lobby_at: dict[int, int] = {}
    evac = set(scenario.evacuation_lifts)
    for li, lid in enumerate(net.lifts.ids):
        if lid in evac:
            for lv, node in enumerate(net.lifts.stop_node[li]):
                if node >= 0:
                    lobby_at.setdefault(lv, int(node))

    refuge_nodes: dict[int, list[int]] = {}
    for i, t in enumerate(net.node_type):
        if t == NodeType.REFUGE:
            refuge_nodes.setdefault(int(net.node_level[i]), []).append(i)
    landing_xy: dict[tuple[int, int], tuple[float, float]] = {}
    for n in b.nodes_of_type(NodeType.STAIR_LANDING):
        if n.stair is not None:
            landing_xy[(net.stair_ids.index(n.stair), n.level)] = (n.x, n.y)

    for g in range(G):
        slot = 0
        if mode[g] == Mode.WAIT_LIFT:
            lobby = lobby_at.get(int(pop.group_level[g]))
            if lobby is None:
                mode[g] = Mode.WAIT_RESCUE
            else:
                wp_node[g, 0] = lobby
                wp_kind[g, 0] = 1
            continue
        target = pop.group_waypoint[g]
        if target is not None:
            wp_node[g, slot] = net.node(target)
            wp_dwell[g, slot] = pop.group_waypoint_dwell[g]
            slot += 1
        if pop.group_refuge_rest[g] > 0:
            below = [lv for lv in refuge_nodes if lv < pop.group_level[g]]
            if below:
                lv = max(below)
                stair = int(group_class[g]) - 1
                ref = landing_xy.get((stair, lv)) if stair >= 0 else None
                home_node = b.node_by_id[pop.group_unit[g]]
                ref = ref or (home_node.x, home_node.y)
                nodes = refuge_nodes[lv]
                pick = min(
                    nodes,
                    key=lambda i: (b.nodes[i].x - ref[0]) ** 2 + (b.nodes[i].y - ref[1]) ** 2,
                )
                wp_node[g, slot] = pick
                wp_dwell[g, slot] = pop.group_refuge_rest[g]
                slot += 1
    del home
    targets = sorted({int(x) for x in wp_node.ravel() if x >= 0})
    return wp_node, wp_dwell, wp_kind, mode, targets


def prepare(
    net: SimNetwork,
    pop: Population,
    scenario: SimScenario,
    config: SimConfig,
    params: Params,
    router: Router,
) -> _Prepared:
    """Turn model objects into the kernel's argument tuple."""
    G = pop.n_groups
    home = np.array([net.node(u) for u in pop.group_unit], dtype=np.int32)

    # --- blockages
    blocked_from = np.full(net.n_arcs, np.inf)
    events: list[tuple[float, list[int]]] = []
    for blk in scenario.blockages:
        arcs = [int(a) for a in net.arcs_of_edge(blk.edge_id)]
        events.append((float(blk.time), arcs))
        blocked_from[arcs] = np.minimum(blocked_from[arcs], blk.time)

    # --- route classes: assigned stair per floor, else a logit choice.
    base = router.tables(events)
    temp = params.scalar("behaviour.route_choice.stair_choice_temperature")
    group_class = choose_stair_class(
        base, home, pop.group_route_u, temp, needs_stair=pop.group_level > 0
    )
    for lv, sid in scenario.stair_assignment.items():
        cls = net.stair_ids.index(sid) + 1
        group_class[pop.group_level == lv] = cls

    wp_node, wp_dwell, wp_kind, mode, targets = _waypoints(net, pop, scenario, group_class, home)
    tables = router.tables(events, targets)
    row = {int(t): i for i, t in enumerate(tables.wp_targets)}
    wp_row = np.where(wp_node >= 0, np.vectorize(lambda x: row.get(int(x), -1))(wp_node), -1)
    wp_row = wp_row.astype(np.int32)

    # --- timing
    ready = pop.group_premovement.copy()
    for lv, t_rel in scenario.phased_release.items():
        sel = pop.group_level == lv
        ready[sel] = np.maximum(ready[sel], t_rel)

    # --- merging weights
    r = params.scalar("movement.merge.floor_deference_ratio")
    landing_dst = net.node_is_landing[net.arc_dst]
    arc_mw = np.where(landing_dst, np.where(net.arc_kind != ARC_FLAT, 1.0 - r, r), 1.0)
    pool_w = np.where(net.node_is_landing, r, 1.0)

    # --- lifts in evacuation service
    evac = [i for i, lid in enumerate(net.lifts.ids) if lid in set(scenario.evacuation_lifts)]
    outage = dict(scenario.lift_outages)
    lifts = net.lifts
    delay = params.scalar("lifts.evacuation_mode_delay")
    lift_from = np.full(len(evac), delay)
    lift_until = np.array([outage.get(lifts.ids[i], np.inf) for i in evac], dtype=np.float64)

    # --- rescue
    rescue_start = (
        scenario.rescue_start
        if scenario.rescue_start is not None
        else float(params["rescue.operations_start"].ppf(np.array([0.5]))[0])
    )
    teams = (
        scenario.rescue_teams
        if scenario.rescue_teams is not None
        else int(params.value("rescue.teams"))
    )

    # --- hazard
    if scenario.hazard is not None:
        hz_dt = float(scenario.hazard.dt)
        hz_speed = np.ascontiguousarray(scenario.hazard.speed_multiplier, dtype=np.float32)
        hz_fed = np.ascontiguousarray(scenario.hazard.fed_rate, dtype=np.float32)
    else:
        hz_dt = 1.0
        hz_speed = np.ones((0, net.n_nodes), dtype=np.float32)
        hz_fed = np.zeros((0, net.n_nodes), dtype=np.float32)

    rec_every = max(1, round(config.record_interval / config.dt))
    n_rec = int(np.ceil(config.t_max / config.dt / rec_every)) + 1 if config.record_series else 0
    fed_threshold = (
        config.fed_threshold
        if config.fed_threshold is not None
        else params.scalar("hazard.tenability.fed_incapacitation")
    )
    # Doors held open by policy get their full (open doorway) capacity back.
    arc_cap = net.arc_cap
    if scenario.held_open_doors:
        factor = params.scalar("movement.self_closing_door_capacity_factor")
        held = np.zeros(net.n_arcs, dtype=bool)
        for eid in scenario.held_open_doors:
            held[net.arcs_of_edge(eid)] = True
        arc_cap = np.where(held & net.arc_self_closing, net.arc_cap / factor, net.arc_cap)
    arc_store = net.arc_store
    if scenario.capacity_multipliers:
        mult = np.ones(net.n_arcs)
        for eid, f in scenario.capacity_multipliers:
            mult[net.arcs_of_edge(eid)] *= f
        arc_cap = arc_cap * mult
        arc_store = arc_store * mult
    hyd = "movement.hydraulic."
    args = (
        net.arc_src,
        net.arc_dst,
        net.arc_kind,
        net.arc_len,
        net.arc_area,
        arc_store,
        arc_cap,
        net.arc_rev,
        arc_mw,
        blocked_from,
        net.in_ptr,
        net.in_arc,
        net.node_is_exit,
        net.node_level,
        pool_w,
        tables.next_arc,
        tables.state_time,
        tables.wp_targets,
        tables.wp_next,
        home,
        ready,
        pop.group_space.astype(np.float64),
        pop.group_size.astype(np.float64),
        pop.group_h_speed,
        pop.group_down_speed,
        pop.group_up_speed,
        pop.group_fatigue_min,
        pop.group_fatigue_efold,
        group_class,
        mode.astype(np.int8),
        pop.group_level.astype(np.int32),
        wp_row,
        wp_dwell,
        wp_kind,
        np.ascontiguousarray(lifts.stop_node[evac]).reshape(len(evac), lifts.stop_node.shape[1]),
        lifts.discharge_node[evac],
        lifts.discharge_level[evac],
        lifts.capacity[evac],
        lifts.speed[evac],
        lift_from,
        lift_until,
        net.level_elevation,
        params.scalar("lifts.door_cycle_time"),
        params.scalar("lifts.boarding_time_per_person"),
        params.scalar("lifts.start_stop_overhead"),
        LIFT_RULES[scenario.lift_priority],
        float(rescue_start),
        teams,
        params.scalar("rescue.climb_time_per_floor"),
        params.scalar("rescue.carry_down_time_per_floor"),
        params.scalar("rescue.handling_time"),
        hz_dt,
        hz_speed,
        hz_fed,
        fed_threshold,
        params.scalar(hyd + "speed_density_a"),
        params.scalar(hyd + "free_flow_density"),
        config.speed_floor,
        params.scalar("movement.counterflow.capacity_penalty"),
        config.dt,
        config.t_max,
        rec_every,
        config.stuck_time,
    )
    del G
    named = {
        "tables": tables,
        "blocked_from": blocked_from,
        "home": home,
        "ready": ready,
        "group_class": group_class,
        "mode": mode,
        "wp_row": wp_row,
        "wp_dwell": wp_dwell,
        "wp_kind": wp_kind,
        "hz_dt": hz_dt,
        "hz_speed": hz_speed,
    }
    return _Prepared(args=args, group_class=group_class, n_rec=n_rec, named=named)


def run_meso(
    building: Building | SimNetwork,
    population: Population,
    scenario: SimScenario | None = None,
    config: SimConfig | None = None,
    *,
    params: Params | None = None,
    router: Router | None = None,
) -> MesoResult:
    """Simulate one evacuation and return per-group and per-arc results."""
    p = params or get_params()
    net = building if isinstance(building, SimNetwork) else compile_network(building, p)
    sc = scenario or SimScenario()
    cfg = config or SimConfig()
    rt = router if router is not None else Router(net, p)
    t0 = time.perf_counter()
    prep = prepare(net, population, sc, cfg, p, rt)
    G, M = population.n_groups, net.n_arcs
    out_exit = np.empty(G)
    out_left = np.empty(G)
    out_fed = np.empty(G)
    out_rescued = np.empty(G, dtype=np.bool_)
    out_state = np.empty(G, dtype=np.int8)
    maxq = np.zeros(M)
    qint = np.zeros(M)
    entries = np.zeros(M)
    series = np.zeros((M, prep.n_rec), dtype=np.float32)
    meta = np.zeros(K.META_SIZE)
    kernel: Any = K.run_kernel
    kernel(
        *prep.args,
        out_exit,
        out_left,
        out_fed,
        out_rescued,
        out_state,
        maxq,
        qint,
        entries,
        series,
        meta,
    )
    n_rec = int(meta[K.META_N_REC])
    rec_dt = max(1, round(cfg.record_interval / cfg.dt)) * cfg.dt
    return MesoResult(
        net=net,
        population=population,
        group_exit=out_exit,
        group_left_floor=out_left,
        group_fed=out_fed,
        group_rescued=out_rescued,
        group_state=out_state,
        group_class=prep.group_class,
        arc_max_queue=maxq,
        arc_queue_integral=qint,
        arc_entries=entries,
        series=series[:, :n_rec] if cfg.record_series else None,
        series_times=(np.arange(n_rec) * rec_dt).astype(np.float64) if cfg.record_series else None,
        steps=int(meta[K.META_STEPS]),
        forced_entries=int(meta[K.META_FORCED]),
        lift_trips=int(meta[K.META_LIFT_TRIPS]),
        wall_time=time.perf_counter() - t0,
        config=cfg,
        scenario=sc,
    )
