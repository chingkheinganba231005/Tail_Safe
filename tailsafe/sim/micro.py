"""Microscopic replay engine: people as discs on the floor plan.

The meso engine (:mod:`tailsafe.sim.meso`) is fast enough for thousands of
Monte Carlo runs; this engine replays a *selected* scenario (median, P95,
worst, before/after an intervention) person by person for animation, and to
cross-check the meso engine.

Model (see ``_micro_kernel``): a collision-free speed model (after Tordeux,
Chraibi & Seyfried 2016) in the rectangles of each floor, connected by door
openings, with stair flights as multi-lane strips between landings. Everything
that is a *decision* is shared with the meso engine through
:func:`tailsafe.sim.meso.prepare` — reaction times, stair choice, stair
assignments, counter-flow and refuge waypoints, blockages and when people
discover them, smoke-reduced walking speeds — so the two engines differ only
in how people move and queue.

Not walked here (taken from the meso run of the same scenario when one is
given): households waiting for an evacuation lift or for fire-service rescue,
and anyone left without a usable route. Toxic dose is not recomputed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tailsafe.building.model import Building
from tailsafe.config import Params, get_params
from tailsafe.population.profiles import Profile
from tailsafe.population.synth import Mode, Population
from tailsafe.rng import stream
from tailsafe.sim import _micro_kernel as MK
from tailsafe.sim.meso import MesoResult, SimConfig, SimScenario, prepare
from tailsafe.sim.network import ARC_FLAT, ARC_STAIR_DOWN, SimNetwork, compile_network
from tailsafe.sim.routing import Router

Point = tuple[float, float]


@dataclass(frozen=True)
class MicroConfig:
    """Numerical settings of the micro engine."""

    dt: float = 0.1
    t_max: float = 4 * 3600.0
    record_interval: float = 2.0
    stuck_time: float = 30.0


@dataclass(frozen=True)
class MicroGeometry:
    """Rectangles of the nodes and portals / flight end points of the arcs."""

    n_x0: NDArray[np.float64]
    n_x1: NDArray[np.float64]
    n_y0: NDArray[np.float64]
    n_y1: NDArray[np.float64]
    arc_lanes: NDArray[np.int32]  # people abreast in a doorway or on a flight
    arc_lane_w: NDArray[np.float64]
    p_x0: NDArray[np.float64]  # flat arcs: portal start; stair arcs: departure point
    p_y0: NDArray[np.float64]
    p_x1: NDArray[np.float64]  # flat arcs: portal end; stair arcs: arrival point
    p_y1: NDArray[np.float64]
    p_nx: NDArray[np.float64]  # flat arcs: portal normal towards the target room;
    p_ny: NDArray[np.float64]  # stair arcs: unit vector across the landings


def _bbox(poly: list[list[float]] | list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def _shared_boundary(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[Point, Point] | None:
    """Longest common piece of the boundaries of two axis-aligned rectangles."""
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    tol = 1e-3
    cands: list[tuple[Point, Point]] = []
    for xa in (ax0, ax1):
        for xb in (bx0, bx1):
            if abs(xa - xb) < tol:
                lo, hi = max(ay0, by0), min(ay1, by1)
                if hi - lo > tol:
                    cands.append(((xa, lo), (xa, hi)))
    for ya in (ay0, ay1):
        for yb in (by0, by1):
            if abs(ya - yb) < tol:
                lo, hi = max(ax0, bx0), min(ax1, bx1)
                if hi - lo > tol:
                    cands.append(((lo, ya), (hi, ya)))
    if not cands:
        return None
    return max(cands, key=lambda c: abs(c[1][0] - c[0][0]) + abs(c[1][1] - c[0][1]))


def _clip_centred(seg: tuple[Point, Point], width: float) -> tuple[Point, Point]:
    (x0, y0), (x1, y1) = seg
    length = float(np.hypot(x1 - x0, y1 - y0))
    if width >= length or length <= 0:
        return seg
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    h = width / 2
    return (mx - ux * h, my - uy * h), (mx + ux * h, my + uy * h)


def _facing_side(
    box: tuple[float, float, float, float], tx: float, ty: float, width: float
) -> tuple[Point, Point]:
    """A segment of ``width`` on the rectangle side nearest to point (tx, ty)."""
    x0, x1, y0, y1 = box
    px, py = min(max(tx, x0), x1), min(max(ty, y0), y1)
    if x0 < tx < x1 and y0 < ty < y1:  # target inside: use the nearest side
        d = {"x0": tx - x0, "x1": x1 - tx, "y0": ty - y0, "y1": y1 - ty}
        side = min(d, key=d.__getitem__)
    else:
        dx = max(x0 - tx, 0.0, tx - x1)
        dy = max(y0 - ty, 0.0, ty - y1)
        side = ("x0" if tx < x0 else "x1") if dx >= dy else ("y0" if ty < y0 else "y1")
    h = width / 2
    if side in ("x0", "x1"):
        xs = x0 if side == "x0" else x1
        c = min(max(py, y0 + h), y1 - h) if y1 - y0 > width else (y0 + y1) / 2
        return (xs, c - h), (xs, c + h)
    ys = y0 if side == "y0" else y1
    c = min(max(px, x0 + h), x1 - h) if x1 - x0 > width else (x0 + x1) / 2
    return (c - h, ys), (c + h, ys)


def micro_problems(building: Building) -> list[str]:
    """Why a building cannot be replayed microscopically (empty if it can).

    The micro engine walks people through room geometry, so every non-exit node
    needs a polygon, and rooms joined by a walkway (not a door) must touch.
    """
    out: list[str] = []
    by = building.node_by_id
    missing = [n.id for n in building.nodes if n.type.value != "exit" and not n.polygon]
    if missing:
        out.append(f"{len(missing)} node(s) without a polygon, e.g. {missing[0]}")
    apart = []
    for e in building.edges:
        if e.kind.value != "flat" or e.opening:
            continue
        s, t = by[e.source], by[e.target]
        if s.polygon and t.polygon and _shared_boundary(_bbox(s.polygon), _bbox(t.polygon)) is None:
            apart.append(e.id)
    if apart:
        out.append(f"{len(apart)} walkway(s) between rooms that do not touch, e.g. {apart[0]}")
    return out


def compile_geometry(net: SimNetwork, params: Params | None = None) -> MicroGeometry:
    """Rectangles, door portals and stair end points for the micro engine.

    Node polygons are used through their bounding boxes (the procedural
    templates only produce axis-aligned rectangles). Nodes without geometry get
    a square of their floor area around their position.
    """
    p = params or get_params()
    b: Building = net.building
    N, M = net.n_nodes, net.n_arcs
    boxes: list[tuple[float, float, float, float]] = []
    for n in b.nodes:
        if n.polygon and len(n.polygon) >= 3:
            boxes.append(_bbox(n.polygon))
        else:
            h = float(np.sqrt(max(n.area or 4.0, 1.0))) / 2
            boxes.append((n.x - h, n.x + h, n.y - h, n.y + h))
    box = np.array(boxes, dtype=np.float64).reshape(N, 4)

    lane_w = p.scalar("movement.micro.lane_width")
    lanes = np.ones(M, dtype=np.int32)
    lane_width = np.full(M, lane_w)
    px0, py0, px1, py1 = (np.zeros(M) for _ in range(4))
    pnx, pny = np.zeros(M), np.zeros(M)

    def landing_points(v: int) -> tuple[Point, Point, Point]:
        """(down-flight side, up-flight side, unit vector across) of a landing.

        Dog-leg stairs meet a floor landing at one end, side by side: the flight
        down leaves from one half of that end and the flight from above arrives
        on the other half.
        """
        x0, x1, y0, y1 = box[v]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if x1 - x0 >= y1 - y0:
            end = x0 + min(0.5, (x1 - x0) / 4)
            q = (y1 - y0) / 4
            return (end, cy - q), (end, cy + q), (0.0, 1.0)
        end = y0 + min(0.5, (y1 - y0) / 4)
        q = (x1 - x0) / 4
        return (cx - q, end), (cx + q, end), (1.0, 0.0)

    for a in range(M):
        e = b.edge_by_id[net.edge_ids[int(net.arc_edge[a])]]
        src, dst = int(net.arc_src[a]), int(net.arc_dst[a])
        if net.arc_kind[a] == ARC_FLAT:
            seg: tuple[Point, Point] | None = None
            if e.opening and len(e.opening) == 2:
                seg = (
                    (float(e.opening[0][0]), float(e.opening[0][1])),
                    (float(e.opening[1][0]), float(e.opening[1][1])),
                )
            elif not net.node_is_exit[dst]:
                shared = _shared_boundary(boxes[src], boxes[dst])
                if shared is not None:
                    seg = _clip_centred(shared, float(e.width))
            if seg is None:
                # No opening: a doorway of the edge's width on the side of the
                # source rectangle facing the target node.
                tgt = b.nodes[dst]
                seg = _facing_side(boxes[src], tgt.x, tgt.y, float(e.width))
            (x0, y0), (x1, y1) = seg
            px0[a], py0[a], px1[a], py1[a] = x0, y0, x1, y1
            # normal pointing from the source rectangle's centre through the portal
            tx, ty = x1 - x0, y1 - y0
            tl = float(np.hypot(tx, ty)) or 1.0
            nx, ny = -ty / tl, tx / tl
            sx0, sx1, sy0, sy1 = box[src]
            cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
            if (x0 + x1) / 2 * nx + (y0 + y1) / 2 * ny < cx * nx + cy * ny:
                nx, ny = -nx, -ny
            # snap to the axes (rectangles are axis-aligned)
            if abs(nx) >= abs(ny):
                nx, ny = float(np.sign(nx)), 0.0
            else:
                nx, ny = 0.0, float(np.sign(ny))
            pnx[a], pny[a] = nx, ny
            portal = float(np.hypot(x1 - x0, y1 - y0))
            lanes[a] = max(1, int(np.floor(portal / lane_w + 1e-9)))
            lane_width[a] = portal / lanes[a]
        else:
            lanes[a] = max(1, int(np.floor(float(e.width) / lane_w + 1e-9)))
            lane_width[a] = float(e.width) / lanes[a]
            s_down, s_up, across = landing_points(src)
            d_down, d_up, _ = landing_points(dst)
            if net.arc_kind[a] == ARC_STAIR_DOWN:
                dep, arr = s_down, d_up  # arrive on the side the next flight up starts
            else:
                dep, arr = s_up, d_down
            px0[a], py0[a] = dep
            px1[a], py1[a] = arr
            pnx[a], pny[a] = across
    return MicroGeometry(
        n_x0=box[:, 0].copy(),
        n_x1=box[:, 1].copy(),
        n_y0=box[:, 2].copy(),
        n_y1=box[:, 3].copy(),
        arc_lanes=lanes,
        arc_lane_w=lane_width,
        p_x0=px0,
        p_y0=py0,
        p_x1=px1,
        p_y1=py1,
        p_nx=pnx,
        p_ny=pny,
    )


@dataclass
class MicroResult:
    """Outcome of a micro replay."""

    net: SimNetwork
    population: Population
    agent_exit: NDArray[np.float64]  # s; inf = not out
    agent_left_floor: NDArray[np.float64]
    agent_walked: NDArray[np.bool_]  # False: taken from the meso run (lift, rescue, stranded)
    agent_state: NDArray[np.int8]
    agent_node: NDArray[np.int32]  # final room; -1 - arc when on a stair flight
    agent_arc: NDArray[np.int32]  # final target arc (-2 = none yet)
    frame_times: NDArray[np.float64]
    frame_level: NDArray[np.int16]  # [frames, agents], -1 = out of the building
    frame_x: NDArray[np.float32]
    frame_y: NDArray[np.float32]
    frame_state: NDArray[np.int8]
    steps: int
    forced_moves: int
    wall_time: float

    def _times(self, walked_only: bool) -> NDArray[np.float64]:
        t = self.agent_exit
        return t[self.agent_walked] if walked_only else t

    def total_time(self, walked_only: bool = False) -> float:
        """Time the last person is out (``inf`` if someone never is)."""
        t = self._times(walked_only)
        return float(t.max()) if t.size else 0.0

    def exit_time_quantile(self, q: float, walked_only: bool = False) -> float:
        """Occupant quantile of exit times (same definition as the meso engine)."""
        t = np.sort(self._times(walked_only))
        if t.size == 0:
            return 0.0
        return float(t[min(int(np.ceil(q * t.size)) - 1, t.size - 1)] if q > 0 else t[0])

    def summary(self) -> dict[str, Any]:
        """Headline numbers (all occupants, and walkers only)."""
        return {
            "occupants": int(self.agent_exit.size),
            "walked": int(self.agent_walked.sum()),
            "total_time_s": self.total_time(),
            "p50_exit_s": self.exit_time_quantile(0.5),
            "p95_exit_s": self.exit_time_quantile(0.95),
            "walkers_last_out_s": self.total_time(walked_only=True),
            "walkers_p95_exit_s": self.exit_time_quantile(0.95, walked_only=True),
            "not_out": int((~np.isfinite(self.agent_exit)).sum()),
            "forced_moves": self.forced_moves,
            "steps": self.steps,
            "wall_time_s": self.wall_time,
        }


def _initial_positions(
    geom: MicroGeometry,
    home: NDArray[np.int32],
    agent_group: NDArray[np.int32],
    radius: NDArray[np.float64],
    seed: int,
    index: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Members of each household spread on a grid in their flat, with jitter."""
    rng = stream(seed, "micro", index)
    A = agent_group.size
    x, y = np.empty(A), np.empty(A)
    jitter = rng.random((A, 2))
    rank = np.zeros(A, dtype=np.int64)
    seen: dict[int, int] = {}
    for i, g in enumerate(agent_group.tolist()):
        rank[i] = seen.get(g, 0)
        seen[g] = rank[i] + 1
    for i in range(A):
        v = int(home[agent_group[i]])
        r = float(radius[i])
        x0, x1 = geom.n_x0[v] + r, geom.n_x1[v] - r
        y0, y1 = geom.n_y0[v] + r, geom.n_y1[v] - r
        cols = max(1, int((x1 - x0) // (3 * r)))
        k = int(rank[i])
        gx = (k % cols + 0.5 + 0.3 * (jitter[i, 0] - 0.5)) / cols
        rows = max(1, int((y1 - y0) // (3 * r)))
        gy = ((k // cols) % rows + 0.5 + 0.3 * (jitter[i, 1] - 0.5)) / rows
        x[i] = x0 + max(0.0, x1 - x0) * gx if x1 > x0 else (geom.n_x0[v] + geom.n_x1[v]) / 2
        y[i] = y0 + max(0.0, y1 - y0) * gy if y1 > y0 else (geom.n_y0[v] + geom.n_y1[v]) / 2
    return x, y


def run_micro(
    building: Building | SimNetwork,
    population: Population,
    scenario: SimScenario | None = None,
    config: MicroConfig | None = None,
    *,
    params: Params | None = None,
    router: Router | None = None,
    meso: MesoResult | None = None,
    seed: int = 0,
    index: int = 0,
) -> MicroResult:
    """Replay one scenario person by person.

    ``meso`` (the meso run of the same scenario) supplies the exit times of the
    households this engine does not walk (lift and rescue). ``seed`` and
    ``index`` only place people inside their flats.
    """
    p = params or get_params()
    net = building if isinstance(building, SimNetwork) else compile_network(building, p)
    problems = micro_problems(net.building)
    if problems:
        raise ValueError("building cannot be replayed microscopically: " + "; ".join(problems))
    sc = scenario or SimScenario()
    cfg = config or MicroConfig()
    rt = router if router is not None else Router(net, p)
    t0 = time.perf_counter()
    prep = prepare(net, population, sc, SimConfig(dt=cfg.dt, t_max=cfg.t_max), p, rt).named
    geom = compile_geometry(net, p)
    tables = prep["tables"]

    pop = population
    ag = pop.agent_group
    base_r = p.scalar("movement.micro.body_radius")
    factor = {int(pr): p.scalar(f"profiles.{pr.key}.space_factor") for pr in Profile}
    radius = base_r * np.sqrt(np.array([factor[int(x)] for x in pop.agent_profile]))
    mode = prep["mode"]
    walked = np.isin(mode[ag], [int(Mode.WALK), int(Mode.ASSISTED_STAIR)])
    home = prep["home"]
    x0, y0 = _initial_positions(geom, home, ag, radius, seed, index)

    A = ag.size
    n_rec = int(np.ceil(cfg.t_max / cfg.record_interval)) + 1
    rec_every = max(1, round(cfg.record_interval / cfg.dt))
    # Recording arrays are sized for the worst case but trimmed after the run.
    n_rec = min(n_rec, 8000)
    rec_level = np.full((n_rec, A), -1, dtype=np.int16)
    rec_x = np.zeros((n_rec, A), dtype=np.float32)
    rec_y = np.zeros((n_rec, A), dtype=np.float32)
    rec_state = np.zeros((n_rec, A), dtype=np.int8)
    out_exit = np.empty(A)
    out_left = np.empty(A)
    out_state = np.empty(A, dtype=np.int8)
    out_node = np.empty(A, dtype=np.int32)
    out_arc = np.empty(A, dtype=np.int32)
    meta = np.zeros(MK.META_SIZE)
    micro = "movement.micro."
    kernel: Any = MK.run_micro_kernel
    kernel(
        geom.n_x0,
        geom.n_x1,
        geom.n_y0,
        geom.n_y1,
        net.node_level,
        net.node_is_exit,
        net.node_area * p.scalar("movement.hydraulic.jam_density"),
        net.arc_src,
        net.arc_dst,
        net.arc_kind,
        net.arc_len,
        geom.arc_lanes,
        geom.arc_lane_w,
        net.arc_rev,
        geom.p_x0,
        geom.p_y0,
        geom.p_x1,
        geom.p_y1,
        geom.p_nx,
        geom.p_ny,
        tables.next_arc,
        tables.state_time,
        tables.wp_targets,
        tables.wp_next,
        prep["blocked_from"],
        home[ag].astype(np.int32),
        x0,
        y0,
        prep["ready"][ag],
        pop.group_h_speed[ag],
        pop.group_down_speed[ag],
        pop.group_up_speed[ag],
        pop.group_fatigue_min[ag],
        pop.group_fatigue_efold[ag],
        radius,
        prep["group_class"][ag].astype(np.int32),
        np.ascontiguousarray(prep["wp_row"][ag]),
        np.ascontiguousarray(prep["wp_dwell"][ag]),
        np.ascontiguousarray(prep["wp_kind"][ag]),
        walked,
        float(prep["hz_dt"]),
        prep["hz_speed"],
        p.scalar(micro + "time_gap"),
        p.scalar(micro + "repulsion_strength"),
        p.scalar(micro + "repulsion_range"),
        p.scalar(micro + "wall_repulsion_strength"),
        p.scalar(micro + "wall_repulsion_range"),
        cfg.dt,
        cfg.t_max,
        cfg.stuck_time,
        rec_every,
        out_exit,
        out_left,
        out_state,
        out_node,
        out_arc,
        rec_level,
        rec_x,
        rec_y,
        rec_state,
        meta,
    )
    n_frames = int(meta[MK.META_N_REC])
    exit_t = out_exit.copy()
    left_t = out_left.copy()
    stranded = out_state == MK.S_STRANDED
    from_meso = ~walked | stranded
    if meso is not None:
        exit_t[from_meso] = meso.group_exit[ag[from_meso]]
        left_t[from_meso] = meso.group_left_floor[ag[from_meso]]
    walked_final = walked & ~stranded
    frame_times = (np.arange(n_frames) * (rec_every * cfg.dt)).astype(np.float64)
    level = rec_level[:n_frames].copy()
    # People not walked here are shown at home until they leave (per meso).
    if (~walked_final).any():
        idx = np.flatnonzero(~walked_final)
        gone = frame_times[:, None] >= exit_t[idx][None, :]
        level[:, idx] = np.where(gone, -1, level[:, idx])
    return MicroResult(
        net=net,
        population=pop,
        agent_exit=exit_t,
        agent_left_floor=left_t,
        agent_walked=walked_final,
        agent_state=out_state,
        agent_node=out_node,
        agent_arc=out_arc,
        frame_times=frame_times,
        frame_level=level,
        frame_x=rec_x[:n_frames].copy(),
        frame_y=rec_y[:n_frames].copy(),
        frame_state=rec_state[:n_frames].copy(),
        steps=int(meta[MK.META_STEPS]),
        forced_moves=int(meta[MK.META_FORCED]),
        wall_time=time.perf_counter() - t0,
    )


def fundamental_diagram(
    densities: list[float] | tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
    *,
    length: float = 16.0,
    width: float = 2.0,
    free_speed: float | None = None,
    seed: int = 0,
    params: Params | None = None,
    dt: float = 0.05,
    t_warm: float = 30.0,
    t_meas: float = 30.0,
) -> list[dict[str, float]]:
    """Mean speed against density in a periodic corridor, with the hydraulic reference.

    Everyone has the same free speed (default: the hydraulic horizontal speed
    constant) and able-adult size. The hydraulic reference is
    ``S = k (1 - a D)`` above the free-flow density.
    """
    p = params or get_params()
    hyd = "movement.hydraulic."
    k = free_speed if free_speed is not None else p.scalar(hyd + "k_horizontal")
    a = p.scalar(hyd + "speed_density_a")
    d0 = p.scalar(hyd + "free_flow_density")
    r = p.scalar("movement.micro.body_radius")
    micro = "movement.micro."
    rng = stream(seed, "micro-fd")
    out = []
    kernel: Any = MK.periodic_corridor
    for dens in densities:
        n = max(1, round(dens * length * width))
        # start on a jittered lattice so nobody overlaps
        cols = max(1, int(np.ceil(np.sqrt(n * length / width))))
        rows = int(np.ceil(n / cols))
        idx = np.arange(n)
        x0 = ((idx % cols) + 0.5 + 0.2 * (rng.random(n) - 0.5)) * length / cols
        y0 = ((idx // cols) + 0.5 + 0.2 * (rng.random(n) - 0.5)) * width / rows
        y0 = np.clip(y0, r, width - r)
        v = kernel(
            x0,
            y0,
            np.full(n, r),
            np.full(n, k),
            length,
            width,
            p.scalar(micro + "time_gap"),
            p.scalar(micro + "repulsion_strength"),
            p.scalar(micro + "repulsion_range"),
            p.scalar(micro + "wall_repulsion_strength"),
            p.scalar(micro + "wall_repulsion_range"),
            dt,
            t_warm,
            t_meas,
        )
        density = n / (length * width)
        hyd_speed = k if density <= d0 else max(0.0, k * (1 - a * density))
        out.append(
            {
                "density": density,
                "speed": float(v),
                "flow": float(v) * density,
                "hydraulic_speed": hyd_speed,
                "hydraulic_flow": hyd_speed * density,
            }
        )
    return out
