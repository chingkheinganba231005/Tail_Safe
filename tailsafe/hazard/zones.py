"""Zone network for smoke transport, derived from the building graph.

Every non-exit node is a well-mixed zone (volume = floor area × storey height).
Physical connections become *exchange flows* (m³/s) in each direction:

* open connections (``flat`` edges, open doors): ``v_h × opening area``;
* closed self-closing doors leak a fraction of that (less if fire-rated); stair
  doors are also open part of the time while people pass through;
* stair flights and lift shafts exchange vertically, scaled by ``1 + b`` upwards
  and ``1 - b`` downwards (stack effect);
* exits are sinks (outside air); open-sided refuge floors are ventilated.

Products carried by these flows follow ``dm_i/dt = Σ_j F_ji m_j/V_j − Σ_j F_ij m_i/V_i``,
a linear network model in the spirit of multi-zone (CONTAM-type) models, much
simpler than CFD.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tailsafe.building.model import Building, EdgeKind, NodeType
from tailsafe.config import Params, get_params

_DEFAULT_AREA = 10.0


@dataclass(frozen=True)
class ZoneNetwork:
    """Directed exchange flows between zones."""

    node_ids: list[str]
    volume: NDArray[np.float64]
    is_sink: NDArray[np.bool_]
    ex_src: NDArray[np.int32]
    ex_dst: NDArray[np.int32]
    ex_rate: NDArray[np.float64]
    ex_edge: NDArray[np.int32]  # physical edge index, -1 for lift shafts
    vent_rate: NDArray[np.float64]  # 1/s, loss to outside

    @property
    def n_zones(self) -> int:
        """Number of zones (nodes)."""
        return int(self.volume.size)

    def stable_dt(self, safety: float = 0.8) -> float:
        """Largest explicit time step keeping every zone's outflow fraction ≤ ``safety``."""
        out = np.bincount(self.ex_src, weights=self.ex_rate, minlength=self.n_zones)
        rate = out / self.volume + self.vent_rate
        worst = float(rate[~self.is_sink].max(initial=0.0))
        return safety / worst if worst > 0 else 10.0

    def with_door_open(self, edge_index: int, open_rate: float) -> ZoneNetwork:
        """Copy with the exchange through one door set to ``open_rate`` both ways."""
        rate = self.ex_rate.copy()
        rate[self.ex_edge == edge_index] = open_rate
        return ZoneNetwork(
            node_ids=self.node_ids,
            volume=self.volume,
            is_sink=self.is_sink,
            ex_src=self.ex_src,
            ex_dst=self.ex_dst,
            ex_rate=rate,
            ex_edge=self.ex_edge,
            vent_rate=self.vent_rate,
        )


def open_door_rate(width: float, params: Params | None = None) -> float:
    """Exchange flow (m³/s) through an open doorway of ``width`` metres."""
    p = params or get_params()
    t = "hazard.transport."
    return p.scalar(t + "horizontal_exchange_velocity") * width * p.scalar(t + "opening_height")


def build_zones(
    building: Building,
    params: Params | None = None,
    *,
    held_open: Iterable[str] = (),
    fire_node: str | None = None,
) -> ZoneNetwork:
    """Zone network of ``building``. ``held_open`` door edges are propped open.

    ``fire_node`` gets the fire-room vent (broken windows / openings).
    """
    p = params or get_params()
    t = "hazard.transport."
    v_h = p.scalar(t + "horizontal_exchange_velocity")
    h_open = p.scalar(t + "opening_height")
    v_s = p.scalar(t + "shaft_exchange_velocity")
    bias = p.scalar(t + "stack_bias")
    leak = p.scalar(t + "closed_door_leakage")
    fire_leak = p.scalar(t + "fire_door_leakage")
    usage = p.scalar(t + "door_usage_open_fraction")
    lift_area = p.scalar(t + "lift_shaft_leakage_area")
    lift_v = p.scalar(t + "lift_shaft_velocity")
    refuge_vent = p.scalar(t + "refuge_vent_rate")
    fire_vent = p.scalar(t + "fire_room_vent_rate")
    landing_depth = p.scalar("building_defaults.stair_landing_depth")
    held = set(held_open)

    nodes = building.nodes
    index = {n.id: i for i, n in enumerate(nodes)}
    heights = {lv.index: lv.height for lv in building.levels}
    volume = np.array(
        [(n.area or _DEFAULT_AREA) * heights.get(n.level, 3.0) for n in nodes], dtype=np.float64
    )
    is_sink = np.array([n.type == NodeType.EXIT for n in nodes], dtype=bool)
    stairish = {NodeType.STAIR_LANDING, NodeType.PROTECTED_LOBBY}

    src: list[int] = []
    dst: list[int] = []
    rate: list[float] = []
    edge: list[int] = []

    def add(i: int, j: int, f: float, e: int) -> None:
        if f <= 0.0:
            return
        if not is_sink[i]:
            src.append(i)
            dst.append(j)
            rate.append(f)
            edge.append(e)

    for k, e in enumerate(building.edges):
        i, j = index[e.source], index[e.target]
        if e.kind == EdgeKind.STAIR:
            area = e.width * landing_depth
            add(j, i, v_s * area * (1.0 + bias), k)  # lower -> upper
            add(i, j, v_s * area * (1.0 - bias), k)  # upper -> lower
            continue
        if e.kind == EdgeKind.FLAT:
            f = v_h * min(e.width, 3.0) * h_open
        else:
            f_open = v_h * e.width * h_open
            if e.id in held or not e.self_closing:
                frac = 1.0
            else:
                frac = fire_leak if e.fire_rated else leak
                a, b = nodes[i], nodes[j]
                if a.type in stairish or b.type in stairish:
                    frac = min(1.0, frac + usage)
            f = f_open * frac
        add(i, j, f, k)
        add(j, i, f, k)

    for lift in building.lifts:
        stops = sorted(lift.stops, key=lambda s: s.level)
        for lo, hi in itertools.pairwise(stops):
            lo_i, hi_i = index[lo.node], index[hi.node]
            add(lo_i, hi_i, lift_v * lift_area * (1.0 + bias), -1)
            add(hi_i, lo_i, lift_v * lift_area * (1.0 - bias), -1)

    vent = np.array(
        [refuge_vent if n.type == NodeType.REFUGE else 0.0 for n in nodes], dtype=np.float64
    )
    if fire_node is not None:
        vent[index[fire_node]] = max(vent[index[fire_node]], fire_vent)
    return ZoneNetwork(
        node_ids=[n.id for n in nodes],
        volume=volume,
        is_sink=is_sink,
        ex_src=np.array(src, dtype=np.int32),
        ex_dst=np.array(dst, dtype=np.int32),
        ex_rate=np.array(rate, dtype=np.float64),
        ex_edge=np.array(edge, dtype=np.int32),
        vent_rate=vent,
    )
