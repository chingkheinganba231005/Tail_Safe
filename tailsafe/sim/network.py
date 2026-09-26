"""Compile a :class:`~tailsafe.building.model.Building` into simulator arrays.

Each physical edge becomes one or two directed arcs. Every arc gets

* an **effective width** — clear width minus the boundary layers of the
  hydraulic model (stairs, corridors, doors);
* an **inflow capacity** (persons/s) = max specific flow × effective width
  (or the edge's explicit ``capacity``);
* an **area** used for density: length × clear width, plus, for doors, a share
  of the space in front of the door (the source node's area divided by its
  number of exits), because queues for a door form in the room before it;
* a **storage capacity** = jam density × area.

Arcs leaving exit nodes are omitted (exits are sinks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from tailsafe.building.graph import ArcKind, arcs
from tailsafe.building.model import Building, EdgeKind, NodeType
from tailsafe.config import Params, get_params

ARC_FLAT = 0
ARC_STAIR_DOWN = 1
ARC_STAIR_UP = 2
_KIND_CODE = {
    ArcKind.FLAT: ARC_FLAT,
    ArcKind.STAIR_DOWN: ARC_STAIR_DOWN,
    ArcKind.STAIR_UP: ARC_STAIR_UP,
}
_DEFAULT_NODE_AREA = 10.0  # m², for nodes without geometry


@dataclass(frozen=True)
class LiftData:
    """Lift arrays aligned with ``SimNetwork.lift_ids``."""

    ids: list[str]
    stop_node: NDArray[np.int32]  # [n_lifts, n_levels], -1 where the lift doesn't stop
    discharge_node: NDArray[np.int32]
    discharge_level: NDArray[np.int32]
    capacity: NDArray[np.float64]  # persons (able-adult space units)
    speed: NDArray[np.float64]
    firefighting: NDArray[np.bool_]


@dataclass(frozen=True)
class SimNetwork:
    """Array form of a building for the simulators."""

    building: Building
    node_ids: list[str]
    node_level: NDArray[np.int32]
    node_type: list[NodeType]
    node_area: NDArray[np.float64]
    node_is_exit: NDArray[np.bool_]
    node_is_landing: NDArray[np.bool_]
    arc_src: NDArray[np.int32]
    arc_dst: NDArray[np.int32]
    arc_edge: NDArray[np.int32]
    arc_kind: NDArray[np.int8]
    arc_len: NDArray[np.float64]
    arc_width: NDArray[np.float64]
    arc_weff: NDArray[np.float64]
    arc_cap: NDArray[np.float64]
    arc_area: NDArray[np.float64]
    arc_store: NDArray[np.float64]
    arc_rev: NDArray[np.int32]
    arc_stair: NDArray[np.int32]
    arc_is_door: NDArray[np.bool_]
    in_ptr: NDArray[np.int32]
    in_arc: NDArray[np.int32]
    out_ptr: NDArray[np.int32]
    out_arc: NDArray[np.int32]
    edge_ids: list[str]
    stair_ids: list[str]
    level_elevation: NDArray[np.float64]
    lifts: LiftData
    params_digest: str
    _index: dict[str, int] = field(repr=False)

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.node_ids)

    @property
    def n_arcs(self) -> int:
        """Number of directed arcs."""
        return int(self.arc_src.size)

    def node(self, node_id: str) -> int:
        """Index of a node id."""
        return self._index[node_id]

    @cached_property
    def edge_index(self) -> dict[str, int]:
        """Index of each physical edge id."""
        return {e: i for i, e in enumerate(self.edge_ids)}

    def arcs_of_edge(self, edge_id: str) -> NDArray[np.int32]:
        """Arc indices (one or two) of a physical edge."""
        out: NDArray[np.int32] = np.flatnonzero(self.arc_edge == self.edge_index[edge_id]).astype(
            np.int32
        )
        return out

    def arc_label(self, a: int) -> str:
        """Plain-English label of arc ``a`` including direction."""
        e = self.building.edge_by_id[self.edge_ids[int(self.arc_edge[a])]]
        base = self.building.describe_edge(e.id)
        if self.arc_kind[a] == ARC_STAIR_UP:
            return base + " (upwards)"
        return base


def compile_network(building: Building, params: Params | None = None) -> SimNetwork:
    """Build the simulator arrays for ``building`` using registry values from ``params``."""
    p = params or get_params()
    hyd = "movement.hydraulic."
    bl = "movement.boundary_layer."
    fs_h = p.scalar(hyd + "max_specific_flow_horizontal")
    fs_s = p.scalar(hyd + "max_specific_flow_stair")
    d_jam = p.scalar(hyd + "jam_density")
    bl_stair, bl_corr, bl_door = (p.scalar(bl + k) for k in ("stair", "corridor", "door"))

    nodes = building.nodes
    index = {n.id: i for i, n in enumerate(nodes)}
    N = len(nodes)
    node_area = np.array(
        [n.area if n.area is not None else _DEFAULT_NODE_AREA for n in nodes], dtype=np.float64
    )
    is_exit = np.array([n.type == NodeType.EXIT for n in nodes], dtype=bool)

    all_arcs = [a for a in arcs(building) if not is_exit[index[a.source]]]
    edges = building.edge_by_id
    edge_ids = [e.id for e in building.edges]
    edge_idx = {e: i for i, e in enumerate(edge_ids)}
    stair_ids = [s.id for s in building.stairs]
    stair_idx = {s: i for i, s in enumerate(stair_ids)}

    src = np.array([index[a.source] for a in all_arcs], dtype=np.int32)
    dst = np.array([index[a.target] for a in all_arcs], dtype=np.int32)
    outdeg = np.bincount(src, minlength=N).astype(np.float64)
    kind = np.array([_KIND_CODE[a.kind] for a in all_arcs], dtype=np.int8)
    length = np.array([edges[a.edge_id].length for a in all_arcs])
    width = np.array([edges[a.edge_id].width for a in all_arcs])
    is_door = np.array([edges[a.edge_id].kind == EdgeKind.DOOR for a in all_arcs], dtype=bool)
    is_stair = kind != ARC_FLAT
    boundary = np.where(is_stair, bl_stair, np.where(is_door, bl_door, bl_corr))
    weff = np.maximum(width - 2.0 * boundary, 0.2)
    cap = np.where(is_stair, fs_s, fs_h) * weff
    override = np.array([edges[a.edge_id].capacity or np.nan for a in all_arcs], dtype=np.float64)
    cap = np.where(np.isnan(override), cap, override)
    area = length * width + np.where(is_door, node_area[src] / np.maximum(outdeg[src], 1.0), 0.0)
    store = np.maximum(d_jam * area, 2.0)

    pair: dict[tuple[str, bool], int] = {(a.edge_id, a.forward): i for i, a in enumerate(all_arcs)}
    rev = np.array([pair.get((a.edge_id, not a.forward), -1) for a in all_arcs], dtype=np.int32)
    arc_stair = np.array(
        [stair_idx.get(edges[a.edge_id].stair or "", -1) for a in all_arcs], dtype=np.int32
    )

    in_order = np.argsort(dst, kind="stable")
    in_ptr = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(np.bincount(dst, minlength=N), out=in_ptr[1:])
    out_order = np.argsort(src, kind="stable")
    out_ptr = np.zeros(N + 1, dtype=np.int32)
    np.cumsum(np.bincount(src, minlength=N), out=out_ptr[1:])

    levels = sorted(building.level_by_index)
    if levels[0] < 0:
        raise ValueError("basement levels are not supported by the simulator yet")
    elevation = np.zeros(levels[-1] + 1)
    for lv in building.levels:
        elevation[lv.index] = lv.elevation

    lift_ids = [lf.id for lf in building.lifts]
    stop_node = np.full((len(lift_ids), elevation.size), -1, dtype=np.int32)
    for i, lf in enumerate(building.lifts):
        for stop in lf.stops:
            stop_node[i, stop.level] = index[stop.node]
    default_cap = p.scalar("lifts.car_capacity")
    default_speed = p.scalar("lifts.rated_speed")
    lifts = LiftData(
        ids=lift_ids,
        stop_node=stop_node,
        discharge_node=np.array(
            [stop_node[i, lf.discharge_level] for i, lf in enumerate(building.lifts)],
            dtype=np.int32,
        ),
        discharge_level=np.array([lf.discharge_level for lf in building.lifts], dtype=np.int32),
        capacity=np.array([lf.car_capacity or default_cap for lf in building.lifts], dtype=float),
        speed=np.array([lf.rated_speed or default_speed for lf in building.lifts], dtype=float),
        firefighting=np.array([lf.firefighting for lf in building.lifts], dtype=bool),
    )

    return SimNetwork(
        building=building,
        node_ids=[n.id for n in nodes],
        node_level=np.array([n.level for n in nodes], dtype=np.int32),
        node_type=[n.type for n in nodes],
        node_area=node_area,
        node_is_exit=is_exit,
        node_is_landing=np.array([n.type == NodeType.STAIR_LANDING for n in nodes], dtype=bool),
        arc_src=src,
        arc_dst=dst,
        arc_edge=np.array([edge_idx[a.edge_id] for a in all_arcs], dtype=np.int32),
        arc_kind=kind,
        arc_len=length,
        arc_width=width,
        arc_weff=weff,
        arc_cap=cap,
        arc_area=area,
        arc_store=store,
        arc_rev=rev,
        arc_stair=arc_stair,
        arc_is_door=is_door,
        in_ptr=in_ptr,
        in_arc=in_order.astype(np.int32),
        out_ptr=out_ptr,
        out_arc=out_order.astype(np.int32),
        edge_ids=edge_ids,
        stair_ids=stair_ids,
        level_elevation=elevation,
        lifts=lifts,
        params_digest=p.digest,
        _index=index,
    )
