"""Structural bottlenecks of the egress graph (no simulation needed).

* **Min-cut / max-flow.** With every unit as a source and every exit as a sink,
  and arc capacities equal to their flow capacities (persons/s), the maximum
  flow is the building's best possible egress rate; the arcs of a minimum cut
  are the set that limits it.
* **Flow-weighted load.** Each unit's expected occupants are routed along the
  simulator's route tables (split between staircases by the same logit as the
  simulator). An arc's *load* is how many people are expected to use it; load
  divided by capacity is the minimum time needed to pass them all — the
  structural "clearance time" of that element.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from tailsafe.config import Params, get_params
from tailsafe.sim.network import SimNetwork
from tailsafe.sim.routing import Router

_SOURCE = "__units__"
_SINK = "__exits__"


@dataclass(frozen=True)
class Structural:
    """Graph-only bottleneck indicators per arc."""

    max_flow: float  # persons/s
    min_cut_arcs: list[int]
    load: NDArray[np.float64]  # expected persons per arc
    clearance_time: NDArray[np.float64]  # load / capacity (s)


def max_flow_min_cut(net: SimNetwork) -> tuple[float, list[int]]:
    """Maximum egress rate (persons/s) and the arcs of a minimum cut."""
    g = nx.DiGraph()
    for a in range(net.n_arcs):
        u, v = int(net.arc_src[a]), int(net.arc_dst[a])
        cap = float(net.arc_cap[a])
        if g.has_edge(u, v):
            g[u][v]["capacity"] += cap
            g[u][v]["arcs"].append(a)
        else:
            g.add_edge(u, v, capacity=cap, arcs=[a])
    units = [i for i, t in enumerate(net.node_type) if t.value == "unit"]
    for u in units:
        g.add_edge(_SOURCE, u)  # no capacity attribute = infinite
    for x in np.flatnonzero(net.node_is_exit):
        g.add_edge(int(x), _SINK)
    value, (reach, _) = nx.minimum_cut(g, _SOURCE, _SINK)
    cut: list[int] = []
    for u in reach:
        for v, data in g[u].items():
            if v not in reach and "arcs" in data:
                cut.extend(data["arcs"])
    return float(value), sorted(cut)


def flow_load(
    net: SimNetwork,
    unit_weight: NDArray[np.float64] | None = None,
    params: Params | None = None,
    router: Router | None = None,
) -> NDArray[np.float64]:
    """Expected persons on each arc when every unit evacuates along its routes."""
    p = params or get_params()
    rt = router or Router(net, p)
    tables = rt.tables()
    units = np.array([i for i, t in enumerate(net.node_type) if t.value == "unit"], dtype=np.int32)
    w = np.ones(units.size) if unit_weight is None else unit_weight
    temp = p.scalar("behaviour.route_choice.stair_choice_temperature")
    S = tables.n_classes - 1
    load = np.zeros(net.n_arcs)
    for k, u in enumerate(units):
        if S == 0 or net.node_level[u] == 0:
            probs = {0: 1.0}
        else:
            t = tables.dist[0, 1:, u]
            ok = np.isfinite(t)
            if not ok.any():
                probs = {0: 1.0}
            else:
                e = np.where(ok, np.exp(-(t - t[ok].min()) / max(temp, 1e-9)), 0.0)
                probs = {c + 1: float(e[c] / e.sum()) for c in range(S) if e[c] > 0}
        for cls, pr in probs.items():
            v, guard = int(u), 0
            while not net.node_is_exit[v] and guard < 10_000:
                a = int(tables.next_arc[0, cls, v])
                if a < 0:
                    a = int(tables.next_arc[0, 0, v])
                if a < 0:
                    break
                load[a] += w[k] * pr
                v = int(net.arc_dst[a])
                guard += 1
    return load


def structural(
    net: SimNetwork,
    unit_weight: NDArray[np.float64] | None = None,
    params: Params | None = None,
) -> Structural:
    """Min-cut and flow-weighted load for ``net``."""
    value, cut = max_flow_min_cut(net)
    load = flow_load(net, unit_weight, params)
    return Structural(
        max_flow=value,
        min_cut_arcs=cut,
        load=load,
        clearance_time=load / np.maximum(net.arc_cap, 1e-9),
    )
