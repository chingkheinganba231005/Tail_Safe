"""Route tables for the simulators.

Groups follow *next-hop tables* rather than stored paths: for every network
**state** (which edges are blocked), route **class** (no stair preference, or a
preferred stair) and node, the table gives the arc to take towards the nearest
exit by estimated travel time. Tables for waypoint targets (a relative's flat, a
lift lobby, a refuge area) are computed the same way.

When a blockage appears, groups keep using the table they know until they reach
a node whose next arc is blocked; they then switch to the table of the current
state ("discover on arrival").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from tailsafe.config import Params, get_params
from tailsafe.sim.network import ARC_FLAT, ARC_STAIR_DOWN, SimNetwork


@dataclass(frozen=True)
class RoutingTables:
    """Next-hop tables. ``-1`` means "at an exit" or "no route"."""

    next_arc: NDArray[np.int32]  # [K states, C classes, N nodes]
    dist: NDArray[np.float64]  # [K, C, N] estimated seconds to the nearest exit
    state_time: NDArray[np.float64]  # [K] time from which each state applies
    wp_targets: NDArray[np.int32]  # [T] waypoint node indices
    wp_next: NDArray[np.int32]  # [T, N]

    @property
    def n_states(self) -> int:
        """Number of network states (blockage epochs)."""
        return int(self.state_time.size)

    @property
    def n_classes(self) -> int:
        """Number of route classes (1 + number of stairs)."""
        return int(self.next_arc.shape[1])

    def waypoint_index(self, node: int) -> int:
        """Row of ``wp_next`` for waypoint node ``node``."""
        hits = np.flatnonzero(self.wp_targets == node)
        if hits.size == 0:
            raise KeyError(node)
        return int(hits[0])


def arc_travel_cost(net: SimNetwork, params: Params | None = None) -> NDArray[np.float64]:
    """Estimated free-flow traversal time of each arc (s), for route choice only."""
    p = params or get_params()
    rc = "behaviour.route_choice."
    speed = np.where(
        net.arc_kind == ARC_FLAT,
        p.scalar(rc + "nominal_speed_flat"),
        np.where(
            net.arc_kind == ARC_STAIR_DOWN,
            p.scalar(rc + "nominal_speed_stair_down"),
            p.scalar(rc + "nominal_speed_stair_up"),
        ),
    )
    out: NDArray[np.float64] = net.arc_len / speed
    return out


def _next_hops(
    net: SimNetwork,
    cost: NDArray[np.float64],
    usable: NDArray[np.bool_],
    targets: Sequence[int] | NDArray[np.int32],
    min_only: bool,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Shortest-path next arcs towards ``targets`` over usable arcs.

    Runs Dijkstra from the targets on the reversed graph. With ``min_only`` the
    result is one row (nearest target); otherwise one row per target.
    """
    n = net.n_nodes
    idx = np.flatnonzero(usable)
    src, dst, c = net.arc_src[idx], net.arc_dst[idx], cost[idx]
    # Keep only the cheapest arc between each ordered node pair.
    order = np.lexsort((c, dst, src))
    src, dst, c, idx = src[order], dst[order], c[order], idx[order]
    first = np.ones(src.size, dtype=bool)
    first[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])
    src, dst, c, idx = src[first], dst[first], c[first], idx[first]
    keys = src.astype(np.int64) * n + dst  # sorted, unique
    rev = csr_matrix((c, (dst, src)), shape=(n, n))
    tgt = np.asarray(targets, dtype=np.int32)
    rows = 1 if min_only else tgt.size
    if tgt.size == 0:
        return np.full((rows, n), -1, dtype=np.int32), np.full((rows, n), np.inf)
    if min_only:
        dist, pred, _ = dijkstra(
            rev, directed=True, indices=tgt, min_only=True, return_predecessors=True
        )
    else:
        dist, pred = dijkstra(rev, directed=True, indices=tgt, return_predecessors=True)
    dist = np.asarray(dist, dtype=np.float64).reshape(rows, n)
    pred = np.asarray(pred).reshape(rows, n)
    nxt = np.full((rows, n), -1, dtype=np.int32)
    has = pred >= 0
    r_i, u_i = np.nonzero(has)
    q = u_i.astype(np.int64) * n + pred[r_i, u_i]
    pos = np.searchsorted(keys, q)
    nxt[r_i, u_i] = idx[pos].astype(np.int32)
    return nxt, dist


class Router:
    """Computes and caches route tables for one network.

    Exit tables are cached per (blocked-arc set, class) and waypoint rows per
    target node, so Monte Carlo workers pay for each distinct table once.
    """

    def __init__(self, net: SimNetwork, params: Params | None = None) -> None:
        self.net = net
        self.cost = arc_travel_cost(net, params)
        self.exits = np.flatnonzero(net.node_is_exit).astype(np.int32)
        self._exit_cache: dict[
            tuple[frozenset[int], int], tuple[NDArray[np.int32], NDArray[np.float64]]
        ] = {}
        self._wp_cache: dict[int, NDArray[np.int32]] = {}

    def exit_table(
        self, blocked: frozenset[int], cls: int
    ) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
        """Next arcs and times to the nearest exit with ``blocked`` arcs removed."""
        key = (blocked, cls)
        if key not in self._exit_cache:
            usable = np.ones(self.net.n_arcs, dtype=bool)
            if blocked:
                usable[np.fromiter(blocked, dtype=np.intp)] = False
            if cls > 0:
                usable &= (self.net.arc_stair < 0) | (self.net.arc_stair == cls - 1)
            nxt, d = _next_hops(self.net, self.cost, usable, self.exits, min_only=True)
            self._exit_cache[key] = (nxt[0], d[0])
        return self._exit_cache[key]

    def waypoint_rows(self, targets: Sequence[int]) -> NDArray[np.int32]:
        """Next arcs towards each target node (unblocked network)."""
        missing = sorted({int(t) for t in targets} - self._wp_cache.keys())
        if missing:
            usable = np.ones(self.net.n_arcs, dtype=bool)
            nxt, _ = _next_hops(self.net, self.cost, usable, missing, min_only=False)
            for t, row in zip(missing, nxt, strict=True):
                self._wp_cache[t] = row
        if not targets:
            return np.zeros((0, self.net.n_nodes), dtype=np.int32)
        return np.stack([self._wp_cache[int(t)] for t in targets])

    def tables(
        self,
        blockages: Sequence[tuple[float, Sequence[int]]] = (),
        waypoints: Sequence[int] = (),
    ) -> RoutingTables:
        """Route tables for a scenario.

        ``blockages`` are ``(time, arc indices)`` events; state ``k`` has the arcs
        of the first ``k`` events (in time order) blocked. Class 0 routes by the
        fastest route; class ``s + 1`` may only use staircase ``s`` (other
        staircases' flights are unusable).
        """
        events = sorted(blockages, key=lambda e: e[0])
        K = len(events) + 1
        C = len(self.net.stair_ids) + 1
        N = self.net.n_nodes
        next_arc = np.empty((K, C, N), dtype=np.int32)
        dist = np.empty((K, C, N))
        blocked: set[int] = set()
        for k in range(K):
            if k > 0:
                blocked.update(int(a) for a in events[k - 1][1])
            frozen = frozenset(blocked)
            for c in range(C):
                next_arc[k, c], dist[k, c] = self.exit_table(frozen, c)
        wp = np.array(sorted({int(w) for w in waypoints}), dtype=np.int32)
        return RoutingTables(
            next_arc=next_arc,
            dist=dist,
            state_time=np.array([-np.inf] + [float(t) for t, _ in events]),
            wp_targets=wp,
            wp_next=self.waypoint_rows(list(wp)),
        )


def build_routing(
    net: SimNetwork,
    *,
    blockages: Sequence[tuple[float, Sequence[int]]] = (),
    waypoints: Sequence[int] = (),
    params: Params | None = None,
) -> RoutingTables:
    """One-off convenience wrapper around :class:`Router`."""
    return Router(net, params).tables(blockages, waypoints)


def choose_stair_class(
    tables: RoutingTables,
    home: NDArray[np.int32],
    u: NDArray[np.float64],
    temperature: float,
    needs_stair: NDArray[np.bool_] | None = None,
) -> NDArray[np.int32]:
    """Route class (preferred stair + 1) per group by a logit on travel time.

    ``P(stair s) ∝ exp(-(T_s - T_min) / temperature)`` where ``T_s`` is the
    estimated time from the group's home to an exit using only stair ``s``.
    Draws use the group's uniform ``u`` (inverse CDF), so choices are coupled
    across scenarios. Groups that need no stair (``needs_stair`` False, e.g. on
    the ground floor) get class 0.
    """
    S = tables.n_classes - 1
    if S == 0:
        return np.zeros(home.size, dtype=np.int32)
    t = tables.dist[0, 1:, :][:, home].T  # [G, S]
    tmin = t.min(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        w = np.where(np.isfinite(t), np.exp(-(t - tmin) / max(temperature, 1e-9)), 0.0)
    total = w.sum(axis=1, keepdims=True)
    ok = total[:, 0] > 0
    probs = np.where(ok[:, None], w / np.where(total > 0, total, 1.0), 0.0)
    cum = np.cumsum(probs, axis=1)
    choice = (u[:, None] >= cum).sum(axis=1)
    choice = np.minimum(choice, S - 1)
    out = np.where(ok, choice + 1, 0).astype(np.int32)
    if needs_stair is not None:
        out[~needs_stair] = 0
    return out
