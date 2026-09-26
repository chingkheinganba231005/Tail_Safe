from __future__ import annotations

import numpy as np
import pytest

from tailsafe.building.model import Building
from tailsafe.config import get_params
from tailsafe.sim.network import ARC_FLAT, ARC_STAIR_DOWN, ARC_STAIR_UP, SimNetwork
from tailsafe.sim.routing import Router, choose_stair_class


def follow(net: SimNetwork, table: np.ndarray, start: int, limit: int = 10_000) -> list[int]:
    path, v = [], start
    while not net.node_is_exit[v]:
        a = int(table[v])
        assert a >= 0, f"no route from {net.node_ids[v]}"
        path.append(a)
        v = int(net.arc_dst[a])
        assert len(path) < limit
    return path


def test_arc_expansion_and_csr(slab10_net: SimNetwork) -> None:
    net = slab10_net
    assert not net.node_is_exit[net.arc_src].any(), "exits are sinks"
    for v in range(net.n_nodes):
        ins = net.in_arc[net.in_ptr[v] : net.in_ptr[v + 1]]
        outs = net.out_arc[net.out_ptr[v] : net.out_ptr[v + 1]]
        assert (net.arc_dst[ins] == v).all() and (net.arc_src[outs] == v).all()
    both = net.arc_rev >= 0
    assert (net.arc_rev[net.arc_rev[both]] == np.flatnonzero(both)).all()
    downs = net.arc_kind == ARC_STAIR_DOWN
    assert (net.node_level[net.arc_src[downs]] > net.node_level[net.arc_dst[downs]]).all()
    ups = net.arc_kind == ARC_STAIR_UP
    assert ups.sum() == downs.sum()


def test_capacities_follow_hydraulic_model(slab10_net: SimNetwork) -> None:
    p = get_params()
    net = slab10_net
    stair = net.arc_kind != ARC_FLAT
    expect = p.scalar("movement.hydraulic.max_specific_flow_stair") * (
        net.arc_width[stair] - 2 * p.scalar("movement.boundary_layer.stair")
    )
    np.testing.assert_allclose(net.arc_cap[stair], expect)
    doors = net.arc_is_door
    expect = p.scalar("movement.hydraulic.max_specific_flow_horizontal") * (
        net.arc_width[doors] - 2 * p.scalar("movement.boundary_layer.door")
    )
    np.testing.assert_allclose(net.arc_cap[doors], expect)
    assert (net.arc_store >= 2.0).all()


def test_every_unit_routes_to_an_exit(slab10: Building, slab10_net: SimNetwork) -> None:
    tb = Router(slab10_net).tables()
    for u in slab10.units:
        for c in range(tb.n_classes):
            follow(slab10_net, tb.next_arc[0, c], slab10_net.node(u.id))


def test_class_tables_use_only_their_stair(slab10_net: SimNetwork) -> None:
    net = slab10_net
    tb = Router(net).tables()
    start = net.node("L09.unit.01")
    for s in range(len(net.stair_ids)):
        path = follow(net, tb.next_arc[0, s + 1], start)
        used = {int(net.arc_stair[a]) for a in path if net.arc_stair[a] >= 0}
        assert used == {s}


def test_blocked_state_avoids_blocked_arcs(slab10_net: SimNetwork) -> None:
    net = slab10_net
    stair_a = np.flatnonzero(net.arc_stair == 0)
    tb = Router(net).tables(blockages=[(120.0, stair_a)])
    assert tb.n_states == 2 and tb.state_time[1] == 120.0
    for u in range(net.n_nodes):
        if net.node_level[u] > 0 and not net.node_is_exit[u]:
            path = follow(net, tb.next_arc[1, 0], u)
            assert not set(path) & set(stair_a.tolist())


def test_waypoint_rows_reach_targets(slab10_net: SimNetwork) -> None:
    net = slab10_net
    target = net.node("L05.unit.03")
    tb = Router(net).tables(waypoints=[target])
    row = tb.wp_next[tb.waypoint_index(target)]
    v, steps = net.node("L09.unit.07"), 0
    while v != target:
        v = int(net.arc_dst[row[v]])
        steps += 1
        assert steps < 500


def test_router_caches_tables(slab10_net: SimNetwork) -> None:
    r = Router(slab10_net)
    a = r.tables(waypoints=[5])
    b = r.tables(waypoints=[5])
    assert a.next_arc is not b.next_arc  # fresh arrays...
    np.testing.assert_array_equal(a.next_arc, b.next_arc)  # ...same content
    assert len(r._exit_cache) == len(slab10_net.stair_ids) + 1


def test_stair_choice_logit() -> None:
    from tailsafe.sim.routing import RoutingTables

    dist = np.zeros((1, 3, 2))
    dist[0, 1, :] = [100.0, 100.0]  # stair A
    dist[0, 2, :] = [130.0, 100.0]  # stair B: 30 s worse from node 0, equal from node 1
    tb = RoutingTables(
        next_arc=np.zeros((1, 3, 2), dtype=np.int32),
        dist=dist,
        state_time=np.array([-np.inf]),
        wp_targets=np.zeros(0, dtype=np.int32),
        wp_next=np.zeros((0, 2), dtype=np.int32),
    )
    u = (np.arange(20_000) + 0.5) / 20_000
    home = np.zeros(u.size, dtype=np.int32)
    cls = choose_stair_class(tb, home, u, temperature=30.0)
    p_a = np.mean(cls == 1)
    assert p_a == pytest.approx(1 / (1 + np.exp(-1.0)), abs=0.01)
    cls = choose_stair_class(tb, np.ones(u.size, dtype=np.int32), u, temperature=30.0)
    assert np.mean(cls == 1) == pytest.approx(0.5, abs=0.01)
    none = choose_stair_class(tb, home, u, 30.0, needs_stair=np.zeros(u.size, dtype=bool))
    assert (none == 0).all()
