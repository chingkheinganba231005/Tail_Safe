"""Meso engine vs hydraulic hand calculations (see docs/validation.md)."""

from __future__ import annotations

import numpy as np
import pytest

from tailsafe.config import get_params
from tailsafe.sim._kernel import _speed_factor
from tailsafe.sim.cases import corridor_building, stair_tower, uniform_population
from tailsafe.sim.meso import SimConfig, run_meso
from tailsafe.sim.network import ARC_STAIR_DOWN, compile_network

P = get_params()
FS_H = P.scalar("movement.hydraulic.max_specific_flow_horizontal")
FS_S = P.scalar("movement.hydraulic.max_specific_flow_stair")
BL_DOOR = P.scalar("movement.boundary_layer.door")
BL_STAIR = P.scalar("movement.boundary_layer.stair")
V_H = 0.85 * P.scalar("movement.hydraulic.k_horizontal")  # unimpeded hydraulic speeds
V_S = 0.85 * P.scalar("movement.hydraulic.k_stair")


@pytest.mark.parametrize("n,exit_width", [(100, 1.0), (300, 0.9), (60, 1.6)])
def test_corridor_matches_hydraulic_estimate(n: int, exit_width: float) -> None:
    """T = walk time to the door + N / door flow (door is the only constriction)."""
    b = corridor_building(length=30.0, width=2.4, exit_width=exit_width)
    pop = uniform_population(b, n, h_speed=V_H)
    res = run_meso(b, pop)
    door_flow = FS_H * (exit_width - 2 * BL_DOOR)
    expected = 31.0 / V_H + n / door_flow
    assert res.total_time == pytest.approx(expected, rel=0.03)


@pytest.mark.parametrize("floors,per_floor", [(10, 30), (20, 15), (6, 60)])
def test_single_stair_matches_hydraulic_estimate(floors: int, per_floor: int) -> None:
    """T = first arrival from 1/F + P / stair flow, the stair being saturated."""
    b = stair_tower(floors=floors, stair_width=1.2, door_width=1.2)
    net = compile_network(b)
    pop = uniform_population(b, per_floor, h_speed=V_H, down_speed=V_S)
    res = run_meso(net, pop)
    stair_flow = FS_S * (1.2 - 2 * BL_STAIR)
    flight = net.arc_len[net.arc_kind == ARC_STAIR_DOWN][0]
    first = 2.0 / V_H + flight / V_S + 1.0 / V_H  # 1/F room -> landing -> G/F -> exit
    expected = first + floors * per_floor / stair_flow
    assert res.total_time == pytest.approx(expected, rel=0.03)


def test_saturated_stair_flow_equals_capacity() -> None:
    b = stair_tower(floors=12)
    net = compile_network(b)
    pop = uniform_population(b, 40, h_speed=V_H, down_speed=V_S)
    res = run_meso(net, pop)
    exits = np.sort(res.agent_exit)
    mid = exits[(exits > 0.2 * exits[-1]) & (exits < 0.8 * exits[-1])]
    measured = (mid.size - 1) / (mid[-1] - mid[0])
    cap = net.arc_cap[net.arc_kind == ARC_STAIR_DOWN][0]
    assert measured == pytest.approx(cap, rel=0.03)


@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
def test_merge_deference_ratio(ratio: float) -> None:
    """With both streams queueing, the floor gets `ratio` of the stair below."""
    params = P.with_overrides({"movement.merge.floor_deference_ratio": ratio})
    b = stair_tower(floors=2, door_width=1.6)
    net = compile_network(b, params)
    pop = uniform_population(b, 150, h_speed=V_H, down_speed=V_S)
    res = run_meso(net, pop, params=params)
    exits = res.agent_exit
    from_floor1 = pop.group_level == 1
    t_end = min(exits[from_floor1].max(), exits[~from_floor1].max())
    window = (exits > 20.0) & (exits < t_end - 5.0)
    share = (window & from_floor1).sum() / window.sum()
    assert share == pytest.approx(ratio, abs=0.04)


def test_speed_density_relation() -> None:
    a = P.scalar("movement.hydraulic.speed_density_a")
    d0 = P.scalar("movement.hydraulic.free_flow_density")
    assert _speed_factor(0.3, a, d0, 0.05) == 1.0
    assert _speed_factor(1.5, a, d0, 0.05) == pytest.approx((1 - a * 1.5) / (1 - a * d0))
    # Capped at the flow-maximising density 1/(2a): queues represent denser crowds.
    crit = (1 - 0.5) / (1 - a * d0)
    assert _speed_factor(3.5, a, d0, 0.05) == pytest.approx(crit)
    assert _speed_factor(1.0 / (2 * a), a, d0, 0.05) == pytest.approx(crit)


def test_time_step_convergence() -> None:
    """Halving the time step changes the stair result by well under 1%."""
    b = stair_tower(floors=8)
    pop = uniform_population(b, 25, h_speed=V_H, down_speed=V_S)
    coarse = run_meso(b, pop, config=SimConfig(dt=1.0)).total_time
    fine = run_meso(b, pop, config=SimConfig(dt=0.25)).total_time
    assert coarse == pytest.approx(fine, rel=0.01)
