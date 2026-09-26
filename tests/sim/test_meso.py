from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from tailsafe.building.model import Building, EdgeKind
from tailsafe.config import get_params
from tailsafe.population.synth import (
    Mode,
    Population,
    PopulationConfig,
    TimeSlot,
    sample_population,
)
from tailsafe.sim.cases import stair_tower, uniform_population
from tailsafe.sim.meso import Blockage, SimConfig, SimScenario, run_meso
from tailsafe.sim.network import ARC_STAIR_DOWN, SimNetwork


@pytest.fixture(scope="module")
def night_pop(slab10: Building) -> Population:
    return sample_population(slab10, PopulationConfig(time_slot=TimeSlot.WEEKEND_NIGHT), seed=11)


def test_deterministic(slab10_net: SimNetwork, night_pop: Population) -> None:
    a = run_meso(slab10_net, night_pop)
    b = run_meso(slab10_net, night_pop)
    np.testing.assert_array_equal(a.group_exit, b.group_exit)
    np.testing.assert_array_equal(a.arc_queue_integral, b.arc_queue_integral)


def test_everyone_gets_out(slab10_net: SimNetwork, night_pop: Population) -> None:
    res = run_meso(slab10_net, night_pop)
    assert res.n_not_evacuated == 0
    assert np.isfinite(res.total_time)
    walkers = ~res.group_rescued  # rescued households leave when the fire service arrives
    assert (res.group_exit[walkers] >= night_pop.group_premovement[walkers] - 1e-9).all()
    s = res.summary()
    assert s["occupants"] == night_pop.n_agents
    assert s["p50_exit_s"] <= s["p95_exit_s"] <= s["total_time_s"]
    clear = res.floor_clearance()
    assert set(clear) == set(np.unique(night_pop.group_level).tolist())


def stair_arcs(net: SimNetwork, stair: str) -> np.ndarray:
    s = net.stair_ids.index(stair)
    return np.flatnonzero((net.arc_stair == s) & (net.arc_kind == ARC_STAIR_DOWN))


def test_blocking_a_stair_diverts_everyone_and_slows_evacuation(
    slab10: Building, slab10_net: SimNetwork, night_pop: Population
) -> None:
    base = run_meso(slab10_net, night_pop)
    flights = [e.id for e in slab10.edges if e.kind == EdgeKind.STAIR and e.stair == "A"]
    blocked = run_meso(
        slab10_net, night_pop, SimScenario(blockages=tuple(Blockage(e, 0.0) for e in flights))
    )
    assert blocked.arc_entries[stair_arcs(slab10_net, "A")].sum() == 0
    assert blocked.n_not_evacuated == 0
    assert blocked.self_evacuation_time >= base.self_evacuation_time - 1e-6
    p95 = [r.exit_time_quantile(0.95) for r in (base, blocked)]
    assert p95[1] > p95[0]


def test_blockage_discovered_later_still_evacuates(
    slab10: Building, slab10_net: SimNetwork, night_pop: Population
) -> None:
    flights = [e.id for e in slab10.edges if e.kind == EdgeKind.STAIR and e.stair == "B"]
    res = run_meso(
        slab10_net, night_pop, SimScenario(blockages=tuple(Blockage(e, 240.0) for e in flights))
    )
    assert res.n_not_evacuated == 0


def test_wider_stairs_are_never_slower() -> None:
    narrow = stair_tower(floors=8, stair_width=1.0)
    wide = stair_tower(floors=8, stair_width=1.6)
    t = []
    for b in (narrow, wide):
        pop = uniform_population(b, 30)
        t.append(run_meso(b, pop).total_time)
    assert t[1] < t[0]


def test_phased_release(slab10_net: SimNetwork, night_pop: Population) -> None:
    release = {lv: 900.0 for lv in range(1, 5)}
    res = run_meso(slab10_net, night_pop, SimScenario(phased_release=release))
    low = np.isin(night_pop.group_level, list(release)) & ~res.group_rescued
    assert (res.group_left_floor[low] >= 900.0 - 1e-6).all()


def test_stair_assignment(slab10_net: SimNetwork, night_pop: Population) -> None:
    assign = {lv: "B" for lv in range(1, 10)}
    res = run_meso(slab10_net, night_pop, SimScenario(stair_assignment=assign))
    upper = night_pop.group_level >= 1
    assert (res.group_class[upper] == 2).all()
    assert res.arc_entries[stair_arcs(slab10_net, "A")].sum() == 0


def test_evacuation_lifts_carry_waiting_groups(slab10: Building, slab10_net: SimNetwork) -> None:
    pop = sample_population(
        slab10, PopulationConfig(time_slot=TimeSlot.WEEKDAY_DAY, evacuation_lifts=True), seed=3
    )
    waiting = pop.group_mode == Mode.WAIT_LIFT
    assert waiting.any()
    lifts = tuple(lf.id for lf in slab10.lifts)
    res = run_meso(slab10_net, pop, SimScenario(evacuation_lifts=lifts))
    assert res.lift_trips > 0
    assert res.n_not_evacuated == 0
    delay = get_params().scalar("lifts.evacuation_mode_delay")
    assert (res.group_left_floor[waiting] >= delay).all()
    assert not res.group_rescued[waiting].any()


def test_without_lifts_waiting_groups_are_rescued(slab10: Building, slab10_net: SimNetwork) -> None:
    pop = sample_population(
        slab10, PopulationConfig(time_slot=TimeSlot.WEEKDAY_DAY, evacuation_lifts=True), seed=3
    )
    res = run_meso(slab10_net, pop, SimScenario(evacuation_lifts=()))
    waiting = pop.group_mode == Mode.WAIT_LIFT
    assert res.group_rescued[waiting].all()


def test_rescue_timing(slab10: Building, slab10_net: SimNetwork) -> None:
    p = get_params()
    pop = sample_population(slab10, PopulationConfig(), seed=5)
    pop = replace(
        pop,
        group_mode=np.where(np.arange(pop.n_groups) == 0, Mode.WAIT_RESCUE, Mode.WALK).astype(
            np.int8
        ),
    )
    res = run_meso(slab10_net, pop, SimScenario(rescue_start=600.0, rescue_teams=1))
    lv = pop.group_level[0]
    expect = (
        600.0
        + lv * p.scalar("rescue.climb_time_per_floor")
        + p.scalar("rescue.handling_time")
        + lv * p.scalar("rescue.carry_down_time_per_floor")
    )
    assert res.group_rescued[0]
    assert res.group_exit[0] == pytest.approx(expect)


def test_refuge_rest_adds_dwell() -> None:
    from tailsafe.building.templates import generate

    b = generate("slab", storeys=27, flats_per_side=2)  # refuge floor at 20/F
    pop = uniform_population(b, 1)
    top = int(np.argmax(pop.group_level))
    rest = pop.group_refuge_rest.copy()
    rest[top] = 200.0
    base = run_meso(b, pop)
    rested = run_meso(b, replace(pop, group_refuge_rest=rest))
    extra = rested.group_exit[top] - base.group_exit[top]
    assert 200.0 <= extra <= 260.0


def test_counter_flow_detour_is_slower(slab10: Building, slab10_net: SimNetwork) -> None:
    pop = uniform_population(slab10, 1)
    g = int(np.argmax(pop.group_level == 3))
    wp = list(pop.group_waypoint)
    wp[g] = "L07.unit.02"
    dwell = pop.group_waypoint_dwell.copy()
    dwell[g] = 30.0
    base = run_meso(slab10_net, pop)
    detour = run_meso(slab10_net, replace(pop, group_waypoint=wp, group_waypoint_dwell=dwell))
    assert detour.group_exit[g] > base.group_exit[g] + 30.0
    ups = slab10_net.arc_kind == 2
    assert detour.arc_entries[ups].sum() > 0 == base.arc_entries[ups].sum()


@dataclass
class _Field:
    dt: float
    speed_multiplier: np.ndarray
    fed_rate: np.ndarray


def test_hazard_slows_and_incapacitates(slab10_net: SimNetwork, night_pop: Population) -> None:
    n = slab10_net.n_nodes
    base = run_meso(slab10_net, night_pop)
    slow = _Field(10.0, np.full((1, n), 0.5, np.float32), np.zeros((1, n), np.float32))
    res = run_meso(slab10_net, night_pop, SimScenario(hazard=slow))
    assert res.self_evacuation_time > base.self_evacuation_time
    toxic = _Field(10.0, np.ones((1, n), np.float32), np.full((1, n), 1 / 300.0, np.float32))
    res = run_meso(slab10_net, night_pop, SimScenario(hazard=toxic))
    assert res.n_not_evacuated > 0
    assert (res.group_fed[~res.evacuated] >= 1.0 - 1e-6).all()
    assert (res.group_fed[res.evacuated] < 1.0).all()


def test_stranded_groups_wait_for_rescue(slab10: Building, slab10_net: SimNetwork) -> None:
    exits = [e.id for e in slab10.edges if slab10.node_by_id[e.target].type == "exit"]
    pop = uniform_population(slab10, 1)
    res = run_meso(
        slab10_net,
        pop,
        SimScenario(blockages=tuple(Blockage(e) for e in exits), rescue_start=300.0),
        SimConfig(t_max=40_000),
    )
    assert res.group_rescued.all()


def test_series_recording(slab10_net: SimNetwork, night_pop: Population) -> None:
    res = run_meso(slab10_net, night_pop, config=SimConfig(record_series=True, record_interval=5.0))
    assert res.series is not None and res.series_times is not None
    assert res.series.shape == (slab10_net.n_arcs, res.series_times.size)
    assert res.series.max() == pytest.approx(res.arc_max_queue.max(), rel=0.5)
    tops = res.top_queues(3)
    assert tops and tops[0]["queue_person_seconds"] >= tops[-1]["queue_person_seconds"]
