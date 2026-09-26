from __future__ import annotations

import numpy as np
import pytest

from tailsafe.analysis.agreement import agreement_markdown, meso_micro_agreement
from tailsafe.building.builder import BuildingBuilder, rect
from tailsafe.building.model import Building, EdgeKind, LevelKind, NodeType
from tailsafe.building.templates import generate
from tailsafe.population.synth import Mode, PopulationConfig, TimeSlot, sample_population
from tailsafe.scenarios.spec import reference_spec
from tailsafe.sim import _micro_kernel as MK
from tailsafe.sim.cases import corridor_building, stair_tower, uniform_population
from tailsafe.sim.meso import SimScenario, run_meso, stair_blockage
from tailsafe.sim.micro import (
    _clip_centred,
    _facing_side,
    _shared_boundary,
    compile_geometry,
    fundamental_diagram,
    micro_problems,
    run_micro,
)
from tailsafe.sim.network import ARC_FLAT, compile_network


def room_corridor(length: float = 30.0, door: float = 1.0, width: float = 2.0) -> Building:
    """A 10 × 10 m room, a corridor of ``length`` × ``width`` and an exit door at its end."""
    bld = BuildingBuilder(id="case-room-corridor", name="Room and corridor", typology="custom")
    bld.level(0, elevation=0.0, height=3.0, kind=LevelKind.GROUND)
    bld.node("room", NodeType.UNIT, 0, polygon=rect(0, 0, 10, 10), unit_type="case")
    bld.node(
        "corridor",
        NodeType.CORRIDOR,
        0,
        polygon=rect(10, 5 - width / 2, 10 + length, 5 + width / 2),
    )
    bld.node("exit", NodeType.EXIT, 0, at=(10.5 + length, 5.0))
    bld.edge("room", "corridor", EdgeKind.FLAT, width=width, length=5.0)
    bld.edge(
        "corridor",
        "exit",
        EdgeKind.DOOR,
        width=door,
        length=0.5,
        opening=((10 + length, 5 - door / 2), (10 + length, 5 + door / 2)),
    )
    return bld.build()


def test_rectangle_helpers() -> None:
    seg = _shared_boundary((0, 10, 0, 10), (10, 40, 4, 6))
    assert seg == ((10, 4), (10, 6))
    assert _shared_boundary((0, 1, 0, 1), (2, 3, 0, 1)) is None
    (x0, y0), (x1, y1) = _clip_centred(((0.0, 0.0), (0.0, 6.0)), 2.0)
    assert (x0, y0, x1, y1) == (0.0, 2.0, 0.0, 4.0)
    (a, b) = _facing_side((0, 3, 0, 6), 5.0, 3.0, 2.0)
    assert a == (3, 2.0) and b == (3, 4.0)


def test_geometry_of_a_template() -> None:
    b = generate("cruciform", storeys=4, flats_per_wing=2)
    assert micro_problems(b) == []
    net = compile_network(b)
    g = compile_geometry(net)
    for a in range(net.n_arcs):
        src = int(net.arc_src[a])
        if net.arc_kind[a] == ARC_FLAT:
            # the doorway lies on the source rectangle and its normal points out of it
            mx = (g.p_x0[a] + g.p_x1[a]) / 2
            my = (g.p_y0[a] + g.p_y1[a]) / 2
            assert g.n_x0[src] - 1e-6 <= mx <= g.n_x1[src] + 1e-6
            assert g.n_y0[src] - 1e-6 <= my <= g.n_y1[src] + 1e-6
            cx = (g.n_x0[src] + g.n_x1[src]) / 2
            cy = (g.n_y0[src] + g.n_y1[src]) / 2
            assert (mx - cx) * g.p_nx[a] + (my - cy) * g.p_ny[a] > 0
            assert abs(g.p_nx[a]) + abs(g.p_ny[a]) == pytest.approx(1.0)
        else:
            assert g.arc_lanes[a] == 2  # 1.1 m stairs, 0.55 m per lane


def test_graph_only_buildings_are_refused() -> None:
    b = corridor_building()
    assert micro_problems(b)
    pop = uniform_population(b, 1)
    with pytest.raises(ValueError, match="microscopically"):
        run_micro(b, pop)


def test_single_walker_matches_walking_distance() -> None:
    b = room_corridor(length=30.0)
    pop = uniform_population(b, 1, h_speed=1.2)
    res = run_micro(b, pop)
    # From somewhere in the room to the room's doorway, then 30 m of corridor.
    t = float(res.agent_exit[0])
    assert 30.0 / 1.2 < t < (30.0 + 15.0) / 1.2
    assert res.agent_walked.all() and res.summary()["not_out"] == 0


def test_door_limits_the_flow() -> None:
    # A wide room doorway and corridor, so the exit door is the constriction.
    def case(door: float) -> Building:
        return room_corridor(length=10.0, door=door, width=6.0)

    wide = run_micro(case(1.8), uniform_population(case(1.8), 60))
    narrow = run_micro(case(0.8), uniform_population(case(0.8), 60))
    assert np.isfinite(narrow.agent_exit).all() and np.isfinite(wide.agent_exit).all()
    assert narrow.total_time() > 1.2 * wide.total_time()
    # A 0.8 m door lets people through at a plausible rate (0.3 to 2 persons/s).
    first, last = np.sort(narrow.agent_exit)[[5, -1]]
    rate = (60 - 6) / (last - first)
    assert 0.3 < rate < 2.0


def test_stair_tower_everyone_out_and_close_to_meso() -> None:
    b = stair_tower(floors=6)
    pop = uniform_population(b, 8)
    me = run_meso(b, pop)
    mi = run_micro(b, pop, meso=me)
    assert mi.summary()["not_out"] == 0
    assert mi.total_time() == pytest.approx(me.total_time, rel=0.35)
    # Frames: people start on their floors and are all out at the end.
    assert set(np.unique(mi.frame_level[0])) <= set(range(1, 7))
    assert (mi.frame_level[-1] == -1).sum() >= pop.n_agents - 5


def test_template_night_with_blocked_stair() -> None:
    b = generate("cruciform", storeys=8, flats_per_wing=2)
    pop = sample_population(
        b, PopulationConfig(time_slot=TimeSlot.WEEKEND_NIGHT, share_65_plus=0.3), seed=4
    )
    sc = SimScenario(blockages=stair_blockage(b, "A", 120.0))
    me = run_meso(b, pop, sc)
    a = run_micro(b, pop, sc, meso=me, seed=0, index=3)
    again = run_micro(b, pop, sc, meso=me, seed=0, index=3)
    np.testing.assert_array_equal(a.agent_exit, again.agent_exit)  # deterministic
    assert a.summary()["not_out"] == 0
    # Households the micro engine does not walk take their times from the meso run.
    waiting = np.isin(pop.group_mode[pop.agent_group], [Mode.WAIT_LIFT, Mode.WAIT_RESCUE])
    np.testing.assert_array_equal(a.agent_exit[waiting], me.group_exit[pop.agent_group[waiting]])
    assert not a.agent_walked[waiting].any()
    states = set(np.unique(a.agent_state).tolist())
    assert states <= {MK.S_EXITED, MK.S_EXCLUDED}


def test_fundamental_diagram_is_reasonable() -> None:
    fd = fundamental_diagram((0.25, 1.0, 2.0, 3.5), t_warm=15.0, t_meas=15.0)
    speeds = [r["speed"] for r in fd]
    assert speeds[0] == pytest.approx(fd[0]["hydraulic_speed"], rel=0.1)  # free walking
    assert speeds[0] > speeds[1] > speeds[2] > speeds[3] > 0.0
    assert speeds[-1] < 0.35 * speeds[0]


@pytest.mark.slow
def test_meso_micro_agreement_small_tower() -> None:
    b = generate("cruciform", storeys=10, flats_per_wing=3)
    res = meso_micro_agreement(b, reference_spec().model_copy(update={"fire_level": 6}), 8)
    assert res["not_out_total"] == 0
    last = res["summary"]["last_s"]
    assert abs(last["relative_bias"]) < 0.1 and last["correlation"] > 0.9
    assert abs(res["summary"]["p95_s"]["relative_bias"]) < 0.3
    assert "| Time the last walker gets out |" in agreement_markdown(res)
