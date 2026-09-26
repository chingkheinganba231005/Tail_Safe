from __future__ import annotations

import numpy as np
import pytest

from tailsafe.analysis.bottlenecks import (
    NEVER,
    _counterfactual_spec,
    attribute_bottlenecks,
    candidates,
    queue_recurrence,
)
from tailsafe.analysis.structural import flow_load, max_flow_min_cut, structural
from tailsafe.building.model import Building, EdgeKind
from tailsafe.building.templates import generate
from tailsafe.risk.metrics import cvar
from tailsafe.scenarios.montecarlo import MCConfig, MCResult, run_monte_carlo
from tailsafe.scenarios.spec import ScenarioSpec, StairBlockage
from tailsafe.sim.cases import stair_tower
from tailsafe.sim.network import ARC_STAIR_DOWN, compile_network

SPEC = ScenarioSpec(
    name="blocked-A", share_65_plus=0.25, stair_blockages=[StairBlockage(stair="A")]
)


@pytest.fixture(scope="module")
def tower() -> Building:
    return generate("cruciform", storeys=16, flats_per_wing=2)


@pytest.fixture(scope="module")
def result(tower: Building) -> MCResult:
    return run_monte_carlo(tower, SPEC, MCConfig(n_runs=40, batch_size=40, workers=2))


def test_structural_single_stair() -> None:
    b = stair_tower(floors=6, stair_width=1.2, door_width=1.6, exit_width=3.0)
    net = compile_network(b)
    value, cut = max_flow_min_cut(net)
    stair_cap = net.arc_cap[net.arc_kind == ARC_STAIR_DOWN][0]
    assert value == pytest.approx(stair_cap)
    assert len(cut) == 1 and net.arc_kind[cut[0]] == ARC_STAIR_DOWN
    assert net.node_level[net.arc_dst[cut[0]]] == 0  # the bottom flight
    load = flow_load(net)
    assert load[cut[0]] == pytest.approx(6.0)  # every unit's route uses it
    st = structural(net)
    assert st.clearance_time[cut[0]] == pytest.approx(6.0 / stair_cap)


def test_queue_recurrence_and_candidates(tower: Building, result: MCResult) -> None:
    spots = queue_recurrence(result, "p95_occupant_time", threshold=5.0)
    assert spots and all(0.0 <= s.recurrence <= 1.0 for s in spots)
    assert spots[0].tail_person_seconds >= spots[-1].tail_person_seconds
    assert "Stair B" in spots[0].where  # Stair A is blocked, so B queues
    net = compile_network(tower)
    cands = candidates(tower, SPEC, net, spots)
    keys = {c.key for c in cands}
    assert {"stair:A", "stair:B", "stair_doors:B", "unblock:A"} <= keys
    assert any(c.kind == "exit" for c in cands)


def test_counterfactual_spec_keeps_random_slots(tower: Building) -> None:
    net = compile_network(tower)
    cands = {c.key: c for c in candidates(tower, SPEC, net, [])}
    unb = _counterfactual_spec(SPEC, cands["unblock:A"], 1.5)
    assert len(unb.stair_blockages) == 1 and unb.stair_blockages[0].time.value == NEVER
    wide = _counterfactual_spec(SPEC, cands["stair:B"], 1.5)
    flights = [e.id for e in tower.edges if e.kind == EdgeKind.STAIR and e.stair == "B"]
    assert all(wide.capacity_multipliers[f] == 1.5 for f in flights)


def test_attribution_ranks_and_is_exact(tower: Building, result: MCResult) -> None:
    table = attribute_bottlenecks(result, loss="p95_occupant_time", rerun_fraction=0.1, workers=2)
    ranking = table["ranking"]
    assert [r["rank"] for r in ranking] == list(range(1, len(ranking) + 1))
    top = ranking[0]
    assert top["key"] in ("unblock:A", "stair:B")
    assert top["delta_cvar"]["value"] < 0 and top["delta_cvar"]["hi"] <= 0
    assert table["headline"]
    # Adaptive re-running is exact: compare with re-running every scenario.
    row = next(r for r in ranking if r["key"] == "stair:B")
    net = compile_network(tower)
    cand = next(c for c in candidates(tower, SPEC, net, []) if c.key == "stair:B")
    full = run_monte_carlo(
        tower, _counterfactual_spec(SPEC, cand, 1.5), MCConfig(n_runs=40, batch_size=40, workers=2)
    )
    base = result.loss("p95_occupant_time")
    exact = cvar(np.minimum(full.loss("p95_occupant_time"), base)) - cvar(base)
    assert row["delta_cvar"]["value"] == pytest.approx(exact)
