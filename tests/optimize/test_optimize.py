from __future__ import annotations

import numpy as np
import pytest

from tailsafe.building.model import Building
from tailsafe.building.templates import generate
from tailsafe.config import get_params
from tailsafe.optimize.cmaes import cmaes
from tailsafe.optimize.plan import InterventionPlan
from tailsafe.optimize.search import (
    Objective,
    OptimizeConfig,
    _bands,
    confirm,
    merge,
    optimize,
    screening_plans,
)
from tailsafe.population.synth import Mode
from tailsafe.scenarios.montecarlo import MCConfig, run_monte_carlo
from tailsafe.scenarios.sampler import ScenarioSampler, apply_wardens, scenario_uniforms
from tailsafe.scenarios.spec import HazardSpec, ScenarioSpec, reference_spec


@pytest.fixture(scope="module")
def tower() -> Building:
    return generate("cruciform", storeys=16, flats_per_wing=2)


def test_plan_apply_and_describe(tower: Building) -> None:
    spec = ScenarioSpec(hazard=HazardSpec())
    plan = InterventionPlan(
        evacuation_lifts=True,
        lift_priority="nearest",
        hold_open_stair_doors=True,
        stair_split_level=8,
        upper_stair="B",
        lower_stair="A",
        band_edges=[6, 11],
        band_delays=[0.0, 120.0, 240.0],
        warden_levels=[5],
    )
    out = plan.apply(spec, tower)
    assert out.evacuation_lifts and out.lift_priority == "nearest"
    assert out.hazard is not None and out.hazard.hold_open_stair_doors
    assert out.stair_assignment[8] == "B" and out.stair_assignment[7] == "A"
    assert out.phased_release.get(3, 0.0) == 0.0
    assert out.phased_release[7] == 120.0 and out.phased_release[15] == 240.0
    assert out.warden_levels == [5]
    assert spec.stair_assignment == {}  # the original is untouched
    assert set(plan.levers()) == {"lifts", "hold_open", "stair_assignment", "phasing", "wardens"}
    lines = plan.describe()
    assert any("lifts" in s for s in lines) and any("wardens" in s for s in lines)
    assert InterventionPlan().describe() == ["No change (baseline)."]
    assert InterventionPlan().is_empty() and not plan.is_empty()
    assert out.lift_eligibility == "mobility_impaired"
    wheel = InterventionPlan(evacuation_lifts=True, lift_eligibility="wheelchair_users")
    assert wheel.apply(spec, tower).lift_eligibility == "wheelchair_users"
    assert "wheelchair users only" in wheel.describe()[0]
    assert merge(InterventionPlan(), wheel).lift_eligibility == "wheelchair_users"


def test_merge() -> None:
    a = InterventionPlan(evacuation_lifts=True, warden_levels=[3])
    b = InterventionPlan(hold_open_stair_doors=True, warden_levels=[7])
    m = merge(a, b)
    assert m.evacuation_lifts and m.hold_open_stair_doors and m.warden_levels == [3, 7]


def test_bands() -> None:
    assert _bands(list(range(1, 21)), 4) == [6, 11, 16]
    assert _bands([1, 2], 4) == []


def test_wardens_keep_common_random_numbers(tower: Building) -> None:
    spec = ScenarioSpec(time_slot="weekend_night", share_65_plus=0.4)
    u = scenario_uniforms(0, 0, 1)[0]
    base = ScenarioSampler(tower, spec).sample(0, 0, u).population
    warded = ScenarioSampler(tower, spec.model_copy(update={"warden_levels": [8]})).sample(0, 0, u)
    pop = warded.population
    np.testing.assert_array_equal(base.agent_profile, pop.agent_profile)
    sweep = get_params().scalar("behaviour.wardens.sweep_time")
    covered = np.abs(pop.group_level - 8) <= 1
    assert (pop.group_premovement[covered] <= sweep + 1e-9).all()
    np.testing.assert_array_equal(pop.group_premovement[~covered], base.group_premovement[~covered])


def test_warden_escorts_waiting_household(tower: Building) -> None:
    spec = ScenarioSpec(share_65_plus=0.5)
    u = scenario_uniforms(0, 0, 40)
    sampler = ScenarioSampler(tower, spec)
    for i in range(40):
        pop = sampler.sample(0, i, u[i]).population
        waiting = np.flatnonzero(pop.group_mode == Mode.WAIT_RESCUE)
        if waiting.size:
            g = int(waiting[0])
            lv = int(pop.group_level[g])
            made = apply_wardens(pop, [lv], get_params())
            assert made >= 1
            assert pop.group_mode[g] == Mode.ASSISTED_STAIR
            assert pop.group_down_speed[g] == pop.group_assisted_down_speed[g]
            return
    pytest.skip("no household waiting for rescue in the sample")


def test_cmaes_minimises_quadratic() -> None:
    target = np.array([200.0, 600.0, 50.0])

    def f(x: np.ndarray) -> float:
        return float(np.sum(((x - target) / 900.0) ** 2))

    res = cmaes(f, np.zeros(3), np.zeros(3), np.full(3, 900.0), iterations=25, seed=1)
    assert res.f < 1e-3
    again = cmaes(f, np.zeros(3), np.zeros(3), np.full(3, 900.0), iterations=25, seed=1)
    assert again.f == res.f  # deterministic
    assert (res.x >= 0).all() and (res.x <= 900).all()


def test_objectives_and_confirm(tower: Building) -> None:
    res = run_monte_carlo(
        tower, ScenarioSpec(hazard=HazardSpec()), MCConfig(n_runs=12, batch_size=12, workers=1)
    )
    assert Objective().value(res) >= res.loss("total_time").mean()
    p = Objective(kind="p_rset").value(res)
    assert 0.0 <= p <= 1.0 + 1e-3
    w = Objective(kind="weighted", mean_weight=1.0).value(res)
    assert w == pytest.approx(res.loss("total_time").mean())
    same = confirm(res, res, Objective())
    assert same["losses"]["total_time"]["delta_cvar"]["value"] == 0.0
    assert not same["significant"]
    assert "p_rset_exceeds_aset" in same


def test_screening_plans(tower: Building) -> None:
    plans = screening_plans(tower, reference_spec(), OptimizeConfig())
    assert {"lifts", "hold_open", "stair_assignment", "phasing"} <= set(plans)
    assert all(pl.evacuation_lifts for pl in plans["lifts"])
    who = {pl.lift_eligibility for pl in plans["lifts"]}
    assert who == {"mobility_impaired", "wheelchair_users"}
    only = screening_plans(tower, reference_spec(), OptimizeConfig(levers=("lifts",)))
    assert set(only) == {"lifts"}


def test_optimize_small(tower: Building) -> None:
    res = optimize(
        tower,
        reference_spec().model_copy(update={"fire_level": 6}),
        Objective(),
        OptimizeConfig(
            n_scenarios=16,
            confirm_scenarios=24,
            cmaes_iterations=1,
            max_wardens=1,
            workers=2,
            levers=("lifts", "phasing", "wardens"),
        ),
    )
    assert res.best.value <= res.baseline.value
    assert len(res.history) >= 5
    s = res.summary(tower)
    assert s["plan_description"]
    assert set(s["confirmation"]["losses"]) == {
        "total_time",
        "self_evacuation_time",
        "p95_occupant_time",
    }
    assert res.confirm_baseline.n == res.confirm_best.n == 24
