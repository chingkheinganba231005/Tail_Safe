from __future__ import annotations

import numpy as np
import pytest

from tailsafe.building.model import Building
from tailsafe.building.templates import generate
from tailsafe.population.synth import Mode
from tailsafe.scenarios.sampler import N_DIMS, ScenarioSampler, scenario_uniforms
from tailsafe.scenarios.spec import (
    Dist,
    RandomStairBlockage,
    ScenarioSpec,
    StairBlockage,
    demo_spec,
)


@pytest.fixture(scope="module")
def tower() -> Building:
    return generate("cruciform", storeys=12, flats_per_wing=3)


def test_uniforms_deterministic_and_lhs_stratified() -> None:
    a = scenario_uniforms(7, 0, 50, lhs=True, batch_size=50)
    b = scenario_uniforms(7, 0, 50, lhs=True, batch_size=50)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (50, N_DIMS)
    for d in range(N_DIMS):
        strata = np.floor(a[:, d] * 50).astype(int)
        assert sorted(strata) == list(range(50))
    # Any sub-range of indices gives the same rows as the full batch.
    np.testing.assert_array_equal(scenario_uniforms(7, 10, 5, lhs=True, batch_size=50), a[10:15])
    iid = scenario_uniforms(7, 0, 5, lhs=False)
    np.testing.assert_array_equal(iid[2], scenario_uniforms(7, 2, 1, lhs=False)[0])


def test_dist_ppf() -> None:
    assert Dist.fixed(5.0).ppf([0.1, 0.9]).tolist() == [5.0, 5.0]
    u = Dist(dist="uniform", min=100, max=200).ppf([0.0, 0.5, 1.0])
    assert u[1] == pytest.approx(150.0)
    with pytest.raises(ValueError):
        Dist(dist="lognormal", median=10.0)


def test_demo_spec_sampling(tower: Building) -> None:
    sampler = ScenarioSampler(tower, demo_spec())
    u = scenario_uniforms(0, 0, 1)[0]
    sc = sampler.sample(0, 0, u)
    assert sc.info["fire_level"] == 14
    assert sc.info["blocked_stairs"] == {"A": 240.0}
    assert all(b.time == 240.0 for b in sc.sim.blockages)
    assert len(sc.info["lifts_out"]) == 1
    assert sc.sim.evacuation_lifts == ()  # baseline: lifts are not used for evacuation
    assert sc.population.config.share_65_plus == 0.22
    assert sc.sim.rescue_start == pytest.approx(sc.info["rescue_start"])


def test_common_random_numbers_across_specs(tower: Building) -> None:
    """An intervention changes behaviour, not who is home or when they react."""
    base = ScenarioSampler(tower, demo_spec())
    lifts = ScenarioSampler(tower, demo_spec().model_copy(update={"evacuation_lifts": True}))
    u = scenario_uniforms(3, 0, 4)
    for i in range(4):
        a, b = base.sample(3, i, u[i]), lifts.sample(3, i, u[i])
        np.testing.assert_array_equal(a.population.agent_profile, b.population.agent_profile)
        np.testing.assert_array_equal(
            a.population.group_premovement, b.population.group_premovement
        )
        assert a.info["lifts_out"] == b.info["lifts_out"]
        assert a.info["rescue_start"] == b.info["rescue_start"]
    assert (b.population.group_mode == Mode.WAIT_LIFT).any()
    assert set(b.info["lifts_out"]).isdisjoint(b.sim.evacuation_lifts)


def test_random_blockage_probability(tower: Building) -> None:
    spec = ScenarioSpec(
        random_stair_blockage=RandomStairBlockage(
            probability=0.3, time=Dist(dist="uniform", min=60, max=600)
        )
    )
    sampler = ScenarioSampler(tower, spec)
    u = scenario_uniforms(1, 0, 400, lhs=True, batch_size=400)
    hits = [bool(sampler.sample(1, i, u[i]).info["blocked_stairs"]) for i in range(0, 400, 4)]
    assert np.mean(hits) == pytest.approx(0.3, abs=0.1)


def test_invalid_specs(tower: Building) -> None:
    with pytest.raises(ValueError):
        ScenarioSampler(tower, ScenarioSpec(stair_blockages=[StairBlockage(stair="Z")]))
    with pytest.raises(ValueError):
        ScenarioSampler(tower, ScenarioSpec(stair_assignment={3: "Q"}))
    with pytest.raises(ValueError):
        ScenarioSampler(tower, ScenarioSpec(lifts_out_of_service=9))


def test_spec_roundtrip() -> None:
    spec = demo_spec()
    again = ScenarioSpec.model_validate_json(spec.model_dump_json())
    assert again == spec
    assert "name" not in again.digest_payload()


def test_hazard_sampling(tower: Building) -> None:
    from tailsafe.config import get_params
    from tailsafe.hazard.model import HazardResult
    from tailsafe.scenarios.spec import HazardSpec

    spec = ScenarioSpec(fire_level=6, hazard=HazardSpec(door_open_probability=0.5))
    sampler = ScenarioSampler(tower, spec)
    u = scenario_uniforms(2, 0, 30, lhs=True, batch_size=30)
    opens = []
    for i in range(30):
        sc = sampler.sample(2, i, u[i])
        assert isinstance(sc.sim.hazard, HazardResult)
        fire = sc.info["fire"]
        assert fire["unit"].startswith("L06.unit.")
        opens.append(fire["door_open"])
        g = [k for k, uid in enumerate(sc.population.group_unit) if uid == fire["unit"]]
        if g:
            pre = get_params()["hazard.fire_unit_premovement"]
            assert sc.population.group_premovement[g[0]] <= pre.spec["max"]
    assert 0.3 < np.mean(opens) < 0.7
    with pytest.raises(ValueError):
        ScenarioSampler(tower, ScenarioSpec(hazard=HazardSpec(fire_unit="L99.unit.X")))


def test_hold_open_policy(tower: Building) -> None:
    from tailsafe.scenarios.spec import HazardSpec

    spec = ScenarioSpec(hazard=HazardSpec(hold_open_stair_doors=True))
    sampler = ScenarioSampler(tower, spec)
    assert sampler.held_open
    assert all(".stair." in e or ".pl." in e for e in sampler.held_open)
    sc = sampler.sample(0, 0, scenario_uniforms(0, 0, 1)[0])
    assert sc.sim.held_open_doors == sampler.held_open
