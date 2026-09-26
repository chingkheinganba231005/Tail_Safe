from __future__ import annotations

import numpy as np
import pytest

from tailsafe.building.model import Building
from tailsafe.building.templates import generate
from tailsafe.config import get_params
from tailsafe.population.profiles import AgeClass, Profile, load_profiles
from tailsafe.population.synth import (
    Mode,
    PopulationConfig,
    TimeSlot,
    sample_population,
)
from tailsafe.rng import Streams, stream


@pytest.fixture(scope="module")
def tower() -> Building:
    return generate("cruciform", storeys=30)


def test_streams_are_deterministic_and_independent() -> None:
    a = stream(1, "occupancy", 0).random(5)
    np.testing.assert_array_equal(a, stream(1, "occupancy", 0).random(5))
    assert not np.array_equal(a, stream(1, "speeds", 0).random(5))
    assert not np.array_equal(a, stream(1, "occupancy", 1).random(5))
    assert not np.array_equal(a, stream(2, "occupancy", 0).random(5))
    s = Streams(1, 0)
    assert s["occupancy"] is s["occupancy"]
    with pytest.raises(ValueError):
        stream(-1, "x")


def test_same_seed_same_population(tower: Building) -> None:
    a = sample_population(tower, seed=3, index=7)
    b = sample_population(tower, seed=3, index=7)
    np.testing.assert_array_equal(a.group_premovement, b.group_premovement)
    np.testing.assert_array_equal(a.agent_profile, b.agent_profile)
    c = sample_population(tower, seed=3, index=8)
    assert a.n_agents != c.n_agents or not np.array_equal(a.agent_profile, c.agent_profile)


def test_group_aggregates(tower: Building) -> None:
    pop = sample_population(tower, PopulationConfig(time_slot=TimeSlot.WEEKEND_NIGHT), seed=1)
    specs = load_profiles()
    counts = np.bincount(pop.agent_group, minlength=pop.n_groups)
    np.testing.assert_array_equal(counts, pop.group_size)
    space = np.zeros(pop.n_groups)
    np.add.at(space, pop.agent_group, [specs[Profile(p)].space_factor for p in pop.agent_profile])
    np.testing.assert_allclose(space, pop.group_space)
    assert (pop.group_h_speed > 0).all() and (pop.group_down_speed > 0).all()
    assert (pop.group_premovement > 0).all()


def test_share_65_plus_knob_is_coupled_and_monotone(tower: Building) -> None:
    shares = []
    for s in (0.1, 0.22, 0.4):
        pop = sample_population(tower, PopulationConfig(share_65_plus=s), seed=5)
        census = ~(pop.agent_age == AgeClass.HELPER) & ~pop.agent_is_staff
        older = np.isin(pop.agent_age, [AgeClass.OLDER, AgeClass.FRAIL]) & census
        shares.append(older.sum() / census.sum())
    assert shares[0] < shares[1] < shares[2]
    # Present-share differs from the resident share only through presence rates.
    assert shares[1] == pytest.approx(0.22, abs=0.04)


def test_weekday_day_has_fewer_occupants_than_night(tower: Building) -> None:
    day = np.mean(
        [
            sample_population(
                tower, PopulationConfig(time_slot=TimeSlot.WEEKDAY_DAY), seed=s
            ).n_agents
            for s in range(4)
        ]
    )
    night = np.mean(
        [
            sample_population(
                tower, PopulationConfig(time_slot=TimeSlot.WEEKDAY_NIGHT), seed=s
            ).n_agents
            for s in range(4)
        ]
    )
    assert day < 0.8 * night


def test_night_premovement_is_longer(tower: Building) -> None:
    day = sample_population(tower, PopulationConfig(time_slot=TimeSlot.WEEKDAY_DAY), seed=2)
    night = sample_population(tower, PopulationConfig(time_slot=TimeSlot.WEEKDAY_NIGHT), seed=2)
    assert np.median(night.group_premovement) > 2 * np.median(day.group_premovement)


def test_mobility_modes(tower: Building) -> None:
    for lifts in (False, True):
        pop = sample_population(
            tower, PopulationConfig(evacuation_lifts=lifts, share_65_plus=0.35), seed=9
        )
        wheel_groups = np.unique(pop.agent_group[pop.agent_profile == Profile.WHEELCHAIR_USER])
        upstairs = wheel_groups[pop.group_level[wheel_groups] > 0]
        assert upstairs.size > 0
        # Wheelchair users above ground never simply walk down.
        assert not np.isin(pop.group_mode[upstairs], [Mode.WALK]).any()
        has_lift_mode = (pop.group_mode == Mode.WAIT_LIFT).any()
        assert has_lift_mode == lifts


def test_lift_eligibility_wheelchair_only(tower: Building) -> None:
    both = sample_population(
        tower, PopulationConfig(evacuation_lifts=True, share_65_plus=0.35), seed=9
    )
    wheel = sample_population(
        tower,
        PopulationConfig(evacuation_lifts=True, share_65_plus=0.35, lift_for_frail=False),
        seed=9,
    )
    has_wheel = np.zeros(wheel.n_groups, dtype=bool)
    has_wheel[wheel.agent_group[wheel.agent_profile == Profile.WHEELCHAIR_USER]] = True
    waiting = wheel.group_mode == Mode.WAIT_LIFT
    assert waiting.any() and not (waiting & ~has_wheel).any()
    # Frail households that waited for a lift now walk; the draws are otherwise identical.
    frail_only = (both.group_mode == Mode.WAIT_LIFT) & ~has_wheel
    assert frail_only.any()
    assert (wheel.group_mode[frail_only] == Mode.WALK).all()
    np.testing.assert_array_equal(wheel.group_premovement, both.group_premovement)


def test_counter_flow_probability(tower: Building) -> None:
    none = sample_population(tower, PopulationConfig(counter_flow_probability=0.0), seed=1)
    assert all(w is None for w in none.group_waypoint)
    every = sample_population(tower, PopulationConfig(counter_flow_probability=1.0), seed=1)
    walkers = every.group_mode == Mode.WALK
    with_wp = np.array([w is not None for w in every.group_waypoint])
    assert with_wp[walkers].mean() > 0.95
    assert not with_wp[~walkers].any()
    ids = {n.id for n in tower.units}
    assert {w for w in every.group_waypoint if w} <= ids


def test_refuge_rest_only_below_refuge_floor(tower: Building) -> None:
    pop = sample_population(tower, PopulationConfig(share_65_plus=0.4), seed=4)
    resting = pop.group_refuge_rest > 0
    assert resting.any()
    assert (pop.group_level[resting] > 20).all()


def test_care_home_staffing() -> None:
    home = generate("care_home")
    day = sample_population(home, PopulationConfig(time_slot=TimeSlot.WEEKDAY_DAY), seed=1)
    night = sample_population(home, PopulationConfig(time_slot=TimeSlot.WEEKDAY_NIGHT), seed=1)
    assert day.agent_is_staff.sum() > night.agent_is_staff.sum() >= 1
    assert not (day.agent_age == AgeClass.HELPER).any()
    staffed = np.unique(night.agent_group[night.agent_is_staff])
    assert (night.group_size[staffed] >= 2).all()
    # Night: many unescorted rooms with wheelchair users must wait for rescue.
    assert (night.group_mode == Mode.WAIT_RESCUE).sum() > (day.group_mode == Mode.WAIT_RESCUE).sum()


def test_vacancy_override(tower: Building) -> None:
    empty = sample_population(tower, PopulationConfig(vacancy_rate=1.0), seed=1)
    assert empty.n_agents == 0 and empty.n_groups == 0


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        PopulationConfig(share_65_plus=1.5)


def test_fatigue_curve() -> None:
    spec = load_profiles()[Profile.FRAIL_OLDER_ADULT]
    m = spec.fatigue(np.array([0.0, spec.fatigue_efold_floors, 1e6]))
    assert m[0] == pytest.approx(1.0)
    assert m[-1] == pytest.approx(spec.fatigue_min_multiplier)
    assert m[0] > m[1] > m[2]
    assert get_params()["profiles.frail_older_adult.fatigue_min_multiplier"].is_assumption
