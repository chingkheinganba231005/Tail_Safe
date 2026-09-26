"""Monotonicity checks with common random numbers (spec §7).

Each comparison evaluates two variants on the *same* sampled scenarios, so the
difference per scenario isolates the change being tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from tailsafe.building.model import Building
from tailsafe.building.templates import generate
from tailsafe.config import get_params
from tailsafe.scenarios.montecarlo import MCConfig, MCResult, run_monte_carlo
from tailsafe.scenarios.spec import HazardSpec, ScenarioSpec, StairBlockage

CFG = MCConfig(n_runs=40, batch_size=40, workers=2, keep_groups=False)
BASE = ScenarioSpec(name="base", share_65_plus=0.25)


@pytest.fixture(scope="module")
def tower() -> Building:
    return generate("cruciform", storeys=16, flats_per_wing=2)


@pytest.fixture(scope="module")
def base(tower: Building) -> MCResult:
    return run_monte_carlo(tower, BASE, CFG)


def paired(a: MCResult, b: MCResult, loss: str) -> np.ndarray:
    return b.loss(loss) - a.loss(loss)


def test_blocking_a_stair_never_helps(tower: Building, base: MCResult) -> None:
    spec = BASE.model_copy(update={"stair_blockages": [StairBlockage(stair="A")]})
    blocked = run_monte_carlo(tower, spec, CFG)
    for loss in ("p95_occupant_time", "self_evacuation_time"):
        d = paired(base, blocked, loss)
        assert (d >= -1.0).mean() >= 0.95, loss
        assert d.mean() > 0, loss


def test_wider_doors_and_exits_never_hurt(base: MCResult) -> None:
    """More exit width never slows evacuation.

    Stair doors and final exits are widened; path lengths are unchanged. (A
    wider *stair* also lengthens the walking line around each half-landing
    turn, so it is not a pure capacity change.)
    """
    params = get_params().with_overrides(
        {"building_defaults.stair_door_width": 1.2, "building_defaults.final_exit_width": 2.4}
    )
    wide_building = generate("cruciform", storeys=16, flats_per_wing=2, params=params)
    wide = run_monte_carlo(wide_building, BASE, CFG, params=params)
    for loss in ("p95_occupant_time", "self_evacuation_time", "total_time"):
        d = paired(base, wide, loss)
        assert (d <= 1.0).mean() >= 0.95, loss
        assert d.mean() <= 0.5, loss


def test_smoke_never_speeds_evacuation(tower: Building, base: MCResult) -> None:
    smoky = run_monte_carlo(
        tower, BASE.model_copy(update={"hazard": HazardSpec(door_open_probability=1.0)}), CFG
    )
    d = paired(base, smoky, "p95_occupant_time")
    assert (d >= -1.0).mean() >= 0.95
    assert smoky.p_rset_exceeds_aset().value >= base.p_rset_exceeds_aset().value


def test_open_fire_doors_raise_p_rset_gt_aset(tower: Building) -> None:
    shut = run_monte_carlo(
        tower, BASE.model_copy(update={"hazard": HazardSpec(door_open_probability=0.0)}), CFG
    )
    open_ = run_monte_carlo(
        tower, BASE.model_copy(update={"hazard": HazardSpec(door_open_probability=1.0)}), CFG
    )
    assert open_.p_rset_exceeds_aset().value > shut.p_rset_exceeds_aset().value
