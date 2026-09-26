"""Scenario specifications: the *distribution* of evacuations to stress-test.

A :class:`ScenarioSpec` fixes what is known (time of day, population mix,
operational measures) and describes what is uncertain (when a stair becomes
smoke-logged, whether a random stair is lost, which lifts are out of service,
when fire-service rescue starts). The sampler draws concrete scenarios from it.
Specs are plain JSON/YAML so the API and the web UI can exchange them.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tailsafe.config import Param
from tailsafe.population.synth import TimeSlot


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Dist(_Model):
    """A one-dimensional distribution, sampled by inverse CDF like registry values."""

    dist: Literal["constant", "uniform", "truncnorm", "lognormal"] = "constant"
    value: float | None = None
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    sd: float | None = None
    median: float | None = None
    sigma: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Dist:
        needed = {
            "constant": ("value",),
            "uniform": ("min", "max"),
            "truncnorm": ("mean", "sd", "min", "max"),
            "lognormal": ("median", "sigma"),
        }[self.dist]
        missing = [f for f in needed if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{self.dist} distribution needs {missing}")
        return self

    @classmethod
    def fixed(cls, value: float) -> Dist:
        """A constant."""
        return cls(dist="constant", value=value)

    def as_param(self, name: str = "scenario") -> Param:
        """View as a registry :class:`~tailsafe.config.Param` (for ``ppf``)."""
        spec = {k: v for k, v in self.model_dump().items() if v is not None}
        spec["source"] = "scenario specification"
        return Param(path=name, spec=spec)

    def ppf(self, u: ArrayLike) -> NDArray[np.float64]:
        """Inverse CDF."""
        return np.asarray(self.as_param().ppf(u), dtype=np.float64)


class StairBlockage(_Model):
    """A named staircase becomes impassable (e.g. smoke-logged) at a (random) time."""

    stair: str
    time: Dist = Field(default_factory=lambda: Dist.fixed(0.0))


class RandomStairBlockage(_Model):
    """With ``probability``, one staircase chosen uniformly at random is lost."""

    probability: float = Field(ge=0.0, le=1.0)
    time: Dist = Field(default_factory=lambda: Dist.fixed(0.0))


class HazardSpec(_Model):
    """Fire and smoke. Unset fields fall back to the ``hazard`` registry section."""

    enabled: bool = True
    fire_unit: str | None = Field(
        default=None,
        description="Flat where the fire starts; None = random flat on the fire level.",
    )
    growth: Dist | None = Field(default=None, description="t-squared growth coefficient (kW/s²).")
    peak_hrr: Dist | None = Field(default=None, description="Peak heat release rate (kW).")
    door_open_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    hold_open_stair_doors: bool = Field(
        default=False,
        description="Policy: stair (and protected-lobby) doors are held open — full "
        "flow capacity, but smoke passes freely.",
    )
    held_open_doors: list[str] = Field(default_factory=list)
    horizon: float = Field(default=7200.0, gt=0)
    record_dt: float = Field(default=10.0, gt=0)


class ScenarioSpec(_Model):
    """What to stress-test. Unset fields fall back to the parameter registry."""

    name: str = "custom"
    description: str | None = None
    time_slot: TimeSlot = TimeSlot.WEEKDAY_NIGHT
    share_65_plus: float | None = Field(default=None, ge=0.0, le=1.0)
    share_80_plus_of_65_plus: float | None = Field(default=None, ge=0.0, le=1.0)
    counter_flow_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    vacancy_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    fire_level: int | None = Field(
        default=None, description="Fire floor (used by the hazard model); None = random floor."
    )
    stair_blockages: list[StairBlockage] = Field(default_factory=list)
    random_stair_blockage: RandomStairBlockage | None = None
    evacuation_lifts: bool = Field(
        default=False, description="Put every non-firefighting lift into evacuation service."
    )
    lifts_out_of_service: int = Field(default=0, ge=0, description="Random lifts unavailable.")
    lift_priority: Literal["top_down", "nearest", "bottom_up"] = "top_down"
    phased_release: dict[int, float] = Field(
        default_factory=dict, description="Level -> earliest time its occupants may leave (s)."
    )
    stair_assignment: dict[int, str] = Field(
        default_factory=dict, description="Level -> staircase its occupants must use."
    )
    rescue_start: Dist | None = Field(
        default=None, description="Start of fire-service rescue; default from the registry."
    )
    rescue_teams: int | None = Field(default=None, ge=0)
    hazard: HazardSpec | None = Field(default=None, description="Fire and smoke (M4).")

    def digest_payload(self) -> dict[str, Any]:
        """Canonical content for cache keys (name and description excluded)."""
        return self.model_dump(mode="json", exclude={"name", "description"})


def demo_spec() -> ScenarioSpec:
    """The pitch scenario (spec §10).

    Sunday 3 a.m., 22% of residents aged 65+, fire on 14/F, Stair A smoke-logged
    at t = 4 min, one lift out of service.
    """
    return ScenarioSpec(
        name="demo-sunday-3am",
        description=(
            "Sunday, 3 a.m., 40-storey public housing block, 22% of residents aged 65+, "
            "fire on 14/F, Stair A smoke-logged at t = 4 min, one lift out of service."
        ),
        time_slot=TimeSlot.WEEKEND_NIGHT,
        share_65_plus=0.22,
        fire_level=14,
        stair_blockages=[StairBlockage(stair="A", time=Dist.fixed(240.0))],
        lifts_out_of_service=1,
        hazard=HazardSpec(),
    )
