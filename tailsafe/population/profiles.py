"""Occupant profiles and their movement characteristics (from ``params.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from tailsafe.config import Param, Params, get_params


class Profile(IntEnum):
    """Occupant profile. Integer values index arrays in the simulator."""

    ABLE_ADULT = 0
    CHILD = 1
    OLDER_ADULT = 2
    FRAIL_OLDER_ADULT = 3
    WHEELCHAIR_USER = 4
    DOMESTIC_HELPER = 5

    @property
    def key(self) -> str:
        """Key under ``profiles`` in the parameter registry."""
        return self.name.lower()

    @property
    def label(self) -> str:
        """Human-readable label."""
        return {
            Profile.ABLE_ADULT: "able adult",
            Profile.CHILD: "child",
            Profile.OLDER_ADULT: "older adult (65–79)",
            Profile.FRAIL_OLDER_ADULT: "frail older adult (80+)",
            Profile.WHEELCHAIR_USER: "wheelchair user",
            Profile.DOMESTIC_HELPER: "domestic helper",
        }[self]


class AgeClass(IntEnum):
    """Demographic class used by the occupancy priors."""

    CHILD = 0
    ADULT = 1
    OLDER = 2
    FRAIL = 3
    HELPER = 4

    @property
    def key(self) -> str:
        """Key used in ``population.*`` tables of the registry."""
        return self.name.lower()


AGE_TO_PROFILE = {
    AgeClass.CHILD: Profile.CHILD,
    AgeClass.ADULT: Profile.ABLE_ADULT,
    AgeClass.OLDER: Profile.OLDER_ADULT,
    AgeClass.FRAIL: Profile.FRAIL_OLDER_ADULT,
    AgeClass.HELPER: Profile.DOMESTIC_HELPER,
}

# Order used to pick the "most dependent" member of a household.
DEPENDENCY_ORDER = (
    Profile.WHEELCHAIR_USER,
    Profile.FRAIL_OLDER_ADULT,
    Profile.OLDER_ADULT,
    Profile.CHILD,
    Profile.DOMESTIC_HELPER,
    Profile.ABLE_ADULT,
)

ABLE_ESCORTS = frozenset({Profile.ABLE_ADULT, Profile.DOMESTIC_HELPER})
MOBILITY_IMPAIRED = frozenset({Profile.WHEELCHAIR_USER, Profile.FRAIL_OLDER_ADULT})


@dataclass(frozen=True)
class ProfileSpec:
    """Movement characteristics of one profile."""

    profile: Profile
    horizontal_speed: Param
    stair_down_speed: Param
    stair_up_speed: Param
    fatigue_min_multiplier: float
    fatigue_efold_floors: float
    premovement_multiplier: float
    space_factor: float
    needs_stair_assistance: bool

    def fatigue(self, floors_descended: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Stair-descent speed multiplier after descending ``floors_descended`` storeys.

        ``m(n) = m_min + (1 - m_min) * exp(-n / n_e)``: 1 at the start, decaying
        towards ``m_min`` with an e-folding length of ``n_e`` storeys.
        """
        n = np.asarray(floors_descended, dtype=np.float64)
        m = self.fatigue_min_multiplier
        return m + (1.0 - m) * np.exp(-n / self.fatigue_efold_floors)


def load_profiles(params: Params | None = None) -> dict[Profile, ProfileSpec]:
    """Read every profile's characteristics from the registry."""
    p = params or get_params()
    out: dict[Profile, ProfileSpec] = {}
    for prof in Profile:
        g = f"profiles.{prof.key}."
        out[prof] = ProfileSpec(
            profile=prof,
            horizontal_speed=p[g + "horizontal_speed"],
            stair_down_speed=p[g + "stair_down_speed"],
            stair_up_speed=p[g + "stair_up_speed"],
            fatigue_min_multiplier=p.scalar(g + "fatigue_min_multiplier"),
            fatigue_efold_floors=p.scalar(g + "fatigue_efold_floors"),
            premovement_multiplier=p.scalar(g + "premovement_multiplier"),
            space_factor=p.scalar(g + "space_factor"),
            needs_stair_assistance=bool(p.value(g + "needs_stair_assistance")),
        )
    return out
