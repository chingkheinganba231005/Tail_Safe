"""Operational intervention plans: cheap changes to *how* a building evacuates.

A plan is applied to a :class:`~tailsafe.scenarios.spec.ScenarioSpec`; the
result is evaluated with the same scenario draws as the baseline (common random
numbers). Levers:

* **Evacuation lifts** for mobility-impaired residents (or wheelchair users
  only), with a dispatch rule;
* **Door hold-open policy** for stair doors (more flow, more smoke in stairs);
* **Stair assignment** by floor band (upper floors to one stair, lower floors
  to the other);
* **Phased release**: per floor band, a delay before occupants may leave;
* **Floor wardens** on chosen floors (door-knocking and escorting).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tailsafe.building.builder import hk_level_label
from tailsafe.building.model import Building, LevelKind
from tailsafe.scenarios.spec import HazardSpec, ScenarioSpec


class InterventionPlan(BaseModel):
    """A set of operational measures."""

    model_config = ConfigDict(extra="forbid")

    evacuation_lifts: bool = False
    lift_priority: Literal["top_down", "nearest", "bottom_up"] = "top_down"
    lift_eligibility: Literal["mobility_impaired", "wheelchair_users"] = "mobility_impaired"
    hold_open_stair_doors: bool = False
    stair_split_level: int | None = Field(
        default=None, description="Floors at or above use upper_stair; below use lower_stair."
    )
    upper_stair: str | None = None
    lower_stair: str | None = None
    band_edges: list[int] = Field(
        default_factory=list,
        description="Lowest level of each phasing band above the first (ascending).",
    )
    band_delays: list[float] = Field(
        default_factory=list, description="Release delay (s) per band, bottom band first."
    )
    warden_levels: list[int] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """No measure is active."""
        return self == InterventionPlan()

    def levers(self) -> list[str]:
        """Names of the active levers."""
        out = []
        if self.evacuation_lifts:
            out.append("lifts")
        if self.hold_open_stair_doors:
            out.append("hold_open")
        if self.stair_split_level is not None:
            out.append("stair_assignment")
        if any(d > 0 for d in self.band_delays):
            out.append("phasing")
        if self.warden_levels:
            out.append("wardens")
        return out

    def band_of(self, level: int) -> int:
        """Phasing band index of a level."""
        return sum(1 for e in self.band_edges if level >= e)

    def apply(self, spec: ScenarioSpec, building: Building) -> ScenarioSpec:
        """The scenario specification with this plan in force."""
        update: dict[str, object] = {}
        if self.evacuation_lifts:
            update["evacuation_lifts"] = True
            update["lift_priority"] = self.lift_priority
            update["lift_eligibility"] = self.lift_eligibility
        if self.hold_open_stair_doors:
            hz = spec.hazard or HazardSpec(enabled=False)
            update["hazard"] = hz.model_copy(update={"hold_open_stair_doors": True})
        levels = [
            lv.index for lv in building.levels if lv.index > 0 and lv.kind != LevelKind.REFUGE
        ]
        if self.stair_split_level is not None and self.upper_stair and self.lower_stair:
            assign = dict(spec.stair_assignment)
            for lv in levels:
                assign[lv] = self.upper_stair if lv >= self.stair_split_level else self.lower_stair
            update["stair_assignment"] = assign
        if self.band_delays and any(d > 0 for d in self.band_delays):
            release = dict(spec.phased_release)
            for lv in levels:
                b = self.band_of(lv)
                if b < len(self.band_delays) and self.band_delays[b] > 0:
                    release[lv] = max(release.get(lv, 0.0), float(self.band_delays[b]))
            update["phased_release"] = release
        if self.warden_levels:
            update["warden_levels"] = sorted(set(spec.warden_levels) | set(self.warden_levels))
        return spec.model_copy(update=update) if update else spec

    def describe(self, building: Building | None = None) -> list[str]:
        """Plain-English bullet points."""
        lab = hk_level_label
        out: list[str] = []
        if self.evacuation_lifts:
            rule = {
                "top_down": "highest floors first",
                "nearest": "nearest waiting floor first",
                "bottom_up": "lowest floors first",
            }[self.lift_priority]
            who = (
                "mobility-impaired residents"
                if self.lift_eligibility == "mobility_impaired"
                else "wheelchair users only (frail residents walk)"
            )
            out.append(f"Use the non-firefighting lifts to evacuate {who} ({rule}).")
        if self.hold_open_stair_doors:
            out.append("Hold stair doors open during evacuation.")
        if self.stair_split_level is not None and self.upper_stair and self.lower_stair:
            out.append(
                f"Direct {lab(self.stair_split_level)} and above to Stair {self.upper_stair}; "
                f"floors below to Stair {self.lower_stair}."
            )
        if self.band_delays and any(d > 0 for d in self.band_delays):
            edges = [1, *self.band_edges]
            for b, d in enumerate(self.band_delays):
                if d > 0:
                    lo = edges[b]
                    hi = edges[b + 1] - 1 if b + 1 < len(edges) else None
                    rng = f"{lab(lo)}–{lab(hi)}" if hi is not None else f"{lab(lo)} and above"
                    out.append(f"Hold {rng} for {d / 60:.1f} min before releasing (phased).")
        if self.warden_levels:
            floors = ", ".join(lab(lv) for lv in sorted(self.warden_levels))
            out.append(f"Station floor wardens on {floors}.")
        del building
        return out or ["No change (baseline)."]
