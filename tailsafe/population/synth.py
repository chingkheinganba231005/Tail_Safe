"""Synthetic population: households, occupants, behaviours.

All occupants are synthetic. For each unit we draw a fixed-shape block of
uniform random numbers per named stream (see :mod:`tailsafe.rng`) and transform
them through the priors in ``params.yaml``. Because the *draws* never depend on
the scenario knobs, two scenarios that differ only in a knob (say the share of
residents aged 65+) are coupled: the same units are vacant, the same members
are home, and the change shows up only where the knob matters. This keeps
comparisons between interventions low-variance (common random numbers).

Households are simulated as *groups* that move together at the pace of their
slowest member.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tailsafe.building.model import Building, LevelKind, Node
from tailsafe.config import Params, get_params
from tailsafe.population.profiles import (
    ABLE_ESCORTS,
    AGE_TO_PROFILE,
    DEPENDENCY_ORDER,
    AgeClass,
    Profile,
    ProfileSpec,
    load_profiles,
)
from tailsafe.rng import Streams

MAX_MEMBERS = 6  # household members per unit, excluding the helper / staff slot
_SLOTS = MAX_MEMBERS + 1  # + one live-in helper or care-staff escort


class TimeSlot(StrEnum):
    """Time-of-day × day-type occupancy regime."""

    WEEKDAY_DAY = "weekday_day"
    WEEKDAY_NIGHT = "weekday_night"
    WEEKEND_DAY = "weekend_day"
    WEEKEND_NIGHT = "weekend_night"

    @property
    def is_night(self) -> bool:
        """Night-time slot (most occupants asleep)."""
        return self.value.endswith("night")


class Mode(IntEnum):
    """How a household evacuates."""

    WALK = 0
    ASSISTED_STAIR = 1  # carrying a wheelchair user down the stairs
    WAIT_LIFT = 2  # wait at the floor's lift lobby for an evacuation lift
    WAIT_RESCUE = 3  # stay put and wait for the fire service


@dataclass(frozen=True)
class PopulationConfig:
    """Scenario knobs that shape the synthetic population."""

    time_slot: TimeSlot = TimeSlot.WEEKDAY_NIGHT
    share_65_plus: float | None = None
    share_80_plus_of_65_plus: float | None = None
    evacuation_lifts: bool = False
    lift_for_frail: bool = True  # frail older adults may wait for a lift too (else wheelchair only)
    counter_flow_probability: float | None = None
    vacancy_rate: float | None = None

    def __post_init__(self) -> None:
        for name in ("share_65_plus", "share_80_plus_of_65_plus", "counter_flow_probability"):
            v = getattr(self, name)
            if v is not None and not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass
class Population:
    """Occupants (agents) and the households (groups) they move in.

    Agent arrays are indexed by agent; group arrays by group. ``agent_group``
    maps agents to groups.
    """

    # --- agents
    agent_profile: NDArray[np.int8]
    agent_age: NDArray[np.int8]
    agent_group: NDArray[np.int32]
    agent_is_staff: NDArray[np.bool_]
    # --- groups
    group_unit: list[str]
    group_level: NDArray[np.int32]
    group_size: NDArray[np.int32]
    group_space: NDArray[np.float64]
    group_premovement: NDArray[np.float64]
    group_asleep: NDArray[np.bool_]
    group_mode: NDArray[np.int8]
    group_h_speed: NDArray[np.float64]
    group_down_speed: NDArray[np.float64]
    group_up_speed: NDArray[np.float64]
    group_assisted_down_speed: NDArray[np.float64]
    group_fatigue_min: NDArray[np.float64]
    group_fatigue_efold: NDArray[np.float64]
    group_key_profile: NDArray[np.int8]
    group_waypoint: list[str | None]
    group_waypoint_dwell: NDArray[np.float64]
    group_refuge_rest: NDArray[np.float64]
    group_route_u: NDArray[np.float64]
    config: PopulationConfig = field(default_factory=PopulationConfig)

    @property
    def n_agents(self) -> int:
        """Number of occupants present."""
        return int(self.agent_profile.size)

    @property
    def n_groups(self) -> int:
        """Number of households (groups) present."""
        return len(self.group_unit)

    def summary(self) -> dict[str, Any]:
        """Counts by profile and mode, and the realised 65+ share."""
        prof = Counter(Profile(int(p)).key for p in self.agent_profile)
        modes = Counter(Mode(int(m)).name.lower() for m in self.group_mode)
        census = ~np.isin(self.agent_age, [AgeClass.HELPER]) & ~self.agent_is_staff
        older = np.isin(self.agent_age, [AgeClass.OLDER, AgeClass.FRAIL]) & census
        return {
            "agents": self.n_agents,
            "groups": self.n_groups,
            "profiles": dict(prof),
            "modes": dict(modes),
            "share_65_plus": float(older.sum()) / max(int(census.sum()), 1),
            "counter_flow_groups": sum(w is not None for w in self.group_waypoint),
            "refuge_rest_groups": int((self.group_refuge_rest > 0).sum()),
        }


def _age_mix(params: Params, care: bool, cfg: PopulationConfig) -> dict[AgeClass, float]:
    """Age-class probabilities, adjusted to the scenario's 65+ share if given."""
    key = "population.age_mix.care_home" if care else "population.age_mix.residential"
    outcomes, probs = params[key].categories
    mix = {AgeClass[str(o).upper()]: float(p) for o, p in zip(outcomes, probs, strict=True)}
    if care or cfg.share_65_plus is None:
        return mix
    s = cfg.share_65_plus
    frail_share = (
        cfg.share_80_plus_of_65_plus
        if cfg.share_80_plus_of_65_plus is not None
        else params.scalar("population.frail_share_of_65_plus")
    )
    young = mix.get(AgeClass.CHILD, 0.0) + mix.get(AgeClass.ADULT, 0.0)
    return {
        AgeClass.CHILD: mix.get(AgeClass.CHILD, 0.0) * (1 - s) / young,
        AgeClass.ADULT: mix.get(AgeClass.ADULT, 0.0) * (1 - s) / young,
        AgeClass.OLDER: s * (1 - frail_share),
        AgeClass.FRAIL: s * frail_share,
    }


def _categorical_ppf(u: NDArray[np.float64], mix: dict[AgeClass, float]) -> NDArray[np.int8]:
    classes = np.array([int(k) for k in mix], dtype=np.int8)
    cum = np.cumsum(np.array(list(mix.values()), dtype=np.float64))
    cum /= cum[-1]
    idx = np.minimum(np.searchsorted(cum, u, side="right"), len(classes) - 1)
    out: NDArray[np.int8] = classes[idx]
    return out


def sample_population(
    building: Building,
    config: PopulationConfig | None = None,
    *,
    seed: int = 0,
    index: int = 0,
    params: Params | None = None,
) -> Population:
    """Sample the occupants present in ``building`` for one scenario.

    ``seed`` and ``index`` select the random streams (see :mod:`tailsafe.rng`);
    the same ``(seed, index)`` always yields the same population.
    """
    cfg = config or PopulationConfig()
    p = params or get_params()
    specs = load_profiles(p)
    streams = Streams(seed, index)
    units = building.units
    U = len(units)
    S = MAX_MEMBERS
    slot = cfg.time_slot.value

    # ---------------------------------------------------------------- uniforms
    occ = streams["occupancy"]
    u_vacant = occ.random(U)
    u_size = occ.random(U)
    u_age = occ.random((U, S))
    u_wheel = occ.random((U, S))
    u_helper = occ.random(U)
    u_present = occ.random((U, _SLOTS))
    u_asleep = occ.random(U)
    spd = streams["speeds"]
    u_h = spd.random((U, _SLOTS))
    u_down = spd.random((U, _SLOTS))
    u_up = spd.random((U, _SLOTS))
    u_pre = streams["premovement"].random(U)
    beh = streams["behaviour"]
    u_mode = beh.random((U, 2))
    u_cf = beh.random((U, 4))
    u_rest = beh.random((U, 2))
    u_route = beh.random(U)

    # ---------------------------------------------------------------- households
    care = np.array([n.unit_type == "care_room" for n in units], dtype=bool)
    vacancy = (
        cfg.vacancy_rate if cfg.vacancy_rate is not None else p.scalar("population.vacancy_rate")
    )
    vacant = (u_vacant < vacancy) & ~care
    size = np.zeros(U, dtype=np.int32)
    for utype in {n.unit_type or "flat_medium" for n in units}:
        idx = np.array([(n.unit_type or "flat_medium") == utype for n in units])
        size[idx] = p[f"population.household_size.{utype}"].ppf(u_size[idx]).astype(np.int32)
    size = np.minimum(size, S)
    size[vacant] = 0
    member = np.arange(S)[None, :] < size[:, None]

    age = np.empty((U, S), dtype=np.int8)
    age[~care] = _categorical_ppf(u_age[~care], _age_mix(p, False, cfg))
    if care.any():
        age[care] = _categorical_ppf(u_age[care], _age_mix(p, True, cfg))
    all_children = member.any(axis=1) & ~((age != AgeClass.CHILD) & member).any(axis=1)
    age[all_children, 0] = AgeClass.ADULT

    p_wheel_res = p.value("population.wheelchair_probability")
    p_wheel_care = p.value("population.wheelchair_probability_care_home")
    wheel_p = np.zeros((U, S))
    for a in (AgeClass.CHILD, AgeClass.ADULT, AgeClass.OLDER, AgeClass.FRAIL):
        mask = age == a
        wheel_p[mask & ~care[:, None]] = float(p_wheel_res.get(a.key, 0.0))
        wheel_p[mask & care[:, None]] = float(p_wheel_care.get(a.key, 0.0))
    wheel = (u_wheel < wheel_p) & member

    dependant = (((age == AgeClass.FRAIL) | wheel) & member).any(axis=1)
    children = ((age == AgeClass.CHILD) & member).any(axis=1)
    hp = p.value("population.helper_probability")
    p_helper = np.where(
        dependant, hp["with_dependant"], np.where(children, hp["with_children"], hp["other"])
    )
    helper = (u_helper < p_helper) & (size > 0) & ~care

    presence = p.value("population.presence")[slot]
    pres_p = np.zeros((U, _SLOTS))
    for a in (AgeClass.CHILD, AgeClass.ADULT, AgeClass.OLDER, AgeClass.FRAIL):
        pres_p[:, :S][age == a] = float(presence[a.key])
    pres_p[:, S] = float(presence["helper"])
    present = u_present < pres_p
    present[:, :S] &= member
    present[:, S] &= helper

    profile = np.full((U, _SLOTS), -1, dtype=np.int8)
    for a, prof in AGE_TO_PROFILE.items():
        if a != AgeClass.HELPER:
            profile[:, :S][age == a] = prof
    profile[:, :S][wheel] = Profile.WHEELCHAIR_USER
    profile[:, S] = Profile.DOMESTIC_HELPER
    age_full = np.concatenate([age, np.full((U, 1), AgeClass.HELPER, dtype=np.int8)], axis=1)
    staff = np.zeros((U, _SLOTS), dtype=bool)

    # ---------------------------------------------------------------- care staff
    if care.any():
        _assign_care_staff(building, units, care, present, profile, age_full, staff, cfg, p)

    # ---------------------------------------------------------------- speeds
    h = np.zeros((U, _SLOTS))
    down = np.zeros((U, _SLOTS))
    up = np.zeros((U, _SLOTS))
    for prof, spec in specs.items():
        m = profile == prof
        if m.any():
            h[m] = spec.horizontal_speed.ppf(u_h[m])
            down[m] = spec.stair_down_speed.ppf(u_down[m])
            up[m] = spec.stair_up_speed.ppf(u_up[m])

    # ---------------------------------------------------------------- groups
    occupied = present.any(axis=1)
    g_units = np.flatnonzero(occupied)
    return _build_groups(
        building,
        units,
        g_units,
        present,
        profile,
        age_full,
        staff,
        h,
        down,
        up,
        u_asleep,
        u_pre,
        u_mode,
        u_cf,
        u_rest,
        u_route,
        specs,
        cfg,
        p,
    )


def _assign_care_staff(
    building: Building,
    units: list[Node],
    care: NDArray[np.bool_],
    present: NDArray[np.bool_],
    profile: NDArray[np.int8],
    age_full: NDArray[np.int8],
    staff: NDArray[np.bool_],
    cfg: PopulationConfig,
    p: Params,
) -> None:
    """Place care staff as escorts in the most dependent care rooms of each floor.

    Staff numbers follow ``population.care_staff_per_resident`` (day / night);
    each staff member escorts one room's residents. Rooms without an escort
    that contain someone needing stair assistance wait for rescue (or a lift).
    """
    ratio = p.value("population.care_staff_per_resident")[
        "night" if cfg.time_slot.is_night else "day"
    ]
    S = MAX_MEMBERS
    levels: dict[int, list[int]] = {}
    for i, n in enumerate(units):
        if care[i]:
            levels.setdefault(n.level, []).append(i)
    for rooms in levels.values():
        residents = int(present[rooms, :S].sum())
        if residents == 0:
            continue
        n_staff = max(1, round(ratio * residents))

        # Most dependent rooms first: wheelchair users, then frail residents.
        def need(i: int) -> tuple[int, int]:
            here = profile[i, :S][present[i, :S]]
            return (
                -int((here == Profile.WHEELCHAIR_USER).sum()),
                -int((here == Profile.FRAIL_OLDER_ADULT).sum()),
            )

        order = sorted(rooms, key=need)
        for i in order[:n_staff]:
            if present[i, :S].any():
                present[i, S] = True
                profile[i, S] = Profile.ABLE_ADULT
                age_full[i, S] = AgeClass.ADULT
                staff[i, S] = True


def _build_groups(
    building: Building,
    units: list[Node],
    g_units: NDArray[np.intp],
    present: NDArray[np.bool_],
    profile: NDArray[np.int8],
    age_full: NDArray[np.int8],
    staff: NDArray[np.bool_],
    h: NDArray[np.float64],
    down: NDArray[np.float64],
    up: NDArray[np.float64],
    u_asleep: NDArray[np.float64],
    u_pre: NDArray[np.float64],
    u_mode: NDArray[np.float64],
    u_cf: NDArray[np.float64],
    u_rest: NDArray[np.float64],
    u_route: NDArray[np.float64],
    specs: dict[Profile, ProfileSpec],
    cfg: PopulationConfig,
    p: Params,
) -> Population:
    """Aggregate present occupants of each occupied unit into a moving group."""
    # Restrict every per-unit array to occupied units (rows = groups).
    pres = present[g_units]
    prof = np.where(pres, profile[g_units], 0).astype(np.intp)
    level = np.array([units[int(i)].level for i in g_units], dtype=np.int32)
    G = int(g_units.size)
    lut = {
        name: np.array([getattr(specs[Profile(k)], name) for k in range(len(Profile))])
        for name in (
            "fatigue_min_multiplier",
            "fatigue_efold_floors",
            "premovement_multiplier",
            "space_factor",
        )
    }

    def masked(
        values: NDArray[np.float64], mask: NDArray[np.bool_], fill: float
    ) -> NDArray[np.float64]:
        return np.where(mask, values, fill)

    is_wheel = pres & (prof == Profile.WHEELCHAIR_USER)
    is_frail = pres & (prof == Profile.FRAIL_OLDER_ADULT)
    has_wheel = is_wheel.any(axis=1)
    has_frail = is_frail.any(axis=1)
    escorts = (pres & np.isin(prof, [int(x) for x in ABLE_ESCORTS])).sum(axis=1)

    # --- evacuation mode
    p_lift = p.scalar("behaviour.use_evacuation_lift_probability")
    p_carry = p.scalar("behaviour.carry_down_probability")
    um = u_mode[g_units]
    upstairs = level > 0
    mode = np.full(G, Mode.WALK, dtype=np.int8)
    eligible = has_wheel | (has_frail & cfg.lift_for_frail)
    lift = upstairs & eligible & cfg.evacuation_lifts & (um[:, 0] < p_lift)
    mode[lift] = Mode.WAIT_LIFT
    wheel_up = upstairs & has_wheel & ~lift
    carry = wheel_up & (escorts > 0) & (um[:, 1] < p_carry)
    mode[carry] = Mode.ASSISTED_STAIR
    mode[wheel_up & ~carry] = Mode.WAIT_RESCUE

    # --- speeds: the slowest member sets the pace. Wheelchair users only
    # descend stairs when carried (ASSISTED_STAIR).
    hh = h[g_units]
    dd = down[g_units]
    uu = up[g_units]
    carried = (mode == Mode.ASSISTED_STAIR)[:, None]
    stair_users = pres & (~is_wheel | carried)
    stair_users = np.where(stair_users.any(axis=1)[:, None], stair_users, pres)
    h_speed = masked(hh, pres, np.inf).min(axis=1)
    down_speed = masked(dd, stair_users, np.inf).min(axis=1)
    up_speed = masked(uu, stair_users, np.inf).min(axis=1)
    # Stair speed if everyone, including wheelchair users, is taken down the stairs.
    assisted_down = masked(dd, pres, np.inf).min(axis=1)

    # Fatigue parameters of the member who will be slowest over the full descent.
    floors = np.maximum(level, 1)[:, None].astype(np.float64)
    m_min = lut["fatigue_min_multiplier"][prof]
    efold = lut["fatigue_efold_floors"][prof]
    eff = dd * (m_min + (1.0 - m_min) * np.exp(-floors / efold))
    lim = np.argmin(masked(eff, stair_users, np.inf), axis=1)
    rows = np.arange(G)
    fatigue_min = m_min[rows, lim]
    fatigue_efold = efold[rows, lim]

    # Most dependent member (drives rest stops and reporting).
    rank = np.empty(len(Profile), dtype=np.int64)
    for r, pr in enumerate(DEPENDENCY_ORDER):
        rank[pr] = r
    key_rank = masked(rank[prof].astype(np.float64), pres, np.inf).min(axis=1).astype(np.int64)
    order = np.array([int(pr) for pr in DEPENDENCY_ORDER], dtype=np.int8)
    key = order[key_rank]

    # --- pre-movement
    slot = cfg.time_slot.value
    asleep = u_asleep[g_units] < float(p.value("population.asleep_probability")[slot])
    staffed = staff[g_units].any(axis=1)
    up_ = u_pre[g_units]
    pre = np.where(
        asleep & ~staffed, p["premovement.asleep"].ppf(up_), p["premovement.awake"].ppf(up_)
    )
    pre = pre * masked(lut["premovement_multiplier"][prof], pres, 0.0).max(axis=1)
    space = masked(lut["space_factor"][prof], pres, 0.0).sum(axis=1)

    # --- counter-flow waypoint (walking households only)
    p_cf = (
        cfg.counter_flow_probability
        if cfg.counter_flow_probability is not None
        else p.scalar("behaviour.counter_flow_probability")
    )
    ucf = u_cf[g_units]
    cf = (mode == Mode.WALK) & (ucf[:, 0] < p_cf)
    units_by_level: dict[int, list[str]] = {}
    for n in units:
        units_by_level.setdefault(n.level, []).append(n.id)
    p_same = p.scalar("behaviour.counter_flow_same_floor_probability")
    max_up = int(p.value("behaviour.counter_flow_max_floors_up"))
    waypoint: list[str | None] = [None] * G
    for g in np.flatnonzero(cf):
        waypoint[g] = _counter_flow_target(
            units[int(g_units[g])], units_by_level, ucf[g, 1], ucf[g, 2], p_same, max_up
        )
    has_wp = np.array([w is not None for w in waypoint], dtype=bool)
    dwell = np.where(has_wp, p["behaviour.counter_flow_dwell"].ppf(ucf[:, 3]), 0.0)

    # --- rest stop at a refuge floor on the way down
    refuge_levels = [lv.index for lv in building.levels if lv.kind == LevelKind.REFUGE]
    below = np.array([any(r < lv for r in refuge_levels) for lv in level], dtype=bool)
    rest_prob = p.value("behaviour.refuge_rest_probability")
    rest_lut = np.array([float(rest_prob.get(Profile(k).key, 0.0)) for k in range(len(Profile))])
    ur = u_rest[g_units]
    rests = below & np.isin(mode, [Mode.WALK, Mode.ASSISTED_STAIR]) & (ur[:, 0] < rest_lut[key])
    rest = np.where(rests, p["behaviour.refuge_rest_duration"].ppf(ur[:, 1]), 0.0)

    # --- agents (row-major over present slots keeps members of a group together)
    grp, slot_idx = np.nonzero(pres)
    src_units = g_units[grp]
    return Population(
        agent_profile=profile[src_units, slot_idx].astype(np.int8),
        agent_age=age_full[src_units, slot_idx].astype(np.int8),
        agent_group=grp.astype(np.int32),
        agent_is_staff=staff[src_units, slot_idx].astype(bool),
        group_unit=[units[int(i)].id for i in g_units],
        group_level=level,
        group_size=pres.sum(axis=1).astype(np.int32),
        group_space=space,
        group_premovement=pre,
        group_asleep=asleep,
        group_mode=mode,
        group_h_speed=h_speed,
        group_down_speed=down_speed,
        group_up_speed=up_speed,
        group_assisted_down_speed=assisted_down,
        group_fatigue_min=fatigue_min,
        group_fatigue_efold=fatigue_efold,
        group_key_profile=key,
        group_waypoint=waypoint,
        group_waypoint_dwell=dwell,
        group_refuge_rest=rest,
        group_route_u=u_route[g_units],
        config=cfg,
    )


def _counter_flow_target(
    unit: Node,
    units_by_level: dict[int, list[str]],
    u_floor: float,
    u_unit: float,
    p_same: float,
    max_up: int,
) -> str | None:
    """Pick the unit a counter-flowing household visits first (own floor or up to
    ``max_up`` floors above; the nearest occupied floor is used if needed)."""
    if u_floor < p_same:
        target_level = unit.level
    else:
        offset = 1 + min(int((u_floor - p_same) / max(1 - p_same, 1e-9) * max_up), max_up - 1)
        target_level = unit.level + offset
    levels = sorted(units_by_level)
    if target_level not in units_by_level:
        target_level = min(levels, key=lambda lv: (abs(lv - target_level), -lv))
    options = [u for u in units_by_level[target_level] if u != unit.id]
    if not options:
        return None
    return options[min(int(u_unit * len(options)), len(options) - 1)]
