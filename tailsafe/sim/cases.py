"""Idealised buildings and populations for analytical validation.

These cases have closed-form answers from the hydraulic method, so the
mesoscopic engine can be checked against hand calculations
(``tests/validation/``, ``docs/validation.md``).
"""

from __future__ import annotations

import numpy as np

from tailsafe.building.builder import BuildingBuilder, rect
from tailsafe.building.model import Building, EdgeKind, LevelKind, NodeType, StairKind
from tailsafe.population.profiles import AgeClass, Profile
from tailsafe.population.synth import Mode, Population, PopulationConfig


def corridor_building(
    length: float = 30.0, width: float = 2.0, exit_width: float = 1.0, room_area: float = 200.0
) -> Building:
    """A holding room, a corridor of ``length`` × ``width`` and an exit door.

    The room's door is as wide as the corridor, so the only constriction is the
    exit door (unless the corridor itself is narrower).
    """
    bld = BuildingBuilder(id="case-corridor", name="Corridor case", typology="custom")
    bld.level(0, elevation=0.0, height=3.0, kind=LevelKind.GROUND)
    bld.node(
        "room", NodeType.UNIT, 0, polygon=rect(-10, -10, 0, 10), area=room_area, unit_type="case"
    )
    bld.node("corridor_in", NodeType.CORRIDOR, 0, at=(0.0, 0.0), area=width * 0.5)
    bld.node("corridor_out", NodeType.CORRIDOR, 0, at=(length, 0.0), area=width * 0.5)
    bld.node("exit", NodeType.EXIT, 0, at=(length + 0.5, 0.0))
    bld.edge("room", "corridor_in", EdgeKind.FLAT, width=width, length=0.5)
    bld.edge("corridor_in", "corridor_out", EdgeKind.FLAT, width=width, length=length)
    bld.edge("corridor_out", "exit", EdgeKind.DOOR, width=exit_width, length=0.5)
    return bld.build()


def stair_tower(
    floors: int = 10,
    stair_width: float = 1.2,
    door_width: float = 1.2,
    floor_height: float = 3.0,
    riser: float = 0.175,
    going: float = 0.28,
    exit_width: float = 2.0,
) -> Building:
    """A tower with one dog-leg stair: on each upper floor a room opens straight
    onto the stair landing; the stair discharges to an exit at ground level."""
    bld = BuildingBuilder(id="case-stair", name="Single-stair tower", typology="custom")
    for lv in range(floors + 1):
        bld.level(
            lv,
            elevation=lv * floor_height,
            height=floor_height,
            kind=LevelKind.GROUND if lv == 0 else LevelKind.TYPICAL,
        )
    bld.stair(
        "A",
        "Stair A",
        kind=StairKind.DOGLEG,
        clear_width=stair_width,
        riser=riser,
        going=going,
    )
    for lv in range(floors + 1):
        bld.node(
            f"L{lv}.landing",
            NodeType.STAIR_LANDING,
            lv,
            polygon=rect(0, 0, 3, 6),
            stair="A",
            label=f"Stair A landing at L{lv}",
        )
        if lv > 0:
            bld.node(
                f"L{lv}.room",
                NodeType.UNIT,
                lv,
                polygon=rect(-20, 0, 0, 10),
                unit_type="case",
            )
            bld.edge(f"L{lv}.room", f"L{lv}.landing", EdgeKind.DOOR, width=door_width, length=2.0)
    bld.node("exit", NodeType.EXIT, 0, at=(5.0, 3.0))
    bld.edge("L0.landing", "exit", EdgeKind.DOOR, width=exit_width, length=1.0)
    bld.connect_stair_flights()
    return bld.build()


def uniform_population(
    building: Building,
    persons_per_unit: int,
    *,
    h_speed: float = 1.19,
    down_speed: float = 0.92,
    up_speed: float = 0.6,
    premovement: float = 0.0,
    space: float = 1.0,
) -> Population:
    """Identical individual occupants (groups of one), all ready at ``premovement``.

    Default speeds are the hydraulic model's unimpeded speeds (0.85 k) for
    corridors (k = 1.40) and 7/11 stairs (k = 1.08). No fatigue.
    """
    units = building.units
    G = len(units) * persons_per_unit
    unit_ids = [u.id for u in units for _ in range(persons_per_unit)]
    levels = np.array([u.level for u in units for _ in range(persons_per_unit)], dtype=np.int32)
    ones = np.ones(G)
    return Population(
        agent_profile=np.full(G, Profile.ABLE_ADULT, dtype=np.int8),
        agent_age=np.full(G, AgeClass.ADULT, dtype=np.int8),
        agent_group=np.arange(G, dtype=np.int32),
        agent_is_staff=np.zeros(G, dtype=bool),
        group_unit=unit_ids,
        group_level=levels,
        group_size=np.ones(G, dtype=np.int32),
        group_space=ones * space,
        group_premovement=ones * premovement,
        group_asleep=np.zeros(G, dtype=bool),
        group_mode=np.full(G, Mode.WALK, dtype=np.int8),
        group_h_speed=ones * h_speed,
        group_down_speed=ones * down_speed,
        group_up_speed=ones * up_speed,
        group_fatigue_min=ones,
        group_fatigue_efold=ones * 1e9,
        group_key_profile=np.full(G, Profile.ABLE_ADULT, dtype=np.int8),
        group_waypoint=[None] * G,
        group_waypoint_dwell=np.zeros(G),
        group_refuge_rest=np.zeros(G),
        group_route_u=np.full(G, 0.5),
        config=PopulationConfig(),
    )
