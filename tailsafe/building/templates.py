"""Procedural generators for common Hong Kong residential typologies.

Each generator returns a validated :class:`~tailsafe.building.model.Building`.
Storey counts include the ground floor: a 40-storey block has levels 0 (G/F)
to 39 (39/F). Default dimensions come from ``building_defaults`` in
``config/params.yaml``.

Typologies
----------
``cruciform``
    Public-housing block with four double-loaded wings around a central core
    (lift lobby + two enclosed staircases), optional stairs at the wing ends.
``slab``
    Long double-loaded corridor with a staircase at each end and a central lift core.
``twin_core``
    Private tower whose central core holds a *twin* pair of interlocking (scissor)
    staircases, each reached through a protected lobby, on top of a podium.
``care_home``
    Low-rise residential care home for the elderly: care rooms along a corridor,
    nurse station at the lift lobby, staircases at both ends.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass
from typing import Any

from tailsafe.building.builder import (
    Bay,
    BayKind,
    BuildingBuilder,
    Column,
    CorridorDoors,
    Transform,
    add_corridor,
    hk_level_label,
    rect,
    refuge_levels,
)
from tailsafe.building.model import Building, EdgeKind, LevelKind, NodeType, Point, StairKind
from tailsafe.config import Params, get_params


@dataclass(frozen=True)
class Dimensions:
    """Default geometry read from ``building_defaults`` in the parameter registry."""

    floor_height: float
    ground_height: float
    stair_width: float
    riser: float
    going: float
    stair_door: float
    flat_door: float
    exit_width: float
    corridor_width: float
    refuge_interval: int
    refuge_min_storeys: int

    @classmethod
    def from_params(cls, params: Params | None = None) -> Dimensions:
        """Read defaults from the registry."""
        p = params or get_params()
        g = "building_defaults."
        return cls(
            floor_height=p.scalar(g + "floor_to_floor_height"),
            ground_height=p.scalar(g + "ground_floor_height"),
            stair_width=p.scalar(g + "stair_clear_width"),
            riser=p.scalar(g + "stair_riser"),
            going=p.scalar(g + "stair_going"),
            stair_door=p.scalar(g + "stair_door_width"),
            flat_door=p.scalar(g + "flat_door_width"),
            exit_width=p.scalar(g + "final_exit_width"),
            corridor_width=p.scalar(g + "corridor_width"),
            refuge_interval=int(p.value(g + "refuge_floor_interval")),
            refuge_min_storeys=int(p.value(g + "refuge_floor_min_storeys")),
        )

    @property
    def doors(self) -> CorridorDoors:
        """Door/corridor widths bundle for :func:`add_corridor`."""
        return CorridorDoors(
            unit_door=self.flat_door,
            stair_door=self.stair_door,
            corridor_width=self.corridor_width,
        )


def _stack_levels(
    bld: BuildingBuilder,
    storeys: int,
    dims: Dimensions,
    refuge: AbstractSet[int],
    podium: AbstractSet[int] = frozenset(),
) -> None:
    elevation = 0.0
    for i in range(storeys):
        h = dims.ground_height if i == 0 else dims.floor_height
        if i == 0:
            kind = LevelKind.GROUND
        elif i in refuge:
            kind = LevelKind.REFUGE
        elif i in podium:
            kind = LevelKind.PODIUM
        else:
            kind = LevelKind.TYPICAL
        bld.level(i, elevation=elevation, height=h, kind=kind)
        elevation += h


def _add_stairs(
    bld: BuildingBuilder, ids: list[str], dims: Dimensions, kind: StairKind, shaft: str | None
) -> None:
    for sid in ids:
        bld.stair(
            sid,
            f"Stair {sid}",
            kind=kind,
            clear_width=dims.stair_width,
            riser=dims.riser,
            going=dims.going,
            shaft=shaft,
        )


def _bounds(poly: list[Point]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _discharge(
    bld: BuildingBuilder,
    from_node: str,
    exit_id: str,
    direction: str,
    width: float,
    label: str,
    offset: float = 2.0,
) -> None:
    """Add a final exit just outside ``from_node``'s polygon edge in ``direction`` (N/S/E/W)."""
    n = bld.nodes[from_node]
    assert n.polygon is not None
    x0, y0, x1, y1 = _bounds(n.polygon)
    h = width / 2.0
    out: Point
    opening: tuple[Point, Point]
    if direction == "N":
        out, opening = (n.x, y1 + offset), ((n.x - h, y1), (n.x + h, y1))
    elif direction == "S":
        out, opening = (n.x, y0 - offset), ((n.x - h, y0), (n.x + h, y0))
    elif direction == "E":
        out, opening = (x1 + offset, n.y), ((x1, n.y - h), (x1, n.y + h))
    elif direction == "W":
        out, opening = (x0 - offset, n.y), ((x0, n.y - h), (x0, n.y + h))
    else:
        raise ValueError(f"direction must be N, S, E or W, not {direction!r}")
    bld.node(exit_id, NodeType.EXIT, 0, at=out, label=label, polygon=None)
    bld.edge(
        from_node,
        exit_id,
        EdgeKind.DOOR,
        width=width,
        opening=opening,
        can_block=True,
        label=f"{label} door",
    )


def _metadata(generator: str, args: dict[str, Any], params: Params | None) -> dict[str, object]:
    p = params or get_params()
    return {"generator": generator, "args": args, "params_digest": p.digest}


def _unit_type(col: int, n_cols: int) -> str:
    if col == n_cols - 1 and n_cols >= 3:
        return "flat_large"
    if col == 0 and n_cols >= 2:
        return "flat_small"
    return "flat_medium"


# ============================================================================ cruciform
def generate_cruciform(
    storeys: int = 40,
    flats_per_wing: int = 5,
    lifts: int = 3,
    wing_end_stairs: bool = False,
    flat_width: float = 6.0,
    flat_depth: float = 8.0,
    params: Params | None = None,
) -> Building:
    """Cruciform public-housing block: four wings around a central lift/stair core.

    Core (plan, metres): lift lobby [-4, 4]²; Stair A in the north-west block,
    Stair B in the south-east block, lift shafts north-east, refuse room
    south-west. Wings run east, north, west and south from the lobby.
    """
    if storeys < 2 or flats_per_wing < 1 or lifts < 0:
        raise ValueError("need storeys >= 2, flats_per_wing >= 1, lifts >= 0")
    args = {
        "storeys": storeys,
        "flats_per_wing": flats_per_wing,
        "lifts": lifts,
        "wing_end_stairs": wing_end_stairs,
        "flat_width": flat_width,
        "flat_depth": flat_depth,
    }
    dims = Dimensions.from_params(params)
    bld = BuildingBuilder(
        id=f"cruciform-{storeys}",
        name=f"Cruciform public housing block, {storeys} storeys",
        typology="cruciform",
        description=(
            f"{flats_per_wing * 4} flats per typical floor on four double-loaded wings "
            f"around a central core with two staircases and {lifts} lift(s)."
        ),
        metadata=_metadata("cruciform", args, params),
    )
    refuge = set(refuge_levels(storeys, dims.refuge_interval, dims.refuge_min_storeys))
    _stack_levels(bld, storeys, dims, refuge)
    wings = {"E": 0.0, "N": 90.0, "W": 180.0, "S": 270.0}
    end_stairs = {"E": "C", "N": "D", "W": "E", "S": "F"}
    core_ids = ["A", "B"]
    _add_stairs(bld, core_ids, dims, StairKind.DOGLEG, None)
    if wing_end_stairs:
        _add_stairs(bld, list(end_stairs.values()), dims, StairKind.DOGLEG, None)

    half_c = dims.corridor_width / 2.0
    n_cols = math.ceil(flats_per_wing / 2)
    # The corridor leaves the lobby at 4 m; flats start once clear of the next wing's flats.
    stub_len = max(3.0, half_c + flat_depth - 4.0)
    lobby_ids: list[tuple[int, str]] = []
    for lv in range(storeys):
        pre = f"L{lv:02d}"
        lab = hk_level_label(lv)
        lobby = f"{pre}.lobby"
        bld.node(
            lobby,
            NodeType.LIFT_LOBBY,
            lv,
            polygon=rect(-4, -4, 4, 4),
            label=("Entrance lobby" if lv == 0 else "Lift lobby") + f" at {lab}",
        )
        lobby_ids.append((lv, lobby))
        for sid, poly, door_x, door_y in (
            ("A", rect(-7, 4, -half_c, 7), -2.5, 4.0),
            ("B", rect(half_c, -7, 7, -4), 2.5, -4.0),
        ):
            land = f"{pre}.stair.{sid}"
            bld.node(
                land,
                NodeType.STAIR_LANDING,
                lv,
                polygon=poly,
                label=f"Stair {sid} landing at {lab}",
                stair=sid,
            )
            w = dims.stair_door / 2.0
            bld.edge(
                lobby,
                land,
                EdgeKind.DOOR,
                width=dims.stair_door,
                opening=((door_x - w, door_y), (door_x + w, door_y)),
                fire_rated=True,
                self_closing=True,
                can_block=True,
                label=f"Stair {sid} door at {lab}",
            )
        # Core features (drawn only).
        for k in range(lifts):
            x0 = half_c + (7 - half_c) * k / max(lifts, 1)
            x1 = half_c + (7 - half_c) * (k + 1) / max(lifts, 1)
            bld.feature(lv, "lift_shaft", rect(x0, 4, x1, 7), f"Lift {k + 1}")
        bld.feature(lv, "service", rect(-7, -7, -half_c, -4), "Refuse room")
        for sx in (-1, 1):
            for sy in (-1, 1):
                bld.feature(lv, "service", rect(4 * sx, half_c * sy, 7 * sx, 4 * sy), None)

        for wing, angle in wings.items():
            tf = Transform(angle)
            stub = Column(width=stub_len)
            end_bay = (
                Bay("stair", depth=5.0, stairs=(end_stairs[wing],)) if wing_end_stairs else None
            )
            if lv == 0:
                # Ground floor: N/S entrance passages, plus corridors to any wing-end stairs.
                if wing not in ("N", "S") and not wing_end_stairs:
                    continue
                cols = [stub, Column(n_cols * flat_width)] if wing_end_stairs else [stub]
                res = add_corridor(
                    bld,
                    lv,
                    prefix=pre,
                    name=wing,
                    columns=cols,
                    doors=dims.doors,
                    start_x=4.0,
                    transform=tf,
                    east_end=end_bay,
                )
                bld.edge(lobby, res.segments[0], EdgeKind.FLAT, width=dims.corridor_width)
                if wing in ("N", "S"):
                    # Leave through the end of the passage, or sideways if a stair is there.
                    seg, direction = (
                        (res.segments[-1], _side_of(wing))
                        if wing_end_stairs
                        else (res.segments[0], wing)
                    )
                    _discharge(
                        bld,
                        seg,
                        f"{pre}.exit.{wing}",
                        direction,
                        dims.exit_width,
                        f"Main entrance ({wing})",
                    )
                if wing_end_stairs:
                    sid = end_stairs[wing]
                    _discharge(
                        bld,
                        res.landings[sid],
                        f"{pre}.exit.{sid}",
                        _compass(angle),
                        dims.exit_width,
                        f"Stair {sid} discharge",
                    )
                continue
            if lv in refuge:
                cols = [
                    stub,
                    Column(
                        n_cols * flat_width,
                        north=Bay("refuge", name=f"{wing}n", depth=flat_depth),
                        south=Bay("refuge", name=f"{wing}s", depth=flat_depth),
                    ),
                ]
            else:
                cols = [stub]
                flat_no = 0
                for c in range(n_cols):
                    bays: list[Bay | None] = []
                    for _side in range(2):
                        flat_no += 1
                        if flat_no > flats_per_wing:
                            bays.append(None)
                        else:
                            bays.append(
                                Bay(
                                    "unit",
                                    name=f"{wing}{flat_no}",
                                    depth=flat_depth,
                                    unit_type=_unit_type(c, n_cols),
                                )
                            )
                    cols.append(Column(flat_width, north=bays[0], south=bays[1]))
            res = add_corridor(
                bld,
                lv,
                prefix=pre,
                name=wing,
                columns=cols,
                doors=dims.doors,
                start_x=4.0,
                transform=tf,
                east_end=end_bay,
            )
            bld.edge(lobby, res.segments[0], EdgeKind.FLAT, width=dims.corridor_width)

    # G/F: core stairs discharge straight outside.
    _discharge(bld, "L00.stair.A", "L00.exit.A", "W", dims.exit_width, "Stair A discharge")
    _discharge(bld, "L00.stair.B", "L00.exit.B", "E", dims.exit_width, "Stair B discharge")
    bld.connect_stair_flights()
    for k in range(lifts):
        bld.lift(f"lift{k + 1}", f"Lift {k + 1}", lobby_ids, firefighting=(k == lifts - 1))
    return bld.build()


def _compass(angle: float) -> str:
    return {0.0: "E", 90.0: "N", 180.0: "W", 270.0: "S"}[angle % 360.0]


def _side_of(wing: str) -> str:
    # Ground-floor passages run N/S; leave sideways into the open ground floor.
    return "E" if wing == "N" else "W"


# ============================================================================ slab
def generate_slab(
    storeys: int = 30,
    flats_per_side: int = 10,
    lifts: int = 2,
    flat_width: float = 6.0,
    flat_depth: float = 8.0,
    params: Params | None = None,
) -> Building:
    """Slab block: long double-loaded corridor, end staircases, central lift core."""
    if storeys < 2 or flats_per_side < 2:
        raise ValueError("need storeys >= 2 and flats_per_side >= 2")
    args = {
        "storeys": storeys,
        "flats_per_side": flats_per_side,
        "lifts": lifts,
        "flat_width": flat_width,
        "flat_depth": flat_depth,
    }
    dims = Dimensions.from_params(params)
    bld = BuildingBuilder(
        id=f"slab-{storeys}",
        name=f"Slab block, {storeys} storeys",
        typology="slab",
        description=(
            f"{2 * flats_per_side} flats per typical floor on a central corridor with "
            f"end staircases and {lifts} lift(s) at mid-length."
        ),
        metadata=_metadata("slab", args, params),
    )
    refuge = set(refuge_levels(storeys, dims.refuge_interval, dims.refuge_min_storeys))
    _stack_levels(bld, storeys, dims, refuge)
    _add_stairs(bld, ["A", "B"], dims, StairKind.DOGLEG, None)
    left = flats_per_side // 2
    right = flats_per_side - left
    core_w = 6.0
    lobby_ids: list[tuple[int, str]] = []
    for lv in range(storeys):
        pre = f"L{lv:02d}"
        core = Column(
            core_w,
            north=Bay("lift_lobby", name="core", depth=4.0),
            south=Bay("service", depth=4.0, label="Refuse room"),
        )
        if lv == 0 or lv in refuge:
            kind: BayKind = "void" if lv == 0 else "refuge"
            cols = [
                Column(
                    left * flat_width,
                    north=Bay(kind, name="Wn", depth=flat_depth),
                    south=Bay(kind, name="Ws", depth=flat_depth),
                ),
                core,
                Column(
                    right * flat_width,
                    north=Bay(kind, name="En", depth=flat_depth),
                    south=Bay(kind, name="Es", depth=flat_depth),
                ),
            ]
        else:
            cols = []
            for i in range(flats_per_side):
                if i == left:
                    cols.append(core)
                ut = "flat_large" if i in (0, flats_per_side - 1) else "flat_medium"
                cols.append(
                    Column(
                        flat_width,
                        north=Bay("unit", name=f"{i + 1:02d}", depth=flat_depth, unit_type=ut),
                        south=Bay(
                            "unit",
                            name=f"{flats_per_side + i + 1:02d}",
                            depth=flat_depth,
                            unit_type=ut,
                        ),
                    )
                )
        res = add_corridor(
            bld,
            lv,
            prefix=pre,
            name="",
            columns=cols,
            doors=dims.doors,
            west_end=Bay("stair", depth=5.0, stairs=("A",)),
            east_end=Bay("stair", depth=5.0, stairs=("B",)),
        )
        lobby = res.lift_lobbies[0]
        lobby_ids.append((lv, lobby))
        lobby_node = bld.nodes[lobby]
        assert lobby_node.polygon is not None
        x0, _, x1, y1 = _bounds(lobby_node.polygon)
        for k in range(lifts):
            lx0 = x0 + (x1 - x0) * k / max(lifts, 1)
            lx1 = x0 + (x1 - x0) * (k + 1) / max(lifts, 1)
            bld.feature(lv, "lift_shaft", rect(lx0, y1, lx1, y1 + 2.5), f"Lift {k + 1}")
        if lv == 0:
            _discharge(bld, lobby, "L00.exit.main", "N", dims.exit_width, "Main entrance")
            _discharge(bld, res.landings["A"], "L00.exit.A", "W", dims.exit_width, "Stair A exit")
            _discharge(bld, res.landings["B"], "L00.exit.B", "E", dims.exit_width, "Stair B exit")
    bld.connect_stair_flights()
    for k in range(lifts):
        bld.lift(f"lift{k + 1}", f"Lift {k + 1}", lobby_ids, firefighting=(k == lifts - 1))
    return bld.build()


# ============================================================================ twin core
def generate_twin_core(
    storeys: int = 45,
    podium_levels: int = 3,
    flats_per_floor: int = 8,
    lifts: int = 3,
    protected_lobbies: bool = True,
    flat_width: float = 7.0,
    flat_depth: float = 9.0,
    params: Params | None = None,
) -> Building:
    """Private residential tower with a twin (scissor) staircase core on a podium.

    Both staircases share one shaft (``shaft = "S1"``) but are separately
    enclosed; each is reached through its own protected lobby. Podium levels
    (car park / clubhouse) have open areas and no flats.
    """
    if storeys < podium_levels + 2 or flats_per_floor < 2 or flats_per_floor % 2:
        raise ValueError("need storeys >= podium_levels + 2 and an even flats_per_floor >= 2")
    args = {
        "storeys": storeys,
        "podium_levels": podium_levels,
        "flats_per_floor": flats_per_floor,
        "lifts": lifts,
        "protected_lobbies": protected_lobbies,
        "flat_width": flat_width,
        "flat_depth": flat_depth,
    }
    dims = Dimensions.from_params(params)
    bld = BuildingBuilder(
        id=f"twin_core-{storeys}",
        name=f"Private tower with scissor-stair core, {storeys} storeys",
        typology="twin_core",
        description=(
            f"{flats_per_floor} flats per typical floor around a central core with a "
            f"scissor staircase pair and {lifts} lift(s), above a {podium_levels}-level podium."
        ),
        metadata=_metadata("twin_core", args, params),
    )
    refuge = set(refuge_levels(storeys, dims.refuge_interval, dims.refuge_min_storeys))
    podium = set(range(1, podium_levels + 1)) - refuge
    _stack_levels(bld, storeys, dims, refuge, podium)
    _add_stairs(bld, ["A", "B"], dims, StairKind.SCISSOR, "S1")
    per_side = flats_per_floor // 2
    left = per_side // 2
    right = per_side - left
    pl_depth = 2.0 if protected_lobbies else 0.0
    core = Column(
        6.4,
        north=Bay("lift_lobby", name="core", depth=3.5),
        south=Bay(
            "stair", depth=pl_depth + 4.5, stairs=("A", "B"), protected_lobby=protected_lobbies
        ),
    )
    lobby_ids: list[tuple[int, str]] = []
    for lv in range(storeys):
        pre = f"L{lv:02d}"
        if lv == 0 or lv in refuge or lv in podium:
            kind: BayKind = "void" if lv == 0 else ("refuge" if lv in refuge else "open")
            cols = [
                Column(
                    left * flat_width,
                    north=Bay(kind, name="Wn", depth=flat_depth),
                    south=Bay(kind, name="Ws", depth=flat_depth),
                ),
                core,
                Column(
                    right * flat_width,
                    north=Bay(kind, name="En", depth=flat_depth),
                    south=Bay(kind, name="Es", depth=flat_depth),
                ),
            ]
        else:
            cols = []
            for i in range(per_side):
                if i == left:
                    cols.append(core)
                ut = "flat_large" if i in (0, per_side - 1) else "flat_medium"
                cols.append(
                    Column(
                        flat_width,
                        north=Bay("unit", name=f"{chr(65 + i)}", depth=flat_depth, unit_type=ut),
                        south=Bay(
                            "unit",
                            name=f"{chr(65 + per_side + i)}",
                            depth=flat_depth,
                            unit_type=ut,
                        ),
                    )
                )
        res = add_corridor(bld, lv, prefix=pre, name="", columns=cols, doors=dims.doors)
        lobby = res.lift_lobbies[0]
        lobby_ids.append((lv, lobby))
        ln = bld.nodes[lobby]
        assert ln.polygon is not None
        x0, _, x1, y1 = _bounds(ln.polygon)
        for k in range(lifts):
            lx0 = x0 + (x1 - x0) * k / max(lifts, 1)
            lx1 = x0 + (x1 - x0) * (k + 1) / max(lifts, 1)
            bld.feature(lv, "lift_shaft", rect(lx0, y1, lx1, y1 + 2.5), f"Lift {k + 1}")
        if lv == 0:
            first, last = res.segments[0], res.segments[-1]
            _discharge(bld, first, "L00.exit.W", "W", dims.exit_width, "Main entrance (west)")
            _discharge(bld, last, "L00.exit.E", "E", dims.exit_width, "Side entrance (east)")
            for sid in ("A", "B"):
                _discharge(
                    bld,
                    res.landings[sid],
                    f"L00.exit.{sid}",
                    "S",
                    dims.exit_width,
                    f"Stair {sid} discharge",
                )
    bld.connect_stair_flights()
    for k in range(lifts):
        bld.lift(f"lift{k + 1}", f"Lift {k + 1}", lobby_ids, firefighting=(k == lifts - 1))
    return bld.build()


# ============================================================================ care home
def generate_care_home(
    storeys: int = 4,
    rooms_per_side: int = 6,
    lifts: int = 1,
    room_width: float = 4.5,
    room_depth: float = 6.0,
    params: Params | None = None,
) -> Building:
    """Low-rise residential care home for the elderly (high-dependency occupants)."""
    if storeys < 1 or rooms_per_side < 2:
        raise ValueError("need storeys >= 1 and rooms_per_side >= 2")
    args = {
        "storeys": storeys,
        "rooms_per_side": rooms_per_side,
        "lifts": lifts,
        "room_width": room_width,
        "room_depth": room_depth,
    }
    dims = Dimensions.from_params(params)
    bld = BuildingBuilder(
        id=f"care_home-{storeys}",
        name=f"Residential care home for the elderly, {storeys} storeys",
        typology="care_home",
        description=(
            f"{2 * rooms_per_side} care rooms per floor, nurse station at the lift lobby, "
            "staircases at both ends."
        ),
        metadata=_metadata("care_home", args, params),
    )
    _stack_levels(bld, storeys, dims, set())
    _add_stairs(bld, ["A", "B"], dims, StairKind.DOGLEG, None)
    left = rooms_per_side // 2
    lobby_ids: list[tuple[int, str]] = []
    for lv in range(storeys):
        pre = f"L{lv:02d}"
        cols = []
        for i in range(rooms_per_side):
            if i == left:
                cols.append(
                    Column(
                        5.0,
                        north=Bay(
                            "lift_lobby",
                            name="nurse",
                            depth=room_depth,
                            label="Nurse station / lift lobby",
                            tags=("staff_station",),
                        ),
                        south=Bay("service", depth=room_depth, label="Sluice and store"),
                    )
                )
            cols.append(
                Column(
                    room_width,
                    north=Bay(
                        "unit",
                        name=f"{i + 1:02d}",
                        depth=room_depth,
                        unit_type="care_room",
                        label="Room",
                        tags=("care_home",),
                    ),
                    south=Bay(
                        "unit",
                        name=f"{rooms_per_side + i + 1:02d}",
                        depth=room_depth,
                        unit_type="care_room",
                        label="Room",
                        tags=("care_home",),
                    ),
                )
            )
        res = add_corridor(
            bld,
            lv,
            prefix=pre,
            name="",
            columns=cols,
            doors=dims.doors,
            west_end=Bay("stair", depth=5.0, stairs=("A",)),
            east_end=Bay("stair", depth=5.0, stairs=("B",)),
        )
        lobby = res.lift_lobbies[0]
        lobby_ids.append((lv, lobby))
        if lv == 0:
            _discharge(bld, lobby, "L00.exit.main", "N", dims.exit_width, "Main entrance")
            _discharge(bld, res.landings["A"], "L00.exit.A", "W", dims.exit_width, "Stair A exit")
            _discharge(bld, res.landings["B"], "L00.exit.B", "E", dims.exit_width, "Stair B exit")
    bld.connect_stair_flights()
    for k in range(lifts):
        bld.lift(f"lift{k + 1}", f"Lift {k + 1}", lobby_ids, firefighting=(k == lifts - 1))
    return bld.build()


# ============================================================================ registry
TEMPLATES: dict[str, Callable[..., Building]] = {
    "cruciform": generate_cruciform,
    "slab": generate_slab,
    "twin_core": generate_twin_core,
    "care_home": generate_care_home,
}


def generate(template: str, **kwargs: Any) -> Building:
    """Generate a building from a named template with keyword overrides."""
    try:
        fn = TEMPLATES[template]
    except KeyError:
        raise ValueError(
            f"unknown template {template!r}; choose from {sorted(TEMPLATES)}"
        ) from None
    return fn(**kwargs)


def dimensions_dict(params: Params | None = None) -> dict[str, Any]:
    """Default dimensions as a plain dict (for display)."""
    return asdict(Dimensions.from_params(params))
