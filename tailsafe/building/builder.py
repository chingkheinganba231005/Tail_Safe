"""Incremental construction of :class:`~tailsafe.building.model.Building` objects.

The procedural templates describe a floor as a *double-loaded corridor*: a
corridor running along local +x with "bays" (flats, stair enclosures, lift
lobbies, refuge areas, ...) on its north (+y) and south (-y) sides, grouped in
columns. :func:`add_corridor` turns such a description into nodes, edges and
geometry, optionally rotated/translated into place (cruciform wings are four
rotated corridors around a core).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from tailsafe.building.model import (
    Building,
    Edge,
    EdgeKind,
    Feature,
    Level,
    LevelKind,
    Lift,
    LiftStop,
    Node,
    NodeType,
    Point,
    Stair,
    StairKind,
    polygon_area,
)
from tailsafe.building.validate import check_building

_ROUND = 3  # millimetre precision keeps the JSON compact


# ============================================================================ geometry
def rect(x0: float, y0: float, x1: float, y1: float) -> list[Point]:
    """Axis-aligned rectangle as a counter-clockwise polygon."""
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return [(xa, ya), (xb, ya), (xb, yb), (xa, yb)]


def centroid(poly: Sequence[Point]) -> Point:
    """Area centroid of a simple polygon (vertex mean for degenerate input)."""
    a = cx = cy = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        a += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(a) < 1e-12:
        return (sum(p[0] for p in poly) / n, sum(p[1] for p in poly) / n)
    return (cx / (3 * a), cy / (3 * a))


def dist(p: Point, q: Point) -> float:
    """Euclidean distance."""
    return math.hypot(p[0] - q[0], p[1] - q[1])


@dataclass(frozen=True)
class Transform:
    """Rotation (degrees, counter-clockwise about the origin) then translation."""

    angle_deg: float = 0.0
    dx: float = 0.0
    dy: float = 0.0

    def point(self, p: Point) -> Point:
        """Transform a point."""
        a = math.radians(self.angle_deg)
        c, s = math.cos(a), math.sin(a)
        x, y = p
        return (c * x - s * y + self.dx, s * x + c * y + self.dy)

    def poly(self, poly: Iterable[Point]) -> list[Point]:
        """Transform a polygon (orientation is preserved by rotations)."""
        return [self.point(p) for p in poly]


IDENTITY = Transform()


def hk_level_label(index: int) -> str:
    """Hong Kong style storey label: 0 -> 'G/F', 14 -> '14/F', -1 -> 'B1/F'."""
    if index == 0:
        return "G/F"
    if index < 0:
        return f"B{-index}/F"
    return f"{index}/F"


def refuge_levels(storeys: int, interval: int, min_storeys: int) -> list[int]:
    """Levels that are refuge floors for a building of ``storeys`` (incl. G/F).

    Buildings taller than ``min_storeys`` get a refuge floor every ``interval``
    levels above ground, never on the top storey.
    """
    if storeys <= min_storeys or interval <= 0:
        return []
    top = storeys - 1
    return [lv for lv in range(interval, top, interval)]


def stair_travel_length(rise: float, stair: Stair) -> float:
    """Walking distance along the line of travel for one storey of stair.

    Dog-leg stairs have two flights and a 180° turn on the half landing
    (approximated as a semicircle of radius half the clear width). Scissor and
    straight stairs have a single flight per storey.
    """
    flights = 2 if stair.kind == StairKind.DOGLEG else 1
    flight_rise = rise / flights
    risers = math.ceil(flight_rise / stair.riser - 1e-9)
    run = risers * stair.going
    slope = math.hypot(run, flight_rise)
    turns = (flights - 1) * math.pi * stair.clear_width / 2.0
    return flights * slope + turns


def _r(v: float) -> float:
    return round(v, _ROUND)


def _rp(p: Point) -> Point:
    return (_r(p[0]), _r(p[1]))


# ============================================================================ builder
class BuildingBuilder:
    """Accumulates levels, nodes and edges, then validates into a Building."""

    def __init__(
        self,
        *,
        id: str,
        name: str,
        typology: str,
        description: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.typology = typology
        self.description = description
        self.metadata: dict[str, object] = dict(metadata or {})
        self.levels: dict[int, Level] = {}
        self.stairs: dict[str, Stair] = {}
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.lifts: list[Lift] = []

    # ---------------------------------------------------------------- levels
    def level(self, index: int, *, elevation: float, height: float, kind: LevelKind) -> Level:
        """Add a storey."""
        lv = Level(
            index=index,
            label=hk_level_label(index),
            elevation=_r(elevation),
            height=_r(height),
            kind=kind,
        )
        self.levels[index] = lv
        return lv

    def feature(
        self, level: int, kind: str, poly: Sequence[Point], label: str | None = None
    ) -> None:
        """Add a non-walkable plan feature (lift shaft, service room, void)."""
        self.levels[level].features.append(
            Feature(kind=kind, polygon=[_rp(p) for p in poly], label=label)
        )

    def stair(
        self,
        id: str,
        label: str,
        *,
        kind: StairKind,
        clear_width: float,
        riser: float,
        going: float,
        shaft: str | None = None,
    ) -> Stair:
        """Register a staircase."""
        s = Stair(
            id=id,
            label=label,
            kind=kind,
            clear_width=clear_width,
            riser=riser,
            going=going,
            shaft=shaft,
        )
        self.stairs[id] = s
        return s

    # ---------------------------------------------------------------- nodes/edges
    def node(
        self,
        id: str,
        type: NodeType,
        level: int,
        *,
        polygon: Sequence[Point] | None = None,
        at: Point | None = None,
        label: str | None = None,
        area: float | None = None,
        unit_type: str | None = None,
        stair: str | None = None,
        tags: Iterable[str] = (),
    ) -> Node:
        """Add a node. Its reference point defaults to the polygon centroid."""
        if id in self.nodes:
            raise ValueError(f"duplicate node id {id}")
        poly = [_rp(p) for p in polygon] if polygon is not None else None
        if at is None:
            if poly is None:
                raise ValueError(f"node {id} needs a polygon or a position")
            at = centroid(poly)
        n = Node(
            id=id,
            type=type,
            level=level,
            x=_r(at[0]),
            y=_r(at[1]),
            label=label,
            area=None if area is None else _r(area),
            polygon=poly,
            unit_type=unit_type,
            stair=stair,
            tags=list(tags),
        )
        self.nodes[id] = n
        return n

    def edge(
        self,
        source: str,
        target: str,
        kind: EdgeKind,
        *,
        width: float,
        length: float | None = None,
        opening: tuple[Point, Point] | None = None,
        fire_rated: bool = False,
        self_closing: bool = False,
        can_block: bool = False,
        directed: bool = False,
        stair: str | None = None,
        label: str | None = None,
        id: str | None = None,
    ) -> Edge:
        """Add an edge. Length defaults to the walk between reference points,
        passing through the door opening's midpoint when one is given."""
        a, b = self.nodes[source], self.nodes[target]
        if length is None:
            pa, pb = (a.x, a.y), (b.x, b.y)
            if opening is not None:
                mid = ((opening[0][0] + opening[1][0]) / 2, (opening[0][1] + opening[1][1]) / 2)
                length = dist(pa, mid) + dist(mid, pb)
            else:
                length = dist(pa, pb)
        eid = id or f"{source}--{target}"
        if eid in self.edges:
            raise ValueError(f"duplicate edge id {eid}")
        e = Edge(
            id=eid,
            source=source,
            target=target,
            kind=kind,
            length=_r(max(length, 0.5)),
            width=_r(width),
            directed=directed,
            fire_rated=fire_rated,
            self_closing=self_closing,
            can_block=can_block,
            stair=stair,
            opening=None if opening is None else (_rp(opening[0]), _rp(opening[1])),
            label=label,
        )
        self.edges[eid] = e
        return e

    def connect_stair_flights(self) -> None:
        """Add a flight between every pair of vertically adjacent landings of each stair."""
        by_stair: dict[str, dict[int, Node]] = {}
        for n in self.nodes.values():
            if n.type == NodeType.STAIR_LANDING and n.stair is not None:
                by_stair.setdefault(n.stair, {})[n.level] = n
        for sid, landings in by_stair.items():
            stair = self.stairs[sid]
            levels = sorted(landings)
            for lower, upper in itertools.pairwise(levels):
                rise = self.levels[upper].elevation - self.levels[lower].elevation
                self.edge(
                    landings[upper].id,
                    landings[lower].id,
                    EdgeKind.STAIR,
                    width=stair.clear_width,
                    length=stair_travel_length(rise, stair),
                    fire_rated=True,
                    can_block=True,
                    stair=sid,
                    label=f"{stair.label}, {hk_level_label(upper)} → {hk_level_label(lower)}",
                )

    def lift(
        self,
        id: str,
        label: str,
        stops: Sequence[tuple[int, str]],
        *,
        discharge_level: int = 0,
        firefighting: bool = False,
    ) -> Lift:
        """Add a lift stopping at the given ``(level, node_id)`` pairs."""
        lf = Lift(
            id=id,
            label=label,
            stops=[LiftStop(level=lv, node=nid) for lv, nid in stops],
            discharge_level=discharge_level,
            firefighting=firefighting,
        )
        self.lifts.append(lf)
        return lf

    def build(self, *, validate: bool = True) -> Building:
        """Assemble (and by default validate) the building."""
        b = Building(
            id=self.id,
            name=self.name,
            typology=self.typology,
            description=self.description,
            levels=[self.levels[k] for k in sorted(self.levels)],
            stairs=list(self.stairs.values()),
            nodes=list(self.nodes.values()),
            edges=list(self.edges.values()),
            lifts=self.lifts,
            metadata=dict(self.metadata),
        )
        if validate:
            check_building(b)
        return b


# ============================================================================ corridors
BayKind = Literal["unit", "lift_lobby", "stair", "refuge", "open", "service", "void"]
Side = Literal["north", "south", "east", "west"]


@dataclass
class Bay:
    """Something beside a corridor column (or beyond a corridor end).

    ``stair`` bays hold one or more staircases side by side (two for a scissor
    pair); with ``protected_lobby`` each stair is reached through its own lobby.
    """

    kind: BayKind
    name: str = ""
    depth: float = 8.0
    unit_type: str | None = None
    stairs: tuple[str, ...] = ()
    protected_lobby: bool = False
    label: str | None = None
    tags: tuple[str, ...] = ()


@dataclass
class Column:
    """A slice of corridor (one corridor node) with optional bays either side."""

    width: float
    north: Bay | None = None
    south: Bay | None = None


@dataclass
class CorridorDoors:
    """Widths used by :func:`add_corridor` (m)."""

    unit_door: float
    stair_door: float
    corridor_width: float
    protected_lobby_depth: float = 2.0


@dataclass
class CorridorResult:
    """Node ids created by :func:`add_corridor`."""

    segments: list[str] = field(default_factory=list)
    bays: dict[str, str] = field(default_factory=dict)
    landings: dict[str, str] = field(default_factory=dict)
    lift_lobbies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _BayFrame:
    """Local frame of a bay: ``u`` along the corridor wall, ``v`` away from it."""

    u0: float
    u1: float
    side: Side
    half: float  # corridor half-width
    end_x: float  # corridor end coordinate (for east / west bays)
    tf: Transform

    def point(self, u: float, v: float) -> Point:
        if self.side == "north":
            p = (u, self.half + v)
        elif self.side == "south":
            p = (u, -self.half - v)
        elif self.side == "east":
            p = (self.end_x + v, u)
        else:
            p = (self.end_x - v, u)
        return self.tf.point(p)

    def rect(self, u0: float, u1: float, v0: float, v1: float) -> list[Point]:
        poly = [self.point(u0, v0), self.point(u1, v0), self.point(u1, v1), self.point(u0, v1)]
        return poly if polygon_area(poly) > 0 else poly[::-1]

    def opening(self, u: float, v: float, width: float) -> tuple[Point, Point]:
        return (self.point(u - width / 2.0, v), self.point(u + width / 2.0, v))


def add_corridor(
    bld: BuildingBuilder,
    level: int,
    *,
    prefix: str,
    name: str,
    columns: Sequence[Column],
    doors: CorridorDoors,
    start_x: float = 0.0,
    transform: Transform = IDENTITY,
    west_end: Bay | None = None,
    east_end: Bay | None = None,
    end_width: float = 5.0,
) -> CorridorResult:
    """Add a double-loaded corridor on ``level`` and return the created node ids.

    Local frame: corridor centred on y = 0 running from ``start_x`` towards +x;
    ``north`` bays sit at y > 0 and ``south`` bays at y < 0. ``west_end`` /
    ``east_end`` bays (``end_width`` wide) sit beyond the corridor's ends.
    """
    lab = hk_level_label(level)
    half = doors.corridor_width / 2.0
    res = CorridorResult()
    x = start_x
    for j, col in enumerate(columns):
        x0, x1 = x, x + col.width
        seg_id = f"{prefix}.cor.{name}{j}"
        bld.node(
            seg_id,
            NodeType.CORRIDOR,
            level,
            polygon=transform.poly(rect(x0, -half, x1, half)),
            label=f"Corridor {name} at {lab}",
        )
        if res.segments:
            bld.edge(res.segments[-1], seg_id, EdgeKind.FLAT, width=doors.corridor_width)
        res.segments.append(seg_id)
        for side, bay in (("north", col.north), ("south", col.south)):
            if bay is not None:
                frame = _BayFrame(x0, x1, side, half, 0.0, transform)  # type: ignore[arg-type]
                _add_bay(bld, level, prefix, bay, frame, seg_id, doors, res, lab)
        x = x1
    for side, bay in (("west", west_end), ("east", east_end)):
        if bay is None:
            continue
        seg_id = res.segments[0] if side == "west" else res.segments[-1]
        end_x = start_x if side == "west" else x
        frame = _BayFrame(-end_width / 2, end_width / 2, side, half, end_x, transform)  # type: ignore[arg-type]
        _add_bay(bld, level, prefix, bay, frame, seg_id, doors, res, lab)
    return res


def _add_bay(
    bld: BuildingBuilder,
    level: int,
    prefix: str,
    bay: Bay,
    fr: _BayFrame,
    seg_id: str,
    doors: CorridorDoors,
    res: CorridorResult,
    lab: str,
) -> None:
    """Create the node(s) for one bay and connect them to corridor node ``seg_id``."""
    umid = (fr.u0 + fr.u1) / 2.0
    poly = fr.rect(fr.u0, fr.u1, 0.0, bay.depth)
    if bay.kind in ("service", "void"):
        bld.feature(level, bay.kind, poly, bay.label)
        return
    if bay.kind == "unit":
        nid = f"{prefix}.unit.{bay.name}"
        noun = bay.label or "Flat"
        bld.node(
            nid,
            NodeType.UNIT,
            level,
            polygon=poly,
            label=f"{noun} {bay.name} at {lab}",
            unit_type=bay.unit_type,
            tags=bay.tags,
        )
        bld.edge(
            nid,
            seg_id,
            EdgeKind.DOOR,
            width=doors.unit_door,
            opening=fr.opening(umid, 0.0, doors.unit_door),
            self_closing=True,
            fire_rated=True,
            label=f"Door of {noun} {bay.name} at {lab}",
        )
        res.bays[bay.name] = nid
        return
    if bay.kind in ("lift_lobby", "refuge", "open"):
        ntype, noun, key = {
            "lift_lobby": (NodeType.LIFT_LOBBY, "Lift lobby", "liftlobby"),
            "refuge": (NodeType.REFUGE, "Refuge area", "refuge"),
            "open": (NodeType.OPEN_AREA, "Open area", "open"),
        }[bay.kind]
        nid = f"{prefix}.{key}.{bay.name}"
        bld.node(
            nid,
            ntype,
            level,
            polygon=poly,
            label=f"{bay.label or noun} {bay.name} at {lab}".replace("  ", " "),
            tags=bay.tags,
        )
        bld.edge(nid, seg_id, EdgeKind.FLAT, width=min(fr.u1 - fr.u0, 3.0))
        res.bays[bay.name] = nid
        if bay.kind == "lift_lobby":
            res.lift_lobbies.append(nid)
        return
    if bay.kind == "stair":
        n = len(bay.stairs)
        pl = doors.protected_lobby_depth if bay.protected_lobby else 0.0
        for k, sid in enumerate(bay.stairs):
            su0 = fr.u0 + (fr.u1 - fr.u0) * k / n
            su1 = fr.u0 + (fr.u1 - fr.u0) * (k + 1) / n
            smid = (su0 + su1) / 2.0
            stair = bld.stairs[sid]
            land_id = f"{prefix}.stair.{sid}"
            bld.node(
                land_id,
                NodeType.STAIR_LANDING,
                level,
                polygon=fr.rect(su0, su1, pl, bay.depth),
                label=f"{stair.label} landing at {lab}",
                stair=sid,
            )
            entry = seg_id
            if bay.protected_lobby:
                pl_id = f"{prefix}.pl.{sid}"
                bld.node(
                    pl_id,
                    NodeType.PROTECTED_LOBBY,
                    level,
                    polygon=fr.rect(su0, su1, 0.0, pl),
                    label=f"{stair.label} protected lobby at {lab}",
                )
                bld.edge(
                    seg_id,
                    pl_id,
                    EdgeKind.DOOR,
                    width=doors.stair_door,
                    opening=fr.opening(smid, 0.0, doors.stair_door),
                    fire_rated=True,
                    self_closing=True,
                    can_block=True,
                    label=f"{stair.label} protected-lobby door at {lab}",
                )
                entry = pl_id
            bld.edge(
                entry,
                land_id,
                EdgeKind.DOOR,
                width=doors.stair_door,
                opening=fr.opening(smid, pl, doors.stair_door),
                fire_rated=True,
                self_closing=True,
                can_block=True,
                label=f"{stair.label} door at {lab}",
            )
            res.landings[sid] = land_id
        return
    raise ValueError(f"unknown bay kind {bay.kind}")  # pragma: no cover
