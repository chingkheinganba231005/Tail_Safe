"""Building model: a multi-floor egress graph with 2D geometry.

Design (see ``docs/architecture.md``):

* **Nodes** are places people occupy or pass through: flats/rooms, corridor
  segments, lobbies, stair landings, refuge areas and final exits.
* **Edges** are physical connections stored *once*: ``flat`` (walking within a
  space), ``door`` (through a doorway) and ``stair`` (flights between landings on
  different levels; ``source`` is the upper landing). Simulators expand edges into
  directed arcs (``flat`` / ``stair_down`` / ``stair_up``).
* **Lifts** are separate entities listing the lobby they serve on each level.

Everything is in SI units; levels are integers with ground floor = 0.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import StrEnum
from functools import cached_property
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Literal[1] = 1

ID_PATTERN = r"^[A-Za-z0-9_.\-]+$"

Point = tuple[float, float]


class NodeType(StrEnum):
    """Kinds of places in the egress graph."""

    UNIT = "unit"
    CORRIDOR = "corridor"
    LOBBY = "lobby"
    PROTECTED_LOBBY = "protected_lobby"
    LIFT_LOBBY = "lift_lobby"
    STAIR_LANDING = "stair_landing"
    REFUGE = "refuge"
    OPEN_AREA = "open_area"
    EXIT = "exit"


class EdgeKind(StrEnum):
    """Kinds of physical connections."""

    FLAT = "flat"
    DOOR = "door"
    STAIR = "stair"


class LevelKind(StrEnum):
    """Role of a storey in the stack."""

    GROUND = "ground"
    TYPICAL = "typical"
    REFUGE = "refuge"
    PODIUM = "podium"


class StairKind(StrEnum):
    """Stair geometry, which sets the travel distance per storey."""

    DOGLEG = "dogleg"
    SCISSOR = "scissor"
    STRAIGHT = "straight"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=False)


class Feature(_Model):
    """A non-walkable plan feature drawn for context (lift shaft, service room, void)."""

    kind: str
    polygon: list[Point]
    label: str | None = None


class Level(_Model):
    """One storey."""

    index: int = Field(description="Integer level; ground floor = 0.")
    label: str = Field(description="Display label, Hong Kong style (G/F, 1/F, ...).")
    elevation: float = Field(description="Finished floor level above ground (m).")
    height: float = Field(gt=0, description="Floor-to-floor height (m).")
    kind: LevelKind = LevelKind.TYPICAL
    features: list[Feature] = Field(default_factory=list)


class Stair(_Model):
    """A staircase (the landings and flights reference it by ``id``)."""

    id: str = Field(pattern=ID_PATTERN)
    label: str
    kind: StairKind = StairKind.DOGLEG
    clear_width: float = Field(gt=0, description="Clear width of the flights (m).")
    riser: float = Field(gt=0, description="Riser height (m).")
    going: float = Field(gt=0, description="Tread going (m).")
    shaft: str | None = Field(
        default=None, description="Shared shaft id (scissor pairs share one shaft)."
    )


class Node(_Model):
    """A place in the egress graph."""

    id: str = Field(pattern=ID_PATTERN)
    type: NodeType
    level: int
    x: float = Field(description="Plan position (m), reference point of the space.")
    y: float
    label: str | None = None
    area: float | None = Field(default=None, gt=0, description="Floor area (m²).")
    polygon: list[Point] | None = Field(
        default=None, description="Footprint on its level, counter-clockwise (m)."
    )
    unit_type: str | None = Field(
        default=None, description="For units: key into population.household_size in params."
    )
    stair: str | None = Field(default=None, description="Stair id, for stair landings.")
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fill_area(self) -> Node:
        if self.area is None and self.polygon is not None and len(self.polygon) >= 3:
            self.area = abs(polygon_area(self.polygon)) or None
        return self


class Edge(_Model):
    """A physical connection between two nodes."""

    id: str = Field(pattern=ID_PATTERN)
    source: str
    target: str
    kind: EdgeKind
    length: float = Field(gt=0, description="Travel distance (m); along the slope for stairs.")
    width: float = Field(gt=0, description="Clear width (m): door leaf, stair or corridor.")
    capacity: float | None = Field(
        default=None, gt=0, description="Flow capacity override (persons/s)."
    )
    directed: bool = Field(default=False, description="One-way from source to target only.")
    fire_rated: bool = False
    self_closing: bool = Field(default=False, description="Doors: closes itself after use.")
    can_block: bool = Field(default=False, description="May be blocked in scenarios.")
    stair: str | None = Field(default=None, description="Stair id, for stair flights.")
    opening: tuple[Point, Point] | None = Field(
        default=None, description="Doors: the opening as a plan segment (m)."
    )
    label: str | None = None


class LiftStop(_Model):
    """A lift landing."""

    level: int
    node: str


class Lift(_Model):
    """A lift car serving a set of lobbies."""

    id: str = Field(pattern=ID_PATTERN)
    label: str
    stops: list[LiftStop]
    discharge_level: int = 0
    firefighting: bool = Field(default=False, description="Designated firefighting lift.")
    car_capacity: int | None = Field(default=None, gt=0, description="Override (persons).")
    rated_speed: float | None = Field(default=None, gt=0, description="Override (m/s).")


class Building(_Model):
    """A complete building: levels, egress graph, stairs and lifts."""

    schema_version: Literal[1] = SCHEMA_VERSION
    id: str = Field(pattern=ID_PATTERN)
    name: str
    typology: str = Field(description="cruciform, slab, twin_core, care_home or custom.")
    description: str | None = None
    levels: list[Level]
    stairs: list[Stair] = Field(default_factory=list)
    nodes: list[Node]
    edges: list[Edge]
    lifts: list[Lift] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Generator settings and provenance."
    )

    # ------------------------------------------------------------------ lookups
    # Lookups are cached on first use: treat a Building as immutable once built.
    _CACHED: ClassVar[tuple[str, ...]] = (
        "node_by_id",
        "edge_by_id",
        "level_by_index",
        "stair_by_id",
    )

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Building:
        """Copy, dropping cached lookups so they are rebuilt for the new content."""
        copied = super().model_copy(update=update, deep=deep)
        for name in self._CACHED:
            copied.__dict__.pop(name, None)
        return copied

    @cached_property
    def node_by_id(self) -> dict[str, Node]:
        """Nodes keyed by id."""
        return {n.id: n for n in self.nodes}

    @cached_property
    def edge_by_id(self) -> dict[str, Edge]:
        """Edges keyed by id."""
        return {e.id: e for e in self.edges}

    @cached_property
    def level_by_index(self) -> dict[int, Level]:
        """Levels keyed by index."""
        return {lv.index: lv for lv in self.levels}

    @cached_property
    def stair_by_id(self) -> dict[str, Stair]:
        """Stairs keyed by id."""
        return {s.id: s for s in self.stairs}

    def nodes_of_type(self, *types: NodeType) -> list[Node]:
        """All nodes whose type is one of ``types``."""
        wanted = set(types)
        return [n for n in self.nodes if n.type in wanted]

    @property
    def units(self) -> list[Node]:
        """All occupiable units (flats / rooms)."""
        return self.nodes_of_type(NodeType.UNIT)

    @property
    def exits(self) -> list[Node]:
        """All final exits."""
        return self.nodes_of_type(NodeType.EXIT)

    @property
    def n_storeys(self) -> int:
        """Number of storeys at or above ground."""
        return sum(1 for lv in self.levels if lv.index >= 0)

    def level_label(self, level: int) -> str:
        """Display label of a level index."""
        lv = self.level_by_index.get(level)
        return lv.label if lv is not None else f"L{level}"

    def describe_node(self, node_id: str) -> str:
        """Plain-English label, e.g. 'Stair A landing at 12/F'."""
        n = self.node_by_id[node_id]
        return n.label or f"{n.type.value.replace('_', ' ')} {n.id}"

    def describe_edge(self, edge_id: str) -> str:
        """Plain-English label of an edge."""
        e = self.edge_by_id[edge_id]
        if e.label:
            return e.label
        return f"{self.describe_node(e.source)} ↔ {self.describe_node(e.target)}"

    def digest(self) -> str:
        """Stable SHA-256 of the building content (used in result cache keys)."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def polygon_area(poly: list[Point]) -> float:
    """Signed shoelace area of a polygon (positive if counter-clockwise)."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s
