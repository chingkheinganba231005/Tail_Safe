"""Semantic validation of a :class:`~tailsafe.building.model.Building`.

The JSON schema (and Pydantic) check *shape*; this module checks *meaning*:
references resolve, stairs go between levels, and every unit can reach an exit.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Literal

from tailsafe.building.model import Building, EdgeKind, NodeType


@dataclass(frozen=True)
class Issue:
    """A validation finding."""

    severity: Literal["error", "warning"]
    message: str

    def __str__(self) -> str:
        return f"{self.severity}: {self.message}"


class BuildingValidationError(ValueError):
    """Raised by :func:`check_building` when a building has errors."""

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        lines = "\n  - ".join(str(i) for i in issues)
        super().__init__(f"Building failed validation:\n  - {lines}")


def validate_building(b: Building) -> list[Issue]:
    """Return every error and warning found in ``b`` (empty list = valid)."""
    issues: list[Issue] = []

    def err(msg: str) -> None:
        issues.append(Issue("error", msg))

    def warn(msg: str) -> None:
        issues.append(Issue("warning", msg))

    for what, ids in (
        ("level index", [lv.index for lv in b.levels]),
        ("node id", [n.id for n in b.nodes]),
        ("edge id", [e.id for e in b.edges]),
        ("stair id", [s.id for s in b.stairs]),
        ("lift id", [lf.id for lf in b.lifts]),
    ):
        for value, count in Counter(ids).items():
            if count > 1:
                err(f"duplicate {what} {value!r} ({count}×)")

    levels = {lv.index for lv in b.levels}
    nodes = b.node_by_id
    stairs = {s.id for s in b.stairs}

    for n in b.nodes:
        if n.level not in levels:
            err(f"node {n.id} is on unknown level {n.level}")
        if n.type == NodeType.STAIR_LANDING and n.stair not in stairs:
            err(f"stair landing {n.id} references unknown stair {n.stair!r}")
        if n.type == NodeType.UNIT and not n.unit_type:
            warn(f"unit {n.id} has no unit_type; population priors will use a default")
        if n.type != NodeType.EXIT and n.area is None:
            warn(f"node {n.id} has no area; holding capacity will use a default")

    for e in b.edges:
        src, dst = nodes.get(e.source), nodes.get(e.target)
        if src is None or dst is None:
            err(f"edge {e.id} references a missing node ({e.source} -> {e.target})")
            continue
        if e.source == e.target:
            err(f"edge {e.id} is a self-loop")
        if e.kind == EdgeKind.STAIR:
            if src.level <= dst.level:
                err(f"stair edge {e.id} must go from the upper landing (source) to the lower")
            if e.stair not in stairs:
                err(f"stair edge {e.id} references unknown stair {e.stair!r}")
        elif src.level != dst.level:
            err(f"{e.kind.value} edge {e.id} joins different levels ({src.level}, {dst.level})")
        if e.kind == EdgeKind.DOOR and e.width > 3.0:
            warn(f"door {e.id} is unusually wide ({e.width:.2f} m)")

    for lift in b.lifts:
        if lift.discharge_level not in levels:
            err(f"lift {lift.id} discharges at unknown level {lift.discharge_level}")
        stop_levels = [s.level for s in lift.stops]
        if lift.discharge_level not in stop_levels:
            err(f"lift {lift.id} does not stop at its discharge level")
        for stop in lift.stops:
            node = nodes.get(stop.node)
            if node is None:
                err(f"lift {lift.id} stop references missing node {stop.node}")
            elif node.level != stop.level:
                err(f"lift {lift.id} stop {stop.node} is on level {node.level}, not {stop.level}")

    if not b.exits:
        err("building has no final exit")
    if not b.units:
        warn("building has no units (no occupants will be generated)")

    if not any(i.severity == "error" for i in issues):
        stranded = units_without_exit(b)
        if stranded:
            sample = ", ".join(stranded[:5])
            err(f"{len(stranded)} unit(s) cannot reach any exit (e.g. {sample})")
    return issues


def units_without_exit(b: Building) -> list[str]:
    """Units with no walking route to a final exit (respecting one-way edges)."""
    # Reverse BFS from exits over walkable arcs.
    rev: dict[str, list[str]] = {n.id: [] for n in b.nodes}
    for e in b.edges:
        rev[e.target].append(e.source)  # arc source -> target
        if not e.directed:
            rev[e.source].append(e.target)  # arc target -> source
    seen = {n.id for n in b.exits}
    queue = deque(seen)
    while queue:
        v = queue.popleft()
        for u in rev[v]:
            if u not in seen:
                seen.add(u)
                queue.append(u)
    return [n.id for n in b.units if n.id not in seen]


def check_building(b: Building, *, allow_warnings: bool = True) -> list[Issue]:
    """Validate and raise :class:`BuildingValidationError` on errors.

    Returns the warnings (if ``allow_warnings``) so callers can surface them.
    """
    issues = validate_building(b)
    errors = [i for i in issues if i.severity == "error" or not allow_warnings]
    if errors:
        raise BuildingValidationError(errors)
    return issues
