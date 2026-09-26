"""Graph views of a building: directed arcs and a NetworkX export.

Physical edges are expanded into directed *arcs*: ``flat`` edges and doors give
two ``flat`` arcs (unless one-way); stair edges give a ``stair_down`` arc (upper
to lower landing) and a ``stair_up`` arc.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import networkx as nx

from tailsafe.building.model import Building, EdgeKind


class ArcKind(StrEnum):
    """Direction-aware traversal type of an arc."""

    FLAT = "flat"
    STAIR_DOWN = "stair_down"
    STAIR_UP = "stair_up"


@dataclass(frozen=True)
class Arc:
    """A directed traversal of a physical edge."""

    edge_id: str
    source: str
    target: str
    kind: ArcKind
    forward: bool  # True if source -> target matches the edge's stored direction


def arcs(b: Building) -> list[Arc]:
    """All directed arcs of ``b`` (forward arc first for each edge)."""
    out: list[Arc] = []
    for e in b.edges:
        if e.kind == EdgeKind.STAIR:
            out.append(Arc(e.id, e.source, e.target, ArcKind.STAIR_DOWN, True))
            if not e.directed:
                out.append(Arc(e.id, e.target, e.source, ArcKind.STAIR_UP, False))
        else:
            out.append(Arc(e.id, e.source, e.target, ArcKind.FLAT, True))
            if not e.directed:
                out.append(Arc(e.id, e.target, e.source, ArcKind.FLAT, False))
    return out


def to_networkx(b: Building) -> nx.MultiDiGraph:
    """Directed multigraph with node and arc attributes (for analysis and plots)."""
    g = nx.MultiDiGraph(name=b.name, building_id=b.id)
    for n in b.nodes:
        g.add_node(n.id, type=n.type.value, level=n.level, x=n.x, y=n.y, label=n.label)
    edges = b.edge_by_id
    for a in arcs(b):
        e = edges[a.edge_id]
        g.add_edge(
            a.source,
            a.target,
            key=f"{a.edge_id}:{'f' if a.forward else 'r'}",
            edge_id=a.edge_id,
            kind=a.kind.value,
            length=e.length,
            width=e.width,
            fire_rated=e.fire_rated,
            can_block=e.can_block,
        )
    return g


def summary(b: Building) -> dict[str, Any]:
    """Headline counts for display (CLI ``building info``, API)."""
    types = Counter(n.type.value for n in b.nodes)
    kinds = Counter(e.kind.value for e in b.edges)
    unit_types = Counter(n.unit_type for n in b.units)
    return {
        "id": b.id,
        "name": b.name,
        "typology": b.typology,
        "storeys": b.n_storeys,
        "levels": {lv.label: lv.kind.value for lv in b.levels if lv.kind.value != "typical"},
        "nodes": dict(types),
        "edges": dict(kinds),
        "units": len(b.units),
        "unit_types": dict(unit_types),
        "stairs": [s.label for s in b.stairs],
        "lifts": [lf.label + (" (firefighting)" if lf.firefighting else "") for lf in b.lifts],
        "exits": [n.label or n.id for n in b.exits],
        "height_m": round(b.levels[-1].elevation + b.levels[-1].height, 2),
        "digest": b.digest()[:12],
    }
