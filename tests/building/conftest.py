from __future__ import annotations

import pytest

from tailsafe.building.model import (
    Building,
    Edge,
    EdgeKind,
    Level,
    Node,
    NodeType,
    Stair,
)


def make_tiny() -> Building:
    """Two storeys: one flat on 1/F, a corridor, one stair down to an exit."""
    return Building(
        id="tiny",
        name="Tiny",
        typology="custom",
        levels=[
            Level(index=0, label="G/F", elevation=0.0, height=3.0, kind="ground"),
            Level(index=1, label="1/F", elevation=3.0, height=3.0),
        ],
        stairs=[Stair(id="A", label="Stair A", clear_width=1.1, riser=0.175, going=0.25)],
        nodes=[
            Node(
                id="L1.unit", type=NodeType.UNIT, level=1, x=0, y=0, area=40, unit_type="flat_small"
            ),
            Node(id="L1.cor", type=NodeType.CORRIDOR, level=1, x=5, y=0, area=10),
            Node(
                id="L1.stair",
                type=NodeType.STAIR_LANDING,
                level=1,
                x=10,
                y=0,
                area=8,
                stair="A",
                label="Stair A landing at 1/F",
            ),
            Node(id="L0.stair", type=NodeType.STAIR_LANDING, level=0, x=10, y=0, area=8, stair="A"),
            Node(id="L0.exit", type=NodeType.EXIT, level=0, x=14, y=0),
        ],
        edges=[
            Edge(
                id="d1", source="L1.unit", target="L1.cor", kind=EdgeKind.DOOR, length=5, width=0.85
            ),
            Edge(
                id="c1",
                source="L1.cor",
                target="L1.stair",
                kind=EdgeKind.DOOR,
                length=5,
                width=0.85,
            ),
            Edge(
                id="L1.stair--L0.stair",
                source="L1.stair",
                target="L0.stair",
                kind=EdgeKind.STAIR,
                length=6.0,
                width=1.1,
                stair="A",
                label="Stair A, 1/F → G/F",
            ),
            Edge(
                id="x1",
                source="L0.stair",
                target="L0.exit",
                kind=EdgeKind.DOOR,
                length=4,
                width=1.6,
            ),
        ],
    )


@pytest.fixture
def tiny_building() -> Building:
    return make_tiny()
