from __future__ import annotations

import pytest
from pydantic import ValidationError

from tailsafe.building.model import Building, Edge, EdgeKind, Node, NodeType, polygon_area


def test_polygon_area_and_orientation() -> None:
    sq = [(0.0, 0.0), (2.0, 0.0), (2.0, 3.0), (0.0, 3.0)]
    assert polygon_area(sq) == pytest.approx(6.0)
    assert polygon_area(sq[::-1]) == pytest.approx(-6.0)


def test_node_area_defaults_to_polygon_area() -> None:
    n = Node(
        id="n1",
        type=NodeType.UNIT,
        level=1,
        x=1,
        y=1,
        polygon=[(0, 0), (4, 0), (4, 5), (0, 5)],
    )
    assert n.area == pytest.approx(20.0)


@pytest.mark.parametrize("bad_id", ["has space", "slash/no", ""])
def test_ids_are_restricted(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        Node(id=bad_id, type=NodeType.UNIT, level=0, x=0, y=0)


def test_edge_rejects_non_positive_width_and_length() -> None:
    with pytest.raises(ValidationError):
        Edge(id="e", source="a", target="b", kind=EdgeKind.FLAT, length=1.0, width=0.0)
    with pytest.raises(ValidationError):
        Edge(id="e", source="a", target="b", kind=EdgeKind.FLAT, length=-1.0, width=1.0)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Node.model_validate({"id": "a", "type": "unit", "level": 0, "x": 0, "y": 0, "colour": 1})


def test_lookups_and_labels(tiny_building: Building) -> None:
    b = tiny_building
    assert b.node_by_id["L1.unit"].type == NodeType.UNIT
    assert b.describe_node("L1.stair") == "Stair A landing at 1/F"
    assert "Stair A" in b.describe_edge("L1.stair--L0.stair")
    assert b.level_label(1) == "1/F"
    assert len(b.units) == 1 and len(b.exits) == 1
    assert b.digest() == Building.model_validate(b.model_dump()).digest()
