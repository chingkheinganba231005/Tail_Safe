from __future__ import annotations

import pytest

from tailsafe.building.model import Building, Edge, EdgeKind
from tailsafe.building.validate import (
    BuildingValidationError,
    check_building,
    units_without_exit,
    validate_building,
)


def errors(b: Building) -> list[str]:
    return [i.message for i in validate_building(b) if i.severity == "error"]


def test_tiny_building_is_valid(tiny_building: Building) -> None:
    assert errors(tiny_building) == []
    assert check_building(tiny_building) == []


def test_duplicate_node_ids(tiny_building: Building) -> None:
    b = tiny_building.model_copy(update={"nodes": [*tiny_building.nodes, tiny_building.nodes[0]]})
    assert any("duplicate node id" in e for e in errors(b))


def test_missing_reference(tiny_building: Building) -> None:
    bad = Edge(id="bad", source="L1.cor", target="nowhere", kind=EdgeKind.FLAT, length=1, width=1)
    b = tiny_building.model_copy(update={"edges": [*tiny_building.edges, bad]})
    assert any("missing node" in e for e in errors(b))


def test_stair_must_point_down(tiny_building: Building) -> None:
    flipped = [
        e.model_copy(update={"source": e.target, "target": e.source}) if e.kind == "stair" else e
        for e in tiny_building.edges
    ]
    b = tiny_building.model_copy(update={"edges": flipped})
    assert any("upper landing" in e for e in errors(b))


def test_flat_edge_between_levels_is_an_error(tiny_building: Building) -> None:
    bad = Edge(id="bad", source="L1.cor", target="L0.exit", kind=EdgeKind.FLAT, length=1, width=1)
    b = tiny_building.model_copy(update={"edges": [*tiny_building.edges, bad]})
    assert any("joins different levels" in e for e in errors(b))


def test_unreachable_unit_detected(tiny_building: Building) -> None:
    b = tiny_building.model_copy(
        update={"edges": [e for e in tiny_building.edges if e.kind != "stair"]}
    )
    assert units_without_exit(b) == ["L1.unit"]
    with pytest.raises(BuildingValidationError, match="cannot reach any exit"):
        check_building(b)


def test_one_way_edges_are_respected(tiny_building: Building) -> None:
    # Make the stair one-way *upwards* only by flipping direction semantics via a
    # directed door that points into the flat: the flat can no longer leave.
    edges = [
        e.model_copy(update={"source": "L1.cor", "target": "L1.unit", "directed": True})
        if e.id == "d1"
        else e
        for e in tiny_building.edges
    ]
    b = tiny_building.model_copy(update={"edges": edges})
    assert units_without_exit(b) == ["L1.unit"]


def test_no_exit(tiny_building: Building) -> None:
    b = tiny_building.model_copy(
        update={
            "nodes": [n for n in tiny_building.nodes if n.type != "exit"],
            "edges": [e for e in tiny_building.edges if e.id != "x1"],
        }
    )
    assert "building has no final exit" in errors(b)
