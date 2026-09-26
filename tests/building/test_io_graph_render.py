from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import networkx as nx
import pytest

from tailsafe.building.graph import ArcKind, arcs, summary, to_networkx
from tailsafe.building.io import (
    building_schema,
    load_building,
    save_building,
    schema_dir,
    validate_against_schema,
)
from tailsafe.building.model import Building, EdgeKind, NodeType
from tailsafe.building.render import save_render
from tailsafe.building.templates import generate

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def slab() -> Building:
    return generate("slab", storeys=8)


def test_committed_schema_is_up_to_date() -> None:
    committed = json.loads((schema_dir() / "building.schema.json").read_text())
    assert committed == building_schema(), "run `make schema` and commit the result"


def test_generated_building_matches_json_schema(slab: Building) -> None:
    validate_against_schema(json.loads(slab.model_dump_json(exclude_none=True)))


def test_schema_rejects_bad_data(slab: Building) -> None:
    data = json.loads(slab.model_dump_json())
    data["edges"][0]["width"] = -1
    with pytest.raises(jsonschema.ValidationError):
        validate_against_schema(data)


def test_roundtrip(tmp_path: Path, slab: Building) -> None:
    path = save_building(slab, tmp_path / "b.json")
    again = load_building(path, schema_check=True)
    assert again == slab
    assert again.digest() == slab.digest()


def test_example_files_load() -> None:
    for path in sorted((REPO / "data" / "templates").glob("*.json")):
        load_building(path, schema_check=True)


def test_arcs_expand_edges(slab: Building) -> None:
    all_arcs = arcs(slab)
    assert len(all_arcs) == 2 * len(slab.edges)  # no one-way edges in templates
    stairs = [e for e in slab.edges if e.kind == EdgeKind.STAIR]
    downs = [a for a in all_arcs if a.kind == ArcKind.STAIR_DOWN]
    ups = [a for a in all_arcs if a.kind == ArcKind.STAIR_UP]
    assert len(downs) == len(ups) == len(stairs)
    for a in downs:
        lv = slab.node_by_id
        assert lv[a.source].level > lv[a.target].level


def test_networkx_paths_to_exit(slab: Building) -> None:
    g = to_networkx(slab)
    top_unit = max(slab.units, key=lambda n: n.level)
    lengths = nx.single_source_dijkstra_path_length(g, top_unit.id, weight="length")
    exits = [n.id for n in slab.exits]
    assert any(e in lengths for e in exits)
    assert g.nodes[top_unit.id]["type"] == NodeType.UNIT.value


def test_summary(slab: Building) -> None:
    s = summary(slab)
    assert s["units"] == len(slab.units)
    assert s["storeys"] == 8


def test_render_smoke(tmp_path: Path, slab: Building) -> None:
    out = save_render(slab, tmp_path / "slab.png", dpi=40)
    assert out.stat().st_size > 1000
