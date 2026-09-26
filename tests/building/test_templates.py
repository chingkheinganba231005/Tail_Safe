from __future__ import annotations

import itertools
import math
from collections import Counter

import pytest

from tailsafe.building.builder import (
    Transform,
    centroid,
    hk_level_label,
    rect,
    refuge_levels,
    stair_travel_length,
)
from tailsafe.building.model import Building, EdgeKind, LevelKind, NodeType, Stair, StairKind
from tailsafe.building.templates import TEMPLATES, generate, generate_cruciform
from tailsafe.building.validate import units_without_exit, validate_building


@pytest.fixture(scope="module")
def cruciform40() -> Building:
    return generate_cruciform(storeys=40)


@pytest.fixture(scope="module", params=sorted(TEMPLATES))
def any_template(request: pytest.FixtureRequest) -> Building:
    return generate(request.param)


# ------------------------------------------------------------------ helpers
def test_hk_labels() -> None:
    assert [hk_level_label(i) for i in (-1, 0, 1, 14)] == ["B1/F", "G/F", "1/F", "14/F"]


def test_refuge_levels() -> None:
    assert refuge_levels(40, 20, 25) == [20]
    assert refuge_levels(45, 20, 25) == [20, 40]
    assert refuge_levels(41, 20, 25) == [20]  # never on the top storey
    assert refuge_levels(25, 20, 25) == []


def test_stair_travel_length_matches_hand_calculation() -> None:
    dog = Stair(id="A", label="A", kind=StairKind.DOGLEG, clear_width=1.1, riser=0.175, going=0.25)
    # 2.8 m rise: two flights of 8 risers (1.4 m rise, 2.0 m going) + semicircular turn.
    expected = 2 * math.hypot(8 * 0.25, 1.4) + math.pi * 1.1 / 2
    assert stair_travel_length(2.8, dog) == pytest.approx(expected)
    sc = dog.model_copy(update={"kind": StairKind.SCISSOR})
    assert stair_travel_length(2.8, sc) == pytest.approx(math.hypot(16 * 0.25, 2.8))


def test_transform_rotates_counter_clockwise() -> None:
    p = Transform(90).point((1.0, 0.0))
    assert p[0] == pytest.approx(0.0, abs=1e-12) and p[1] == pytest.approx(1.0)
    assert centroid(rect(0, 0, 2, 4)) == pytest.approx((1.0, 2.0))


# ------------------------------------------------------------------ all templates
def test_template_is_valid(any_template: Building) -> None:
    issues = validate_building(any_template)
    assert [i for i in issues if i.severity == "error"] == []
    assert units_without_exit(any_template) == []
    assert any_template.units and any_template.exits


def test_template_is_deterministic(any_template: Building) -> None:
    again = generate(any_template.typology)
    assert again.model_dump_json() == any_template.model_dump_json()


def test_every_stair_is_continuous(any_template: Building) -> None:
    b = any_template
    for stair in b.stairs:
        landings = sorted(
            n.level for n in b.nodes_of_type(NodeType.STAIR_LANDING) if n.stair == stair.id
        )
        assert landings == list(range(landings[0], landings[-1] + 1))
        flights = [e for e in b.edges if e.kind == EdgeKind.STAIR and e.stair == stair.id]
        assert len(flights) == len(landings) - 1
        assert landings[0] == 0, f"{stair.label} must reach the ground floor"


def test_lifts_stop_at_every_level(any_template: Building) -> None:
    b = any_template
    for lift in b.lifts:
        assert sorted(s.level for s in lift.stops) == [lv.index for lv in b.levels]
        for stop in lift.stops:
            assert b.node_by_id[stop.node].type == NodeType.LIFT_LOBBY
    assert sum(lf.firefighting for lf in b.lifts) == (1 if b.lifts else 0)


def _bbox(poly: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def test_spaces_do_not_overlap(any_template: Building) -> None:
    """All generated spaces are axis-aligned rectangles; none may overlap."""
    b = any_template
    for lv in b.levels:
        boxes = [(n.id, _bbox(n.polygon)) for n in b.nodes if n.level == lv.index and n.polygon] + [
            (f"feature:{f.kind}", _bbox(f.polygon)) for f in lv.features
        ]
        for (ida, a), (idb, c) in itertools.combinations(boxes, 2):
            ox = min(a[2], c[2]) - max(a[0], c[0])
            oy = min(a[3], c[3]) - max(a[1], c[1])
            assert ox <= 1e-6 or oy <= 1e-6, f"{ida} overlaps {idb} on {lv.label}"


def test_door_openings_lie_on_their_rooms(any_template: Building) -> None:
    b = any_template
    for e in b.edges:
        if e.kind != EdgeKind.DOOR or e.opening is None:
            continue
        mid = (
            (e.opening[0][0] + e.opening[1][0]) / 2,
            (e.opening[0][1] + e.opening[1][1]) / 2,
        )
        width = math.dist(e.opening[0], e.opening[1])
        assert width == pytest.approx(e.width, abs=1e-3)
        src = b.node_by_id[e.source]
        assert src.polygon is not None
        x0, y0, x1, y1 = _bbox(src.polygon)
        assert x0 - 1e-6 <= mid[0] <= x1 + 1e-6 and y0 - 1e-6 <= mid[1] <= y1 + 1e-6


# ------------------------------------------------------------------ cruciform specifics
def test_cruciform_40_counts(cruciform40: Building) -> None:
    b = cruciform40
    assert b.n_storeys == 40
    assert [lv.label for lv in b.levels if lv.kind == LevelKind.REFUGE] == ["20/F"]
    residential = [lv for lv in b.levels if lv.kind == LevelKind.TYPICAL]
    assert len(residential) == 38
    assert len(b.units) == 38 * 20
    per_level = Counter(n.level for n in b.units)
    assert set(per_level.values()) == {20}
    assert {s.id for s in b.stairs} == {"A", "B"}
    assert b.levels[-1].label == "39/F"


def test_cruciform_wing_end_stairs() -> None:
    b = generate_cruciform(storeys=12, wing_end_stairs=True)
    assert {s.id for s in b.stairs} == {"A", "B", "C", "D", "E", "F"}
    assert len(b.exits) == 8


def test_cruciform_odd_flats_per_wing() -> None:
    b = generate_cruciform(storeys=5, flats_per_wing=3)
    assert len(b.units) == 4 * 4 * 3  # levels 1-4, 4 wings, 3 flats


def test_twin_core_has_protected_lobbies_and_scissor_pair() -> None:
    b = generate("twin_core")
    assert {s.kind for s in b.stairs} == {StairKind.SCISSOR}
    assert {s.shaft for s in b.stairs} == {"S1"}
    pl = b.nodes_of_type(NodeType.PROTECTED_LOBBY)
    assert len(pl) == 2 * b.n_storeys
    assert not [n for n in b.units if b.level_by_index[n.level].kind == LevelKind.PODIUM]


def test_care_home_rooms_are_tagged() -> None:
    b = generate("care_home")
    assert {n.unit_type for n in b.units} == {"care_room"}
    assert any("staff_station" in n.tags for n in b.nodes)


def test_bad_arguments() -> None:
    with pytest.raises(ValueError):
        generate("pagoda")
    with pytest.raises(ValueError):
        generate_cruciform(storeys=1)
    with pytest.raises(ValueError):
        generate("twin_core", flats_per_floor=7)
