from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from tailsafe.building.model import NodeType
from tailsafe.building.templates import generate
from tailsafe.population.synth import PopulationConfig, sample_population
from tailsafe.sim.meso import run_meso
from tailsafe.sim.micro import micro_problems
from tailsafe.vision.detect import (
    DetectedDoor,
    Scale,
    _largest_rect,
    _rectangles,
    detect_plan,
    otsu_threshold,
    stroke_thickness,
)
from tailsafe.vision.evaluate import evaluation_markdown, match_doors, match_stairs
from tailsafe.vision.graph import assign_doors, plan_to_building
from tailsafe.vision.raster import load_image, to_data_url
from tailsafe.vision.synth import render_plan, to_png_bytes


def scale_for(m_per_px: float) -> Scale:
    return Scale(x1=0, y1=0, x2=10 / m_per_px, y2=0, metres=10.0)


@pytest.fixture(scope="module")
def cruciform():  # type: ignore[no-untyped-def]
    return generate("cruciform", storeys=4, flats_per_wing=3)


def test_otsu_and_stroke_thickness() -> None:
    img = np.ones((80, 80), dtype=np.float32)
    img[20:24, 5:75] = 0.0  # a 4 px wall
    img[5:75, 40:44] = 0.0
    img[50, 5:35] = 0.0  # thin lines
    img[5:35, 60] = 0.0
    t = otsu_threshold(img)
    assert 0.0 < t < 1.0
    assert stroke_thickness(img < t) == pytest.approx(4.0)


def test_largest_rectangles_cover_an_l_shape() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[0:20, 0:6] = True
    mask[14:20, 0:20] = True
    area, r0, c0, r1, c1 = _largest_rect(mask)
    assert area == 120 and (r1 - r0) * (c1 - c0) == 120
    rects = _rectangles(mask, 4)
    covered = np.zeros_like(mask)
    for a, b, c, d in rects:
        covered[a:c, b:d] = True
    assert (covered <= mask).all() and covered.sum() >= 0.97 * mask.sum()
    assert len(rects) == 2


def test_reads_a_typical_floor(cruciform) -> None:  # type: ignore[no-untyped-def]
    img, truth = render_plan(cruciform, 2, seed=2)
    det = detect_plan(img, scale_for(truth.m_per_px))
    doors = match_doors(det, truth)
    assert doors["door_recall"] == 1.0 and doors["precision"] >= 0.9
    stairs = match_stairs(det, truth)
    assert stairs["recall"] == 1.0 and stairs["precision"] == 1.0
    types = {r.type for r in det.rooms}
    assert {"unit", "corridor", "lobby", "stair"} <= types
    units = [r for r in det.rooms if r.type == "unit"]
    assert all(r.unit_type in {"flat_small", "flat_medium", "flat_large"} for r in units)


def test_scale_from_walls_and_entrances_at_corridor_ends(cruciform) -> None:  # type: ignore[no-untyped-def]
    img, truth = render_plan(cruciform, 0, seed=0)
    det = detect_plan(img)  # no reference line
    assert det.scale_source == "walls"
    assert det.m_per_px == pytest.approx(truth.m_per_px, rel=0.05)
    assert any("Scale estimated" in w for w in det.warnings)
    # the main entrances fill the ends of corridor stubs and still count as exits
    to_outside = [d for d in det.doors if "outside" in d.rooms]
    assert len(to_outside) == len(truth.exits)


def test_noisy_plan_still_finds_stairs() -> None:
    b = generate("slab", storeys=4, flats_per_side=6)
    img, truth = render_plan(b, 2, noise=0.06, blur=0.8, seed=3)
    det = detect_plan(img, scale_for(truth.m_per_px))
    assert match_stairs(det, truth)["recall"] == 1.0
    assert match_doors(det, truth)["door_recall"] >= 0.9


def test_plan_to_building_simulates(cruciform) -> None:  # type: ignore[no-untyped-def]
    img, truth = render_plan(cruciform, 2, seed=2)
    det = detect_plan(img, scale_for(truth.m_per_px))
    b = plan_to_building(det, storeys=8)
    assert len(b.levels) == 8 and len(b.stairs) == 2
    landings = [n for n in b.nodes if n.type == NodeType.STAIR_LANDING]
    assert len(landings) == 16
    exits = [n for n in b.nodes if n.type == NodeType.EXIT]
    assert len(exits) == 2 and b.metadata["exits_assumed"] is True  # typical floor: none drawn
    assert micro_problems(b) == []
    pop = sample_population(b, PopulationConfig(), seed=1)
    res = run_meso(b, pop)
    assert res.n_not_evacuated == 0 and pop.n_agents > 0

    ground, gt = render_plan(cruciform, 0, seed=0)
    det0 = detect_plan(ground, scale_for(gt.m_per_px))
    b0 = plan_to_building(det0, storeys=1)
    assert b0.metadata["exits_assumed"] is False
    assert sum(n.type == NodeType.EXIT for n in b0.nodes) == len(gt.exits)


def test_editor_doorways_get_their_rooms(cruciform) -> None:  # type: ignore[no-untyped-def]
    img, truth = render_plan(cruciform, 2, seed=2)
    det = detect_plan(img, scale_for(truth.m_per_px))
    d = next(x for x in det.doors if "outside" not in x.rooms)
    drawn = DetectedDoor(id="new", a=d.a, b=d.b, width_m=d.width_m, rooms=[])
    edited = det.model_copy(update={"doors": [*det.doors, drawn]})
    fixed = {x.id: x for x in assign_doors(edited)}
    assert sorted(fixed["new"].rooms) == sorted(d.rooms)
    with pytest.raises(ValueError, match="staircase"):
        no_stairs = det.model_copy(
            update={"rooms": [r.model_copy(update={"type": "unit"}) for r in det.rooms]}
        )
        plan_to_building(no_stairs, storeys=3)


def test_load_image_formats() -> None:
    img = np.ones((20, 30), dtype=np.float32)
    img[5:8, :] = 0.0
    png = to_png_bytes(img)
    a = load_image(png)
    b = load_image(to_data_url(png))
    assert a.shape == (20, 30) and np.allclose(a, b) and a[6, 3] == 0.0
    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))  # transparent -> white paper
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    assert load_image(buf.getvalue()).min() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="readable"):
        load_image(b"not an image")


def test_evaluation_table() -> None:
    res = {
        "summary": {
            "clean": {
                "plans": 2,
                "door_precision": 1.0,
                "door_recall": 0.95,
                "stair_precision": 1.0,
                "stair_recall": 1.0,
                "max_scale_error": 0.0,
            }
        }
    }
    assert "| clean | 2 | 1.00 | 0.95 |" in evaluation_markdown(res)


@pytest.mark.slow
def test_full_synthetic_evaluation() -> None:
    from tailsafe.vision.evaluate import evaluate

    res = evaluate(reference_scale=True)
    clean = res["summary"]["clean, 20 px/m"]
    assert clean["door_recall"] >= 0.95 and clean["door_precision"] >= 0.9
    assert clean["stair_recall"] == 1.0
