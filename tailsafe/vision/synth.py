"""Synthetic raster floor plans rendered from building models, with ground truth.

Used to test and evaluate the plan reader (:mod:`tailsafe.vision.detect`)
without real drawings: walls are thick lines along room boundaries, doors
are gaps with a leaf and a swing arc, stairs carry tread lines, and rooms get
furniture outlines and text-like clutter. Real plans are messier; results on
these images are an upper bound on what the reader does on scans.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from tailsafe.building.geometry import Box, bbox, shared_boundary
from tailsafe.building.model import Building, EdgeKind, NodeType
from tailsafe.rng import stream


@dataclass
class PlanTruth:
    """Ground truth of a rendered plan, in pixels (x = column, y = row)."""

    m_per_px: float
    doors: list[tuple[float, float, float, float]] = field(default_factory=list)
    openings: list[tuple[float, float, float, float]] = field(default_factory=list)
    stairs: list[tuple[float, float, float, float]] = field(default_factory=list)  # x0 y0 x1 y1
    exits: list[tuple[float, float, float, float]] = field(default_factory=list)
    rooms: int = 0

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "m_per_px": self.m_per_px,
            "doors": self.doors,
            "openings": self.openings,
            "stairs": self.stairs,
            "exits": self.exits,
            "rooms": self.rooms,
        }


class _Canvas:
    """A white page with plan-to-pixel conversion and drawing primitives."""

    def __init__(self, box: Box, px_per_m: float, margin_m: float) -> None:
        x0, x1, y0, y1 = box
        self.ppm = px_per_m
        self.x0 = x0 - margin_m
        self.y1 = y1 + margin_m
        w = int(np.ceil((x1 - x0 + 2 * margin_m) * px_per_m))
        h = int(np.ceil((y1 - y0 + 2 * margin_m) * px_per_m))
        self.img = np.ones((h, w), dtype=np.float32)

    def px(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x0) * self.ppm, (self.y1 - y) * self.ppm

    def rect(self, c0: float, r0: float, c1: float, r1: float, value: float) -> None:
        h, w = self.img.shape
        a, b = sorted((c0, c1))
        c, d = sorted((r0, r1))
        ia, ib = max(0, int(np.floor(a))), min(w, int(np.ceil(b)))
        ic, id_ = max(0, int(np.floor(c))), min(h, int(np.ceil(d)))
        if ib > ia and id_ > ic:
            self.img[ic:id_, ia:ib] = value

    def thick(self, p: tuple[float, float], q: tuple[float, float], t_px: float) -> None:
        """Axis-aligned thick line between plan points."""
        (c0, r0), (c1, r1) = self.px(*p), self.px(*q)
        h = t_px / 2
        if abs(c1 - c0) >= abs(r1 - r0):
            self.rect(min(c0, c1) - h, r0 - h, max(c0, c1) + h, r0 + h, 0.0)
        else:
            self.rect(c0 - h, min(r0, r1) - h, c0 + h, max(r0, r1) + h, 0.0)

    def line(self, p: tuple[float, float], q: tuple[float, float], width: int = 1) -> None:
        """Thin line between pixel points."""
        (c0, r0), (c1, r1) = p, q
        n = int(np.ceil(max(abs(c1 - c0), abs(r1 - r0)) * 2)) + 1
        cs = np.linspace(c0, c1, n)
        rs = np.linspace(r0, r1, n)
        h, w = self.img.shape
        for dc in range(width):
            ci = np.clip(np.round(cs).astype(int) + dc, 0, w - 1)
            ri = np.clip(np.round(rs).astype(int), 0, h - 1)
            self.img[ri, ci] = 0.0

    def arc(self, centre: tuple[float, float], radius: float, a0: float, a1: float) -> None:
        n = max(8, int(radius * abs(a1 - a0)))
        ang = np.linspace(a0, a1, n)
        pts = [(centre[0] + radius * np.cos(a), centre[1] - radius * np.sin(a)) for a in ang]
        for p, q in itertools.pairwise(pts):
            self.line(p, q)


def render_plan(
    building: Building,
    level: int,
    *,
    px_per_m: float = 20.0,
    wall_thickness: float = 0.2,
    margin: float = 2.0,
    clutter: bool = True,
    noise: float = 0.0,
    blur: float = 0.0,
    seed: int = 0,
) -> tuple[NDArray[np.float32], PlanTruth]:
    """Render one level as a grey image (1 = paper, 0 = ink) with its ground truth."""
    rng = stream(seed, "plan-render", level)
    nodes = [n for n in building.nodes if n.level == level]
    rooms = [n for n in nodes if n.polygon and n.type != NodeType.EXIT]
    boxes = {n.id: bbox(n.polygon) for n in rooms if n.polygon}
    pts = [p for n in rooms for p in (n.polygon or [])]
    lv = next(x for x in building.levels if x.index == level)
    feats = [f for f in lv.features]
    pts += [p for f in feats for p in f.polygon]
    cv = _Canvas(bbox(pts), px_per_m, margin)
    t_px = wall_thickness * px_per_m
    truth = PlanTruth(m_per_px=1.0 / px_per_m, rooms=len(rooms))

    def outline(b: Box) -> None:
        x0, x1, y0, y1 = b
        for p, q in (
            ((x0, y0), (x1, y0)),
            ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)),
            ((x0, y1), (x0, y0)),
        ):
            cv.thick(p, q, t_px)

    for b in boxes.values():
        outline(b)
    for f in feats:
        fb = bbox(f.polygon)
        outline(fb)
        if f.kind == "lift_shaft":
            (c0, r0), (c1, r1) = cv.px(fb[0], fb[3]), cv.px(fb[1], fb[2])
            cv.line((c0, r0), (c1, r1))
            cv.line((c0, r1), (c1, r0))

    def clear(seg: tuple[tuple[float, float], tuple[float, float]], extra: float) -> None:
        (x0, y0), (x1, y1) = seg
        (c0, r0), (c1, r1) = cv.px(x0, y0), cv.px(x1, y1)
        h = t_px / 2 + extra
        if abs(c1 - c0) >= abs(r1 - r0):
            cv.rect(min(c0, c1), r0 - h, max(c0, c1), r0 + h, 1.0)
        else:
            cv.rect(c0 - h, min(r0, r1), c0 + h, max(r0, r1), 1.0)

    by = building.node_by_id
    passages: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for e in building.edges:
        s, t = by[e.source], by[e.target]
        if s.level != level and t.level != level:
            continue
        if e.kind == EdgeKind.STAIR:
            continue
        if e.opening:
            (x0, y0), (x1, y1) = e.opening
            clear(((x0, y0), (x1, y1)), 1.0)
            (c0, r0), (c1, r1) = cv.px(x0, y0), cv.px(x1, y1)
            seg = (c0, r0, c1, r1)
            if t.type == NodeType.EXIT or s.type == NodeType.EXIT:
                truth.exits.append(seg)
            truth.doors.append(seg)
            # leaf and swing into the source room
            inner = s if s.type != NodeType.EXIT else t
            ic, ir = cv.px(inner.x, inner.y)
            w = float(np.hypot(c1 - c0, r1 - r0))
            if abs(c1 - c0) >= abs(r1 - r0):
                sgn = 1.0 if ir > r0 else -1.0
                cv.line((c0, r0), (c0, r0 + sgn * w))
                start, end = (0.0, -np.pi / 2) if sgn > 0 else (0.0, np.pi / 2)
                if c1 < c0:
                    start, end = (np.pi, np.pi + (np.pi / 2 if sgn > 0 else -np.pi / 2))
                cv.arc((c0, r0), w, start, end)
            else:
                sgn = 1.0 if ic > c0 else -1.0
                cv.line((c0, r0), (c0 + sgn * w, r0))
                base = -np.pi / 2 if r1 > r0 else np.pi / 2
                cv.arc((c0, r0), w, base, 0.0 if sgn > 0 else np.pi)
        elif e.kind == EdgeKind.FLAT and s.id in boxes and t.id in boxes:
            shared = shared_boundary(boxes[s.id], boxes[t.id])
            if shared is None:
                continue
            clear(shared, 0.5)
            passages.append(shared)

    # A cleared passage is an opening when walls continue beyond both of its
    # ends (a gap in a wall line, e.g. a corridor crossing the flats' wall line).
    h_img, w_img = cv.img.shape
    for (x0, y0), (x1, y1) in passages:
        (c0, r0), (c1, r1) = cv.px(x0, y0), cv.px(x1, y1)
        length = float(np.hypot(c1 - c0, r1 - r0)) or 1.0
        dc, dr = (c1 - c0) / length, (r1 - r0) / length
        beyond = t_px / 2 + 2
        ends = ((c0 - dc * beyond, r0 - dr * beyond), (c1 + dc * beyond, r1 + dr * beyond))
        inked = [
            0 <= int(r) < h_img and 0 <= int(c) < w_img and cv.img[int(r), int(c)] < 0.5
            for c, r in ends
        ]
        if all(inked):
            truth.openings.append((c0, r0, c1, r1))

    # stair symbols: tread lines across the landing's short side
    for n in rooms:
        if n.type != NodeType.STAIR_LANDING:
            continue
        x0, x1, y0, y1 = boxes[n.id]
        (c0, r0), (c1, r1) = cv.px(x0, y1), cv.px(x1, y0)
        truth.stairs.append((c0, r0, c1, r1))
        step = 0.25 * px_per_m
        pad = t_px
        if c1 - c0 >= r1 - r0:  # long side horizontal: vertical treads
            start, stop = c0 + (c1 - c0) * 0.2, c0 + (c1 - c0) * 0.85
            k = start
            while k < stop:
                cv.line((k, r0 + pad), (k, r1 - pad))
                k += step
            cv.line((start, (r0 + r1) / 2), (stop, (r0 + r1) / 2))
        else:
            start, stop = r0 + (r1 - r0) * 0.2, r0 + (r1 - r0) * 0.85
            k = start
            while k < stop:
                cv.line((c0 + pad, k), (c1 - pad, k))
                k += step
            cv.line(((c0 + c1) / 2, start), ((c0 + c1) / 2, stop))

    if clutter:
        for n in rooms:
            x0, x1, y0, y1 = boxes[n.id]
            (c0, r0), (c1, r1) = cv.px(x0, y1), cv.px(x1, y0)
            cx, cy = (c0 + c1) / 2, (r0 + r1) / 2
            # "text": a row of short strokes at the room centre
            for k in range(int(rng.integers(3, 8))):
                gx = cx - 12 + k * 4 + rng.random()
                cv.line((gx, cy - 3), (gx + rng.random() * 2, cy + 3))
            if n.type == NodeType.UNIT and (c1 - c0) > 3 * px_per_m and (r1 - r0) > 3 * px_per_m:
                # furniture: a bed and a table outline
                for fw, fh in ((2.0, 1.5), (1.0, 0.8)):
                    fx = c0 + t_px + rng.random() * max(1.0, (c1 - c0) - fw * px_per_m - 2 * t_px)
                    fy = r0 + t_px + rng.random() * max(1.0, (r1 - r0) - fh * px_per_m - 2 * t_px)
                    gx1, gy1 = fx + fw * px_per_m, fy + fh * px_per_m
                    for p, q in (
                        ((fx, fy), (gx1, fy)),
                        ((gx1, fy), (gx1, gy1)),
                        ((gx1, gy1), (fx, gy1)),
                        ((fx, gy1), (fx, fy)),
                    ):
                        cv.line(p, q)
    img = cv.img
    if blur > 0:
        img = ndimage.gaussian_filter(img, blur).astype(np.float32)
    if noise > 0:
        img = np.clip(img + rng.normal(0, noise, img.shape), 0.0, 1.0).astype(np.float32)
    return img, truth


def to_png_bytes(img: NDArray[np.float32]) -> bytes:
    """Encode a grey image (0..1) as PNG."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()
