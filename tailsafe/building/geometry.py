"""Small geometry helpers for axis-aligned rectangles (plans and replays)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Point = tuple[float, float]
Box = tuple[float, float, float, float]  # x0, x1, y0, y1


def bbox(poly: Sequence[Sequence[float]]) -> Box:
    """Bounding box ``(x0, x1, y0, y1)`` of a polygon."""
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), max(xs), min(ys), max(ys)


def shared_boundary(a: Box, b: Box, tol: float = 1e-3) -> tuple[Point, Point] | None:
    """Longest common piece of the boundaries of two axis-aligned rectangles."""
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    cands: list[tuple[Point, Point]] = []
    for xa in (ax0, ax1):
        for xb in (bx0, bx1):
            if abs(xa - xb) < tol:
                lo, hi = max(ay0, by0), min(ay1, by1)
                if hi - lo > tol:
                    cands.append(((xa, lo), (xa, hi)))
    for ya in (ay0, ay1):
        for yb in (by0, by1):
            if abs(ya - yb) < tol:
                lo, hi = max(ax0, bx0), min(ax1, bx1)
                if hi - lo > tol:
                    cands.append(((lo, ya), (hi, ya)))
    if not cands:
        return None
    return max(cands, key=lambda c: abs(c[1][0] - c[0][0]) + abs(c[1][1] - c[0][1]))


def clip_centred(seg: tuple[Point, Point], width: float) -> tuple[Point, Point]:
    """The middle ``width`` of a segment (the whole segment if it is shorter)."""
    (x0, y0), (x1, y1) = seg
    length = float(np.hypot(x1 - x0, y1 - y0))
    if width >= length or length <= 0:
        return seg
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    h = width / 2
    return (mx - ux * h, my - uy * h), (mx + ux * h, my + uy * h)


def rectangularity(poly: Sequence[Sequence[float]]) -> float:
    """Polygon area divided by its bounding-box area (1 for a rectangle)."""
    xs = np.array([float(p[0]) for p in poly])
    ys = np.array([float(p[1]) for p in poly])
    area = 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))
    x0, x1, y0, y1 = bbox(poly)
    box = (x1 - x0) * (y1 - y0)
    return area / box if box > 0 else 0.0
