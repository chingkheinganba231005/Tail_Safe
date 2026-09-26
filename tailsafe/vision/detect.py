"""Read a raster floor plan: walls, doorways, rooms and stairs.

Classical image processing only (NumPy / SciPy), so it runs anywhere:

1. **Ink** by Otsu's threshold.
2. **Walls** are the thick strokes: the stroke half-width is measured with a
   distance transform and a morphological opening removes everything thinner
   (door leaves and swings, furniture, text, stair treads).
3. **Doorways** are short gaps in wall lines: a closing along each axis
   bridges gaps up to ``vision.door_width_max``; a bridged piece counts as a
   doorway if it is as thin as a wall and at least ``door_width_min`` long.
4. **Rooms** are the connected free spaces once doorways are sealed; spaces
   touching the image border are outside. Each room is split into a few
   rectangles (greedy largest rectangle) grown to the wall centre lines, so
   rooms tile the plan like the procedural templates do.
5. **Stairs** are rooms whose thin lines repeat at a tread spacing
   (autocorrelation of the line profile) and span most of the room.
6. **Types** from shape and doors: stair, corridor (long and narrow), lobby
   (three or more doorways), flat (the rest; size class by area), void (no
   doorway).

The result (:class:`PlanDetection`) is plain JSON in image pixels so the web
editor can draw it over the uploaded image and send corrections back.
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

import numpy as np
from numba import njit
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from scipy import ndimage

from tailsafe.config import Params, get_params

RoomType = Literal["unit", "corridor", "lobby", "stair", "refuge", "void"]
MAX_SIDE = 2000  # px; larger images are downsampled for detection


class Scale(BaseModel):
    """A reference line drawn on the image and its real length."""

    x1: float
    y1: float
    x2: float
    y2: float
    metres: float = Field(gt=0)

    def m_per_px(self) -> float:
        """Metres per image pixel."""
        d = float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))
        if d <= 0:
            raise ValueError("the reference line has zero length")
        return self.metres / d


class DetectedRoom(BaseModel):
    """A room: rectangles in image pixels ``[x0, y0, x1, y1]`` (y down)."""

    id: str
    type: RoomType
    rects: list[list[float]]
    area_m2: float
    doors: int = 0
    stair_score: float = 0.0
    unit_type: str | None = None


class DetectedDoor(BaseModel):
    """A doorway: its two ends in image pixels and the rooms it joins."""

    id: str
    a: list[float]
    b: list[float]
    width_m: float
    rooms: list[str] = Field(default_factory=list)  # room ids, or "outside"


class PlanDetection(BaseModel):
    """What the plan reader found (and what the editor corrects)."""

    width: int
    height: int
    m_per_px: float
    scale_source: Literal["reference", "walls"]
    wall_px: float
    rooms: list[DetectedRoom]
    doors: list[DetectedDoor]
    warnings: list[str] = Field(default_factory=list)

    def room(self, room_id: str) -> DetectedRoom:
        """Look a room up by id."""
        for r in self.rooms:
            if r.id == room_id:
                return r
        raise KeyError(room_id)


# ----------------------------------------------------------------------------- helpers
def otsu_threshold(gray: NDArray[np.floating]) -> float:
    """Otsu's threshold of a grey image in [0, 1]."""
    hist, edges = np.histogram(gray.ravel(), bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    k = int(np.nanargmax(sigma))
    return float(edges[k + 1])


def _crossings(ink: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Lengths of vertical ink runs across lines that continue left and right."""
    h, w = ink.shape
    padded = np.zeros((h + 2, w), dtype=np.int8)
    padded[1:-1] = ink
    d = np.diff(padded, axis=0)
    out: list[int] = []
    for c in range(1, w - 1):
        starts = np.flatnonzero(d[:, c] == 1)
        ends = np.flatnonzero(d[:, c] == -1)
        for r0, r1 in zip(starts, ends, strict=False):
            n = r1 - r0
            if 2 <= n <= 40 and ink[r0:r1, c - 1].all() and ink[r0:r1, c + 1].all():
                out.append(int(n))
    return np.array(out, dtype=np.int64)


@njit(cache=True)
def _run_width(walls, r, c):  # type: ignore[no-untyped-def]
    """Length of the horizontal wall run through (r, c)."""
    w = walls.shape[1]
    a = c
    while a - 1 >= 0 and walls[r, a - 1]:
        a -= 1
    b = c
    while b + 1 < w and walls[r, b + 1]:
        b += 1
    return b - a + 1


@njit(cache=True)
def _jamb_gaps(walls, lmin, lmax, wall_px):  # type: ignore[no-untyped-def]
    """Vertical gaps (lmin..lmax px) in a column between two wall pixels whose
    rows are the ends of a *vertical* wall piece (narrow in that row)."""
    h, w = walls.shape
    out = np.zeros((h, w), dtype=np.bool_)
    limit = int(1.6 * wall_px) + 1
    for c in range(w):
        last = -1
        for r in range(h):
            if not walls[r, c]:
                continue
            gap = r - last - 1
            if (
                last >= 0
                and lmin <= gap <= lmax
                and _run_width(walls, last, c) <= limit
                and _run_width(walls, r, c) <= limit
            ):
                for k in range(last + 1, r):
                    out[k, c] = True
            last = r
    return out


def _caps(walls: NDArray[np.bool_], wall_px: float) -> list[tuple[int, int, int]]:
    """Free bottom ends of vertical walls: (row of the last wall pixel, col0, col1)."""
    h, _ = walls.shape
    below = np.zeros_like(walls)
    below[:-1] = walls[1:]
    edge = walls & ~below
    lab, _ = ndimage.label(edge, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))
    out: list[tuple[int, int, int]] = []
    reach = int(np.ceil(2 * wall_px))
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        r = sl[0].start
        c0, c1 = sl[1].start, sl[1].stop
        width = c1 - c0
        if not (0.5 * wall_px <= width <= 1.6 * wall_px) or r - reach < 0:
            continue
        if walls[r - reach : r + 1, c0:c1].mean() > 0.8:  # a wall running up from here
            out.append((r, c0, c1))
    del h
    return out


def _mouths(
    walls: NDArray[np.bool_], wall_px: float, lmin: float, lmax: float
) -> list[tuple[str, int, int, int, int]]:
    """Openings between the free ends of two parallel walls (e.g. a corridor
    that ends in an entrance): gap boxes ``(axis, r0, c0, r1, c1)``."""
    out: list[tuple[str, int, int, int, int]] = []
    t = int(np.ceil(wall_px))
    views = (
        (walls, False, False),  # bottom ends of vertical walls
        (walls[::-1], True, False),  # top ends
        (walls.T, False, True),  # right ends of horizontal walls
        (walls.T[::-1], True, True),  # left ends
    )
    for img, flipped, transposed in views:
        caps = sorted(_caps(np.ascontiguousarray(img), wall_px), key=lambda x: (x[0], x[1]))
        for i, (ra, _a0, a1) in enumerate(caps):
            for rb, b0, _b1 in caps[i + 1 :]:
                if abs(rb - ra) > wall_px:
                    continue
                gap = b0 - a1
                if not (0.8 * lmin <= gap <= lmax):
                    continue
                r = max(ra, rb)
                if img[max(0, r - t + 1) : r + 1, a1:b0].mean() > 0.2:  # wall in between
                    continue
                g0, g1 = r - t + 1, r + 1
                h_img = img.shape[0]
                if flipped:
                    g0, g1 = h_img - g1, h_img - g0
                if transposed:
                    out.append(("v", a1, g0, b0, g1))
                else:
                    out.append(("h", g0, a1, g1, b0))
                break
    return out


def stroke_thickness(ink: NDArray[np.bool_]) -> float:
    """Thickness (px) of the thick strokes: the upper quartile of line crossings.

    Walls are crossed along their whole length at their full thickness; thin
    lines (text, treads, furniture) give crossings of one or two pixels.
    """
    runs = np.concatenate([_crossings(ink), _crossings(ink.T)])
    if runs.size == 0:
        return 2.0
    return float(max(2.0, np.percentile(runs, 75)))


@njit(cache=True)
def _largest_rect(mask):  # type: ignore[no-untyped-def]
    """Largest axis-aligned rectangle of True cells: (area, r0, c0, r1, c1)."""
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int64)
    best = (0, 0, 0, 0, 0)
    stack = np.empty(w + 1, dtype=np.int64)
    for r in range(h):
        for c in range(w):
            heights[c] = heights[c] + 1 if mask[r, c] else 0
        top = 0
        c = 0
        while c <= w:
            hc = heights[c] if c < w else 0
            if top == 0 or hc >= heights[stack[top - 1]]:
                stack[top] = c
                top += 1
                c += 1
            else:
                top -= 1
                hh = heights[stack[top]]
                left = stack[top - 1] + 1 if top > 0 else 0
                area = hh * (c - left)
                if area > best[0]:
                    best = (area, r - hh + 1, left, r + 1, c)
    return best


def _rectangles(
    mask: NDArray[np.bool_], min_cells: int, keep: float = 0.97
) -> list[tuple[int, int, int, int]]:
    """Greedy cover of a region by its largest rectangles: (r0, c0, r1, c1)."""
    work = mask.copy()
    total = int(mask.sum())
    out: list[tuple[int, int, int, int]] = []
    covered = 0
    while covered < keep * total and len(out) < 12:
        area, r0, c0, r1, c1 = _largest_rect(work)
        if area < min_cells:
            break
        out.append((int(r0), int(c0), int(r1), int(c1)))
        work[r0:r1, c0:c1] = False
        covered += int(area)
    return out


def _stair_score(
    lines: NDArray[np.bool_], region: NDArray[np.bool_], lo: float, hi: float
) -> float:
    """How strongly thin lines in a room repeat at tread spacing (0..1)."""
    best = 0.0
    for axis in (0, 1):
        prof = lines.sum(axis=axis).astype(np.float64)
        extent = region.sum(axis=axis).astype(np.float64)
        span = float(np.median(extent[extent > 0])) if (extent > 0).any() else 0.0
        if span <= 0 or prof.size < 3 * hi:
            continue
        long_lines = prof >= 0.4 * span  # tread lines cross most of the flight
        if long_lines.sum() < 4:
            continue
        x = long_lines.astype(np.float64) - long_lines.mean()
        acf = np.correlate(x, x, mode="full")[x.size - 1 :]
        if acf[0] <= 0:
            continue
        a, b = max(1, int(np.floor(lo))), min(x.size - 1, int(np.ceil(hi)))
        if b < a:
            continue
        best = max(best, float(acf[a : b + 1].max() / acf[0]))
    return best


# ----------------------------------------------------------------------------- main
def detect_plan(
    gray: NDArray[np.floating],
    scale: Scale | None = None,
    *,
    params: Params | None = None,
) -> PlanDetection:
    """Find walls, doorways, rooms and stairs in a grey image (1 = paper)."""
    p = params or get_params()
    h0, w0 = gray.shape
    f = max(1.0, max(h0, w0) / MAX_SIDE)
    g = gray
    if f > 1.0:
        g = ndimage.zoom(gray, 1.0 / f, order=1)
    warnings: list[str] = []

    ink = g < otsu_threshold(g)
    if ink.mean() > 0.5:  # white lines on dark paper
        ink = ~ink
    wall_px = stroke_thickness(ink)
    k = max(2, int(np.floor(0.75 * wall_px)))
    walls = ndimage.binary_opening(ink, structure=np.ones((k, k), dtype=bool))
    if scale is not None:
        m_per_px = scale.m_per_px() * f
        source: Literal["reference", "walls"] = "reference"
    else:
        m_per_px = p.scalar("vision.default_wall_thickness") / wall_px
        source = "walls"
        warnings.append(
            "Scale estimated from wall thickness (assumed "
            f"{p.scalar('vision.default_wall_thickness'):.2f} m); draw a reference line to fix it."
        )
    lab, n = ndimage.label(walls)
    if n:  # specks left by the opening
        sizes = ndimage.sum(walls, lab, index=np.arange(1, n + 1))
        walls[np.isin(lab, np.flatnonzero(sizes < 4 * k * k) + 1)] = False

    # ---- doorways: door-wide gaps between the ends of a wall line
    lmax = int(np.ceil(p.scalar("vision.door_width_max") / m_per_px))
    lmin = p.scalar("vision.door_width_min") / m_per_px
    gaps: list[tuple[NDArray[np.bool_], str]] = []
    for axis, img in (("v", walls), ("h", np.ascontiguousarray(walls.T))):
        mask = _jamb_gaps(img, max(1, int(0.8 * lmin)), lmax, wall_px)
        if axis == "h":
            mask = mask.T
        glab, gn = ndimage.label(mask)
        for sl, idx in zip(ndimage.find_objects(glab), range(1, gn + 1), strict=False):
            if sl is None:
                continue
            rows = sl[0].stop - sl[0].start
            cols = sl[1].stop - sl[1].start
            along, across = (cols, rows) if axis == "h" else (rows, cols)
            if 0.4 * wall_px <= across <= 1.8 * wall_px and along >= 0.8 * lmin:
                key = axis + f"{sl[0].start},{sl[1].start},{sl[0].stop},{sl[1].stop}"
                gaps.append((glab[sl] == idx, key))
    barrier = walls.copy()
    gap_boxes: list[tuple[str, int, int, int, int]] = []
    for m, key in gaps:
        axis = key[0]
        r0, c0, r1, c1 = (int(v) for v in key[1:].split(","))
        barrier[r0:r1, c0:c1] |= m
        gap_boxes.append((axis, r0, c0, r1, c1))
    # corridor mouths: two parallel walls ending side by side, a door apart
    for axis, r0, c0, r1, c1 in _mouths(walls, wall_px, lmin, lmax):
        if any(
            r0 < b1 + 2 and b0 < r1 + 2 and c0 < d1 + 2 and d0 < c1 + 2
            for _, b0, d0, b1, d1 in gap_boxes
        ):
            continue
        barrier[r0:r1, c0:c1] = True
        gap_boxes.append((axis, r0, c0, r1, c1))
    barrier = ndimage.binary_closing(barrier, structure=np.ones((3, 3), bool)) | barrier

    # ---- rooms
    free = ~barrier
    rlab, rn = ndimage.label(free)
    border = set(np.unique(np.concatenate([rlab[0], rlab[-1], rlab[:, 0], rlab[:, -1]])).tolist())
    border.discard(0)
    min_px = p.scalar("vision.min_room_area") / (m_per_px**2)
    areas = ndimage.sum(free, rlab, index=np.arange(1, rn + 1))
    room_labels = [i + 1 for i in range(rn) if (i + 1) not in border and areas[i] >= min_px]

    # ---- doorway -> rooms
    def side_label(r: float, c: float) -> int:
        ri, ci = round(r), round(c)
        if not (0 <= ri < rlab.shape[0] and 0 <= ci < rlab.shape[1]):
            return -1  # beyond the image: outside
        return int(rlab[ri, ci])

    doors: list[DetectedDoor] = []
    name = {lb: f"R{j + 1}" for j, lb in enumerate(room_labels)}
    for j, (axis, r0, c0, r1, c1) in enumerate(gap_boxes):
        rc, cc = (r0 + r1) / 2, (c0 + c1) / 2
        found: list[str] = []
        for sign in (-1, 1):
            votes: Counter[str] = Counter()
            for off in (wall_px + 3, 2 * wall_px + 3, 3 * wall_px + 4):
                for t in (0.3, 0.5, 0.7):
                    if axis == "h":
                        lb = side_label(rc + sign * off, c0 + t * (c1 - c0))
                    else:
                        lb = side_label(r0 + t * (r1 - r0), cc + sign * off)
                    if lb == -1 or lb in border:
                        votes["outside"] += 1
                    elif lb in name:
                        votes[name[lb]] += 1
                if votes:
                    break
            if votes:
                found.append(votes.most_common(1)[0][0])
        if len(found) != 2 or found[0] == found[1] or found == ["outside", "outside"]:
            continue
        if axis == "h":
            a, b = [c0 * f, rc * f], [c1 * f, rc * f]
            width = (c1 - c0) * m_per_px
        else:
            a, b = [cc * f, r0 * f], [cc * f, r1 * f]
            width = (r1 - r0) * m_per_px
        doors.append(DetectedDoor(id=f"D{j + 1}", a=a, b=b, width_m=round(width, 3), rooms=found))

    # ---- room geometry, stairs and types
    thin = ink & ~ndimage.binary_dilation(walls, iterations=1)
    tread_lo = p.scalar("vision.stair_tread_min") / m_per_px
    tread_hi = p.scalar("vision.stair_tread_max") / m_per_px
    count = Counter(r for d in doors for r in d.rooms)
    rooms: list[DetectedRoom] = []
    slices = ndimage.find_objects(rlab)
    h2 = wall_px / 2
    for lb in room_labels:
        sl = slices[lb - 1]
        if sl is None:
            continue
        region = rlab[sl] == lb
        rects_px = _rectangles(region, max(4, int(0.3 * min_px)))
        rects: list[list[float]] = []
        for r0, c0, r1, c1 in rects_px:
            gr0, gc0 = r0 + sl[0].start, c0 + sl[1].start
            gr1, gc1 = r1 + sl[0].start, c1 + sl[1].start
            box = [float(gc0), float(gr0), float(gc1), float(gr1)]
            # grow to the wall centre line where a wall runs along the side
            if gc0 - 1 >= 0 and barrier[gr0:gr1, gc0 - 1].mean() > 0.5:
                box[0] -= h2
            if barrier.shape[1] > gc1 and barrier[gr0:gr1, gc1].mean() > 0.5:
                box[2] += h2
            if gr0 - 1 >= 0 and barrier[gr0 - 1, gc0:gc1].mean() > 0.5:
                box[1] -= h2
            if barrier.shape[0] > gr1 and barrier[gr1, gc0:gc1].mean() > 0.5:
                box[3] += h2
            rects.append([round(v * f, 2) for v in box])
        if not rects:
            continue
        area = float(region.sum()) * m_per_px**2
        rid = name[lb]
        score = _stair_score(thin[sl] & region, region, tread_lo, tread_hi)
        xs = [r[0] for r in rects] + [r[2] for r in rects]
        ys = [r[1] for r in rects] + [r[3] for r in rects]
        long_side = max(max(xs) - min(xs), max(ys) - min(ys)) / f * m_per_px
        short_side = min(max(xs) - min(xs), max(ys) - min(ys)) / f * m_per_px
        width_est = area / max(long_side, 1e-9)
        nd = count.get(rid, 0)
        unit_type = None
        rtype: RoomType
        if nd == 0:
            rtype = "void"
        elif score >= 0.35:
            rtype = "stair"
        elif long_side / max(short_side, 1e-9) >= p.scalar("vision.corridor_min_aspect") or (
            width_est <= p.scalar("vision.corridor_max_width") and long_side >= 3 * width_est
        ):
            rtype = "corridor"
        elif nd >= 3:
            rtype = "lobby"
        else:
            rtype = "unit"
            if area <= p.scalar("vision.unit_area_small_max"):
                unit_type = "flat_small"
            elif area <= p.scalar("vision.unit_area_medium_max"):
                unit_type = "flat_medium"
            else:
                unit_type = "flat_large"
        rooms.append(
            DetectedRoom(
                id=rid,
                type=rtype,
                rects=rects,
                area_m2=round(area, 2),
                doors=nd,
                stair_score=round(score, 3),
                unit_type=unit_type,
            )
        )
    if not any(r.type == "stair" for r in rooms):
        warnings.append("No staircase found; mark the stair rooms in the editor.")
    if not any("outside" in d.rooms for d in doors):
        warnings.append(
            "No doorway to the outside found; exits will be assumed at the foot of each stair."
        )
    voids = sum(r.type == "void" for r in rooms)
    if voids:
        warnings.append(f"{voids} enclosed space(s) without a doorway (shafts, ducts) are ignored.")
    return PlanDetection(
        width=w0,
        height=h0,
        m_per_px=m_per_px / f,
        scale_source=source,
        wall_px=round(wall_px * f, 2),
        rooms=rooms,
        doors=doors,
        warnings=warnings,
    )
