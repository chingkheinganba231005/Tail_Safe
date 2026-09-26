"""Turn a (corrected) plan detection into a multi-storey building.

The plan is taken as the typical floor and repeated for every storey. Rooms
become nodes (a room split into rectangles becomes one node per rectangle,
joined by walkways), doorways become door edges with their openings, stair
rooms become the landings of one staircase each, stacked and joined by
flights. On the ground floor, doorways to the outside become final exits; if
the plan has none, an exit is assumed at the foot of every staircase (and
the building records that assumption).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from tailsafe.building.builder import BuildingBuilder
from tailsafe.building.geometry import Box, shared_boundary
from tailsafe.building.model import Building, EdgeKind, LevelKind, NodeType, StairKind
from tailsafe.config import Params, get_params
from tailsafe.vision.detect import DetectedDoor, DetectedRoom, PlanDetection

_TYPE = {
    "unit": NodeType.UNIT,
    "corridor": NodeType.CORRIDOR,
    "lobby": NodeType.LOBBY,
    "refuge": NodeType.REFUGE,
}


def _metres(det: PlanDetection, rect: list[float]) -> Box:
    """Image rectangle [x0, y0, x1, y1] (y down) -> plan box (x0, x1, y0, y1) in metres."""
    s = det.m_per_px
    x0, y0, x1, y1 = rect
    return (
        round(x0 * s, 3),
        round(x1 * s, 3),
        round((det.height - y1) * s, 3),
        round((det.height - y0) * s, 3),
    )


def _point(det: PlanDetection, xy: list[float]) -> tuple[float, float]:
    return round(xy[0] * det.m_per_px, 3), round((det.height - xy[1]) * det.m_per_px, 3)


def _box_distance(b: Box, x: float, y: float) -> float:
    dx = max(b[0] - x, 0.0, x - b[1])
    dy = max(b[2] - y, 0.0, y - b[3])
    return float(np.hypot(dx, dy))


def assign_doors(det: PlanDetection, reach_m: float = 0.6) -> list[DetectedDoor]:
    """Doorways with the two rooms they join (for doorways drawn in the editor).

    A doorway's rooms are the nearest room on each side of its line within
    ``reach_m``; with nothing on one side it opens to the outside.
    """
    out: list[DetectedDoor] = []
    rooms = [r for r in det.rooms if r.type != "void"]
    for d in det.doors:
        if len(d.rooms) == 2 and all(r == "outside" or _has(det, r) for r in d.rooms):
            out.append(d)
            continue
        (ax, ay), (bx, by) = _point(det, d.a), _point(det, d.b)
        mx, my = (ax + bx) / 2, (ay + by) / 2
        horiz = abs(bx - ax) >= abs(by - ay)
        sides: list[str] = []
        for sign in (-1.0, 1.0):
            px, py = (mx, my + sign * 0.3) if horiz else (mx + sign * 0.3, my)
            best, best_d = "outside", reach_m
            for r in rooms:
                dist = min(_box_distance(_metres(det, rc), px, py) for rc in r.rects)
                if dist <= best_d:
                    best, best_d = r.id, dist
            sides.append(best)
        if sides[0] != sides[1]:
            out.append(d.model_copy(update={"rooms": sides}))
    return out


def _has(det: PlanDetection, room_id: str) -> bool:
    return any(r.id == room_id and r.type != "void" for r in det.rooms)


def plan_to_building(
    det: PlanDetection,
    *,
    storeys: int,
    floor_height: float | None = None,
    ground_height: float | None = None,
    name: str = "Building from floor plan",
    params: Params | None = None,
) -> Building:
    """Stack the plan into a building of ``storeys`` storeys (G/F included)."""
    if storeys < 1:
        raise ValueError("storeys must be at least 1")
    p = params or get_params()
    bd = "building_defaults."
    fh = floor_height or p.scalar(bd + "floor_to_floor_height")
    gh = ground_height or p.scalar(bd + "ground_floor_height")
    doors = assign_doors(det)
    rooms: dict[str, DetectedRoom] = {r.id: r for r in det.rooms if r.type != "void"}
    stairs = sorted(
        (r for r in rooms.values() if r.type == "stair"),
        key=lambda r: (min(b[1] for b in r.rects), min(b[0] for b in r.rects)),
    )
    stair_id = {r.id: chr(ord("A") + i) for i, r in enumerate(stairs)}
    if not stairs and storeys > 1:
        raise ValueError("a building of more than one storey needs at least one staircase")

    digest = hashlib.sha256(det.model_dump_json().encode()).hexdigest()[:8]
    bld = BuildingBuilder(
        id=f"plan-{digest}",
        name=name,
        typology="custom",
        description=f"{storeys}-storey building from an uploaded floor plan "
        f"({len(rooms)} rooms, {len(doors)} doorways per floor).",
        metadata={"source": "floor plan", "m_per_px": det.m_per_px, "scale": det.scale_source},
    )
    for lv in range(storeys):
        elev = 0.0 if lv == 0 else gh + (lv - 1) * fh
        bld.level(
            lv,
            elevation=elev,
            height=gh if lv == 0 else fh,
            kind=LevelKind.GROUND if lv == 0 else LevelKind.TYPICAL,
        )
    for sr in stairs:
        sid = stair_id[sr.id]
        bld.stair(
            sid,
            f"Stair {sid}",
            kind=StairKind.DOGLEG,
            clear_width=p.scalar(bd + "stair_clear_width"),
            riser=p.scalar(bd + "stair_riser"),
            going=p.scalar(bd + "stair_going"),
        )

    def room_boxes(r: DetectedRoom) -> list[Box]:
        if r.type == "stair":  # one landing per floor
            bs = [_metres(det, rc) for rc in r.rects]
            return [
                (
                    min(b[0] for b in bs),
                    max(b[1] for b in bs),
                    min(b[2] for b in bs),
                    max(b[3] for b in bs),
                )
            ]
        return [_metres(det, rc) for rc in r.rects]

    boxes = {rid: room_boxes(r) for rid, r in rooms.items()}
    exits: list[tuple[str, DetectedDoor]] = []
    for lv in range(storeys):
        tag = f"L{lv:02d}"
        for rid, r in rooms.items():
            for k, (x0, x1, y0, y1) in enumerate(boxes[rid]):
                poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                if r.type == "stair":
                    sid = stair_id[rid]
                    bld.node(
                        f"{tag}.stair.{sid}",
                        NodeType.STAIR_LANDING,
                        lv,
                        polygon=poly,
                        stair=sid,
                        label=f"Stair {sid} landing",
                    )
                else:
                    bld.node(
                        f"{tag}.{rid}.{k}",
                        _TYPE[r.type],
                        lv,
                        polygon=poly,
                        unit_type=(r.unit_type or "flat_medium") if r.type == "unit" else None,
                        label=f"{r.type.capitalize()} {rid}",
                    )
            # walkways between the rectangles of one room
            bs = boxes[rid]
            for i in range(len(bs)):
                for j in range(i + 1, len(bs)):
                    seg = shared_boundary(bs[i], bs[j], tol=0.05)
                    if seg is None:
                        continue
                    width = float(np.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]))
                    if width >= 0.5:
                        bld.edge(f"{tag}.{rid}.{i}", f"{tag}.{rid}.{j}", EdgeKind.FLAT, width=width)

        def node_near(rid: str, x: float, y: float, tag: str = tag) -> str:
            r = rooms[rid]
            if r.type == "stair":
                return f"{tag}.stair.{stair_id[rid]}"
            k = int(np.argmin([_box_distance(b, x, y) for b in boxes[rid]]))
            return f"{tag}.{rid}.{k}"

        for d in doors:
            a, b = _point(det, d.a), _point(det, d.b)
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            inside = [r for r in d.rooms if r != "outside"]
            if "outside" in d.rooms:
                if lv == 0 and inside:
                    exits.append((node_near(inside[0], mx, my), d))
                continue
            u, v = node_near(inside[0], mx, my), node_near(inside[1], mx, my)
            stairish = any(rooms[x].type == "stair" for x in inside)
            bld.edge(
                u,
                v,
                EdgeKind.DOOR,
                width=d.width_m,
                opening=(a, b),
                fire_rated=stairish,
                self_closing=stairish,
                can_block=stairish,
                id=f"{tag}.{d.id}",
                label=f"Doorway {d.id} at {'G/F' if lv == 0 else f'{lv}/F'}",
            )
    assumed = False
    if not exits:
        assumed = True
        for sr in stairs:
            sid = stair_id[sr.id]
            x0, x1, y0, y1 = boxes[sr.id][0]
            # the landing side nearest the plan's edge
            W, H = det.width * det.m_per_px, det.height * det.m_per_px
            gaps = {"w": x0, "e": W - x1, "s": y0, "n": H - y1}
            side = min(gaps, key=gaps.__getitem__)
            w = min(p.scalar(bd + "stair_door_width"), 0.8 * min(x1 - x0, y1 - y0))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            seg = {
                "w": ((x0, cy - w / 2), (x0, cy + w / 2)),
                "e": ((x1, cy - w / 2), (x1, cy + w / 2)),
                "s": ((cx - w / 2, y0), (cx + w / 2, y0)),
                "n": ((cx - w / 2, y1), (cx + w / 2, y1)),
            }[side]
            out = {"w": (x0 - 1, cy), "e": (x1 + 1, cy), "s": (cx, y0 - 1), "n": (cx, y1 + 1)}[side]
            ex = f"L00.exit.stair{sid}"
            bld.node(ex, NodeType.EXIT, 0, at=out, label=f"Stair {sid} discharge (assumed)")
            bld.edge(
                f"L00.stair.{sid}",
                ex,
                EdgeKind.DOOR,
                width=w,
                opening=seg,
                label=f"Stair {sid} discharge door (assumed)",
            )
    else:
        for k, (node_id, d) in enumerate(exits):
            a, b = _point(det, d.a), _point(det, d.b)
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            n = bld.nodes[node_id]
            dx, dy = mx - n.x, my - n.y
            norm = float(np.hypot(dx, dy)) or 1.0
            ex = f"L00.exit.{k + 1}"
            bld.node(
                ex, NodeType.EXIT, 0, at=(mx + dx / norm, my + dy / norm), label=f"Exit {k + 1}"
            )
            bld.edge(
                node_id, ex, EdgeKind.DOOR, width=d.width_m, opening=(a, b), label=f"Exit {k + 1}"
            )
    bld.metadata["exits_assumed"] = assumed
    bld.connect_stair_flights()
    return bld.build()


def detection_summary(det: PlanDetection) -> dict[str, Any]:
    """Counts by room type and doorway kind (for the CLI and API)."""
    from collections import Counter

    return {
        "rooms": dict(Counter(r.type for r in det.rooms)),
        "doors": len(det.doors),
        "to_outside": sum("outside" in d.rooms for d in det.doors),
        "m_per_px": det.m_per_px,
        "scale": det.scale_source,
        "warnings": det.warnings,
    }
