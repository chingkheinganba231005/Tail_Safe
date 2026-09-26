"""Precision and recall of the plan reader on rendered plans with known truth.

A detected doorway matches a true door (or open passage) when their centres
are within ``tol_m`` and their orientations agree; each truth item matches at
most once. A detected stair matches a true stair when their boxes overlap by
at least 30% of the smaller one. These are *synthetic* plans rendered from
the procedural templates, so the numbers are an upper bound for real scans.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tailsafe.building.templates import generate
from tailsafe.config import Params, get_params
from tailsafe.vision.detect import PlanDetection, detect_plan
from tailsafe.vision.synth import PlanTruth, render_plan


def _centre(seg: tuple[float, float, float, float] | list[float]) -> tuple[float, float, bool]:
    c0, r0, c1, r1 = seg
    return (c0 + c1) / 2, (r0 + r1) / 2, abs(c1 - c0) >= abs(r1 - r0)


def match_doors(det: PlanDetection, truth: PlanTruth, tol_m: float = 0.5) -> dict[str, Any]:
    """Greedy nearest matching of detected doorways to true doors and openings."""
    tol = tol_m / truth.m_per_px
    items = [("door", s) for s in truth.doors] + [("opening", s) for s in truth.openings]
    used = [False] * len(items)
    tp_door = tp_open = 0
    unmatched: list[str] = []
    for d in det.doors:
        cx, cy, horiz = _centre([d.a[0], d.a[1], d.b[0], d.b[1]])
        best, best_d = -1, tol
        for k, (_, s) in enumerate(items):
            if used[k]:
                continue
            tx, ty, th = _centre(s)
            dist = float(np.hypot(tx - cx, ty - cy))
            if th == horiz and dist <= best_d:
                best, best_d = k, dist
        if best >= 0:
            used[best] = True
            if items[best][0] == "door":
                tp_door += 1
            else:
                tp_open += 1
        else:
            unmatched.append(d.id)
    n_doors = len(truth.doors)
    return {
        "detected": len(det.doors),
        "true_doors": n_doors,
        "true_openings": len(truth.openings),
        "matched_doors": tp_door,
        "matched_openings": tp_open,
        "false_positives": len(unmatched),
        "unmatched": unmatched,
        "precision": (tp_door + tp_open) / len(det.doors) if det.doors else 1.0,
        "door_recall": tp_door / n_doors if n_doors else 1.0,
    }


def match_stairs(det: PlanDetection, truth: PlanTruth) -> dict[str, Any]:
    """Stair rooms against true stair boxes (overlap ≥ 30% of the smaller box)."""
    found = [r for r in det.rooms if r.type == "stair"]
    used = [False] * len(truth.stairs)
    tp = 0
    for r in found:
        x0 = min(b[0] for b in r.rects)
        y0 = min(b[1] for b in r.rects)
        x1 = max(b[2] for b in r.rects)
        y1 = max(b[3] for b in r.rects)
        for k, (a0, b0, a1, b1) in enumerate(truth.stairs):
            if used[k]:
                continue
            ix = max(0.0, min(x1, a1) - max(x0, a0))
            iy = max(0.0, min(y1, b1) - max(y0, b0))
            small = min((x1 - x0) * (y1 - y0), (a1 - a0) * (b1 - b0))
            if small > 0 and ix * iy >= 0.3 * small:
                used[k] = True
                tp += 1
                break
    return {
        "detected": len(found),
        "true": len(truth.stairs),
        "matched": tp,
        "precision": tp / len(found) if found else 1.0,
        "recall": tp / len(truth.stairs) if truth.stairs else 1.0,
    }


SUITE: tuple[tuple[str, dict[str, Any]], ...] = (
    ("cruciform", {"storeys": 4, "flats_per_wing": 3}),
    ("slab", {"storeys": 4, "flats_per_side": 6}),
    ("twin_core", {"storeys": 6, "podium_levels": 1}),
    ("care_home", {"storeys": 2}),
)
QUALITIES: tuple[tuple[str, dict[str, float]], ...] = (
    ("clean, 20 px/m", {"px_per_m": 20.0}),
    ("noisy and blurred, 20 px/m", {"px_per_m": 20.0, "noise": 0.06, "blur": 0.8}),
    ("low resolution, 12 px/m", {"px_per_m": 12.0, "noise": 0.03, "blur": 0.5}),
)


def evaluate(*, params: Params | None = None, reference_scale: bool = False) -> dict[str, Any]:
    """Run the reader on the synthetic suite (ground and a typical floor of each template)."""
    from tailsafe.vision.detect import Scale

    p = params or get_params()
    rows: list[dict[str, Any]] = []
    for template, opts in SUITE:
        b = generate(template, **opts)
        levels = sorted({0, max(lv.index for lv in b.levels) - 1})
        for lv in levels:
            for quality, render in QUALITIES:
                img, truth = render_plan(
                    b,
                    lv,
                    seed=lv,
                    px_per_m=render["px_per_m"],
                    noise=render.get("noise", 0.0),
                    blur=render.get("blur", 0.0),
                )
                scale = None
                if reference_scale:
                    scale = Scale(x1=0, y1=0, x2=10 / truth.m_per_px, y2=0, metres=10.0)
                det = detect_plan(img, scale, params=p)
                rows.append(
                    {
                        "template": template,
                        "level": lv,
                        "quality": quality,
                        "scale_error": det.m_per_px / truth.m_per_px - 1.0,
                        "doors": match_doors(det, truth),
                        "stairs": match_stairs(det, truth),
                        "exits_true": len(truth.exits),
                        "exits_found": sum("outside" in d.rooms for d in det.doors),
                    }
                )
    agg: dict[str, Any] = {}
    for quality, _ in QUALITIES:
        sel = [r for r in rows if r["quality"] == quality]
        det_n = sum(r["doors"]["detected"] for r in sel)
        tp = sum(r["doors"]["matched_doors"] + r["doors"]["matched_openings"] for r in sel)
        true_doors = sum(r["doors"]["true_doors"] for r in sel)
        md = sum(r["doors"]["matched_doors"] for r in sel)
        sd = sum(r["stairs"]["detected"] for r in sel)
        st = sum(r["stairs"]["true"] for r in sel)
        sm = sum(r["stairs"]["matched"] for r in sel)
        agg[quality] = {
            "plans": len(sel),
            "door_precision": tp / det_n if det_n else 1.0,
            "door_recall": md / true_doors if true_doors else 1.0,
            "stair_precision": sm / sd if sd else 1.0,
            "stair_recall": sm / st if st else 1.0,
            "max_scale_error": max(abs(r["scale_error"]) for r in sel),
        }
    return {"plans": rows, "summary": agg, "reference_scale": reference_scale}


def evaluation_markdown(res: dict[str, Any]) -> str:
    """Summary table of :func:`evaluate`."""
    lines = [
        "| Plan quality | Plans | Door precision | Door recall | Stair precision "
        "| Stair recall | Largest scale error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for quality, s in res["summary"].items():
        lines.append(
            f"| {quality} | {s['plans']} | {s['door_precision']:.2f} | {s['door_recall']:.2f} | "
            f"{s['stair_precision']:.2f} | {s['stair_recall']:.2f} | "
            f"{100 * s['max_scale_error']:.0f}% |"
        )
    return "\n".join(lines) + "\n"
