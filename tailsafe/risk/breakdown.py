"""Who carries the tail? Breakdowns of the worst scenarios by occupant and floor.

In each *tail scenario* (loss at or above VaR_α) the **stragglers** are the
households still inside late in that scenario: those whose exit time is at least
``straggler_fraction`` × the scenario's loss. Comparing each category's share of
stragglers with its share of all occupants gives a **risk ratio**. A ratio of 5
means the category is five times over-represented among the people who make
the tail.

Categories use each household's *most dependent member* (a household with a
frail older adult counts as "frail older adult"), because households move
together.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tailsafe.building.builder import hk_level_label
from tailsafe.population.profiles import Profile
from tailsafe.risk.metrics import var
from tailsafe.scenarios.montecarlo import MCResult


@dataclass(frozen=True)
class ShareRow:
    """A category's weight among all occupants vs among tail stragglers."""

    category: str
    occupant_share: float
    straggler_share: float
    risk_ratio: float
    last_out_share: float


@dataclass(frozen=True)
class CellRow:
    """A (profile, floor band) cell's weight among occupants vs tail stragglers."""

    profile: str
    floor_band: str
    occupant_share: float
    straggler_share: float
    risk_ratio: float


def floor_band(level: int, size: int = 10) -> str:
    """Band label, e.g. ``'30/F–39/F'`` (``'G/F–9/F'`` for the lowest band)."""
    lo = (level // size) * size
    return f"{hk_level_label(lo)}–{hk_level_label(lo + size - 1)}"


def _eligible(loss: str, groups: dict[str, np.ndarray]) -> np.ndarray:
    if loss == "self_evacuation_time":
        return ~groups["rescued"].astype(bool)
    return np.ones(groups["exit"].size, dtype=bool)


def tail_breakdown(
    result: MCResult,
    loss: str = "total_time",
    *,
    alpha: float | None = None,
    straggler_fraction: float = 0.9,
    band_size: int = 10,
) -> dict[str, Any]:
    """Profile and floor-band composition of the stragglers in the tail scenarios."""
    if not result.runs or result.runs[0].groups is None:
        raise ValueError("breakdown needs a result run with keep_groups=True")
    a = alpha or result.config.alpha
    losses = result.loss(loss)
    threshold = var(losses, a)
    tail_idx = np.flatnonzero(losses >= threshold)

    occ: dict[str, dict[str, float]] = {"profile": defaultdict(float), "band": defaultdict(float)}
    strag: dict[str, dict[str, float]] = {"profile": defaultdict(float), "band": defaultdict(float)}
    last: dict[str, dict[str, float]] = {"profile": defaultdict(float), "band": defaultdict(float)}
    cells: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])

    def keys(prof: int, level: int) -> tuple[str, str]:
        return Profile(int(prof)).label, floor_band(int(level), band_size)

    for run in result.runs:
        g = run.groups
        assert g is not None
        for prof, level, size in zip(g["profile"], g["level"], g["size"], strict=True):
            kp, kb = keys(prof, level)
            occ["profile"][kp] += float(size)
            occ["band"][kb] += float(size)
            cells[(kp, kb)][0] += float(size)
    for k in tail_idx:
        g = result.runs[int(k)].groups
        assert g is not None
        ok = _eligible(loss, g) & np.isfinite(g["exit"])
        if not ok.any():
            continue
        cutoff = straggler_fraction * losses[k]
        late = ok & (g["exit"] >= cutoff)
        for i in np.flatnonzero(late):
            kp, kb = keys(g["profile"][i], g["level"][i])
            w = float(g["size"][i])
            strag["profile"][kp] += w
            strag["band"][kb] += w
            cells[(kp, kb)][1] += w
        i_last = int(np.flatnonzero(ok)[np.argmax(g["exit"][ok])])
        kp, kb = keys(g["profile"][i_last], g["level"][i_last])
        last["profile"][kp] += 1.0
        last["band"][kb] += 1.0

    def rows(dim: str) -> list[ShareRow]:
        tot_o = sum(occ[dim].values()) or 1.0
        tot_s = sum(strag[dim].values()) or 1.0
        tot_l = sum(last[dim].values()) or 1.0
        out = []
        for cat in occ[dim]:
            o = occ[dim][cat] / tot_o
            s = strag[dim].get(cat, 0.0) / tot_s
            out.append(
                ShareRow(cat, o, s, s / o if o > 0 else 0.0, last[dim].get(cat, 0.0) / tot_l)
            )
        return sorted(out, key=lambda r: -r.straggler_share)

    tot_o = sum(c[0] for c in cells.values()) or 1.0
    tot_s = sum(c[1] for c in cells.values()) or 1.0
    cell_rows = sorted(
        (
            CellRow(
                profile=kp,
                floor_band=kb,
                occupant_share=c[0] / tot_o,
                straggler_share=c[1] / tot_s,
                risk_ratio=(c[1] / tot_s) / (c[0] / tot_o) if c[0] > 0 else 0.0,
            )
            for (kp, kb), c in cells.items()
        ),
        key=lambda r: -r.straggler_share,
    )
    headline = ""
    if cell_rows and cell_rows[0].straggler_share > 0:
        top = cell_rows[0]
        headline = (
            f"Households whose most dependent member is a {top.profile}, on "
            f"{top.floor_band}, are {100 * top.occupant_share:.1f}% of occupants but "
            f"{100 * top.straggler_share:.0f}% of the people still inside late in the "
            f"worst {100 * (1 - a):.0f}% of scenarios."
        )
    return {
        "loss": loss,
        "alpha": a,
        "var": threshold,
        "tail_scenarios": int(tail_idx.size),
        "straggler_fraction": straggler_fraction,
        "profiles": [asdict(r) for r in rows("profile")],
        "floor_bands": [asdict(r) for r in rows("band")],
        "cells": [asdict(c) for c in cell_rows[:10]],
        "headline": headline,
    }
