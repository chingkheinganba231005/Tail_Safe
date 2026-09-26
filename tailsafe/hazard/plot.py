"""Plots of a hazard run: where and when conditions become untenable."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from tailsafe import DISCLAIMER
from tailsafe.building.model import Building, NodeType
from tailsafe.hazard.model import HazardResult

_CIRC = {NodeType.CORRIDOR, NodeType.LOBBY, NodeType.LIFT_LOBBY, NodeType.PROTECTED_LOBBY}


def _per_level(b: Building, hz: HazardResult, types: set[NodeType], stair: str | None = None):  # type: ignore[no-untyped-def]
    assert hz.visibility is not None
    levels = sorted(lv.index for lv in b.levels)
    grid = np.full((len(levels), hz.times.size), 1e3, dtype=np.float32)
    for i, n in enumerate(b.nodes):
        if n.type in types and (stair is None or n.stair == stair):
            r = levels.index(n.level)
            grid[r] = np.minimum(grid[r], hz.visibility[:, i])
    return levels, grid


def plot_hazard(b: Building, hz: HazardResult) -> Figure:
    """Worst visibility per floor over time: circulation spaces, then each stair."""
    if hz.visibility is None:
        raise ValueError("run the hazard model with keep_fields=True to plot it")
    panels = [("Corridors and lobbies", _CIRC, None)] + [
        (f"Stair {s.id}", {NodeType.STAIR_LANDING}, s.id) for s in b.stairs
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6), squeeze=False)
    t_min = hz.times / 60.0
    im = None
    for ax, (title, types, stair) in zip(axes[0], panels, strict=True):
        levels, grid = _per_level(b, hz, types, stair)
        im = ax.imshow(
            np.clip(grid, 0.3, 100.0),
            aspect="auto",
            origin="lower",
            cmap="cividis",
            norm=LogNorm(vmin=0.3, vmax=100.0),
            extent=(t_min[0], t_min[-1], levels[0] - 0.5, levels[-1] + 0.5),
            interpolation="nearest",
        )
        ax.set_title(f"{title}: worst visibility (m)", fontsize=9)
        ax.set_xlabel("time since ignition (min)")
        ax.set_ylabel("level")
        fire_level = b.node_by_id[hz.fire.node].level
        ax.axhline(fire_level, color="#D55E00", lw=0.8, ls="--")
    assert im is not None
    fig.colorbar(im, ax=axes[0, -1], fraction=0.046, pad=0.02, label="visibility (m)")
    fig.suptitle(
        f"Fire in {b.describe_node(hz.fire.node)} — growth {hz.fire.growth:g} kW/s², "
        f"peak {hz.fire.peak:.0f} kW, door {'open' if hz.fire.door_open else 'closed'}",
        fontsize=9,
    )
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=5.5, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


def save_hazard_plot(b: Building, hz: HazardResult, out: Path) -> Path:
    """Render :func:`plot_hazard` to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_hazard(b, hz)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
