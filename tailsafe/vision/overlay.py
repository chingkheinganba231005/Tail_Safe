"""Draw a plan detection over its image (for the CLI and debugging)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from tailsafe.vision.detect import PlanDetection

_FILL = {
    "unit": "#fcfcfb",
    "corridor": "#c3c2b7",
    "lobby": "#898781",
    "stair": "#2a78d6",
    "refuge": "#1baf7a",
    "void": "#e1e0d9",
}


def save_overlay(img: NDArray[np.floating], det: PlanDetection, out: Path) -> Path:
    """PNG of the plan with rooms (by type), doorways and their ids."""
    fig, ax = plt.subplots(figsize=(9, 9 * img.shape[0] / max(img.shape[1], 1)))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    for r in det.rooms:
        for x0, y0, x1, y1 in r.rects:
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0, fc=_FILL[r.type], ec="#52514e", alpha=0.45, lw=0.6
                )
            )
        x0, y0, x1, y1 = r.rects[0]
        ax.text(
            (x0 + x1) / 2, (y0 + y1) / 2, f"{r.id}\n{r.type}", fontsize=5, ha="center", va="center"
        )
    for d in det.doors:
        ax.plot([d.a[0], d.b[0]], [d.a[1], d.b[1]], color="#eb6834", lw=2.5)
        ax.text(d.a[0], d.a[1], d.id, fontsize=4, color="#0b0b0b")
    ax.set_axis_off()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
