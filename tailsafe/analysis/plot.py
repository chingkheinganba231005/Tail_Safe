"""Chart of the counterfactual bottleneck ranking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from tailsafe import DISCLAIMER

VERMILLION, BLUE = "#D55E00", "#0072B2"


def plot_bottlenecks(table: dict[str, Any], top: int = 8) -> Figure:
    """Horizontal bars: change in CVaR (minutes) per relaxed element, with 95% CIs."""
    rows = [r for r in table["ranking"] if r["delta_cvar"] is not None][:top]
    labels = [r["label"] for r in rows][::-1]
    val = np.array([r["delta_cvar"]["value"] for r in rows][::-1]) / 60
    lo = np.array([r["delta_cvar"]["lo"] for r in rows][::-1]) / 60
    hi = np.array([r["delta_cvar"]["hi"] for r in rows][::-1]) / 60
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 1.8))
    colours = [VERMILLION if r["kind"] == "unblock" else BLUE for r in rows][::-1]
    y = np.arange(len(rows))
    ax.barh(y, val, color=colours, alpha=0.85)
    ax.errorbar(val, y, xerr=[val - lo, hi - val], fmt="none", ecolor="#333333", capsize=3, lw=1)
    ax.set_yticks(y, labels, fontsize=8)
    ax.axvline(0, color="#333333", lw=0.8)
    alpha = int(100 * table["alpha"])
    ax.set_xlabel(f"change in CVaR{alpha} of {table['loss'].replace('_', ' ')} (min)")
    ax.set_title(
        f"What if … (capacity ×{table['factor']:g}, or the blocked stair kept usable)",
        fontsize=9,
    )
    ax.grid(alpha=0.3, axis="x")
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=5, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def save_bottleneck_plot(table: dict[str, Any], out: Path) -> Path:
    """Render :func:`plot_bottlenecks` to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_bottlenecks(table)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
