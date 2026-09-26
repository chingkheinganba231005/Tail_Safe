"""Plots of a Monte Carlo stress test: loss distribution and tail breakdown."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from tailsafe import DISCLAIMER
from tailsafe.scenarios.montecarlo import MCResult

BLUE, ORANGE, VERMILLION, GREY = "#0072B2", "#E69F00", "#D55E00", "#555555"


def plot_stress(
    result: MCResult, breakdown: dict[str, Any] | None, loss: str = "total_time"
) -> Figure:
    """Histogram of the loss with mean / P95 / CVaR markers, and who carries the tail."""
    risk = result.risk(loss)
    x = result.loss(loss) / 60.0
    ncols = 2 if breakdown else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4.6), squeeze=False)
    ax = axes[0, 0]
    finite = x[np.isfinite(x)]
    ax.hist(finite, bins=min(40, max(10, finite.size // 10)), color=BLUE, alpha=0.75)
    ymax = ax.get_ylim()[1]
    for est, name, colour, style in (
        (risk.mean, "mean", GREY, ":"),
        (risk.p95, "P95", ORANGE, "--"),
        (risk.cvar, f"CVaR{int(100 * risk.alpha)}", VERMILLION, "-"),
    ):
        v, lo, hi = est.value / 60, est.lo / 60, est.hi / 60
        ax.axvspan(lo, hi, color=colour, alpha=0.15, lw=0)
        ax.axvline(v, color=colour, ls=style, lw=1.6)
        ax.annotate(
            f"{name} {v:.0f} min",
            (v, ymax * 0.92),
            color=colour,
            fontsize=8,
            rotation=90,
            xytext=(3, 0),
            textcoords="offset points",
            va="top",
        )
    label = loss.replace("_", " ")
    ax.set_xlabel(f"{label} (min)")
    ax.set_ylabel("scenarios")
    ax.set_title(f"{result.building_name}\n{result.n} scenarios: {result.spec.name}", fontsize=9)
    ax.grid(alpha=0.3)
    if breakdown:
        axb = axes[0, 1]
        rows = [r for r in breakdown["profiles"] if r["occupant_share"] > 0][:6]
        names = [r["category"] for r in rows][::-1]
        occ = [100 * r["occupant_share"] for r in rows][::-1]
        strag = [100 * r["straggler_share"] for r in rows][::-1]
        y = np.arange(len(names))
        axb.barh(y - 0.2, occ, height=0.4, color=GREY, alpha=0.6, label="share of all occupants")
        axb.barh(y + 0.2, strag, height=0.4, color=VERMILLION, label="share of tail stragglers")
        axb.set_yticks(y, names, fontsize=8)
        axb.set_xlabel("%")
        axb.set_title(
            f"Who is still inside late in the worst {100 * (1 - breakdown['alpha']):.0f}%",
            fontsize=9,
        )
        axb.legend(fontsize=7, loc="lower right")
        axb.grid(alpha=0.3, axis="x")
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=5.5, color=GREY, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def save_stress_plot(
    result: MCResult, out: Path, breakdown: dict[str, Any] | None = None, loss: str = "total_time"
) -> Path:
    """Render :func:`plot_stress` to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_stress(result, breakdown, loss)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
