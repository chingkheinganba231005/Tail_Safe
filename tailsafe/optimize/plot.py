"""Before/after comparison plot for an optimised plan."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from tailsafe import DISCLAIMER
from tailsafe.risk.metrics import cvar
from tailsafe.scenarios.montecarlo import MCResult

GREY, VERMILLION, BLUE, GREEN = "#555555", "#D55E00", "#0072B2", "#009E73"


def plot_before_after(
    before: MCResult,
    after: MCResult,
    plan_lines: list[str],
    losses: tuple[str, ...] = ("total_time", "p95_occupant_time"),
    alpha: float = 0.95,
) -> Figure:
    """Overlaid distributions (same scenarios) with CVaR markers, one panel per loss."""
    fig, axes = plt.subplots(1, len(losses), figsize=(6.2 * len(losses), 4.8), squeeze=False)
    for ax, loss in zip(axes[0], losses, strict=True):
        b = before.loss(loss) / 60
        a = after.loss(loss) / 60
        lo = float(min(b.min(), a.min()))
        hi = float(max(b.max(), a.max()))
        bins = np.linspace(lo, hi, 36)
        ax.hist(b, bins=bins, color=VERMILLION, alpha=0.45, label="baseline")
        ax.hist(a, bins=bins, color=BLUE, alpha=0.55, label="with plan")
        for x, colour, name in ((b, VERMILLION, "baseline"), (a, BLUE, "plan")):
            c = cvar(x, alpha)
            ax.axvline(c, color=colour, lw=1.8)
            ax.annotate(
                f"CVaR{int(100 * alpha)} {name}: {c:.0f} min",
                (c, ax.get_ylim()[1] * 0.95),
                color=colour,
                fontsize=8,
                rotation=90,
                va="top",
                xytext=(3, 0),
                textcoords="offset points",
            )
        ax.set_xlabel(f"{loss.replace('_', ' ')} (min)")
        ax.set_ylabel("scenarios")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    title = "Plan: " + " ".join(plan_lines)
    fig.suptitle(title if len(title) < 160 else title[:157] + "…", fontsize=9)
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=5.5, color=GREY, wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return fig


def save_before_after(
    before: MCResult, after: MCResult, plan_lines: list[str], out: Path, alpha: float = 0.95
) -> Path:
    """Render :func:`plot_before_after` to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_before_after(before, after, plan_lines, alpha=alpha)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
