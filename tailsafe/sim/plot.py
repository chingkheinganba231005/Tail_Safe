"""Plots of a single simulation run: evacuation curve and stair congestion."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from tailsafe import DISCLAIMER
from tailsafe.sim.meso import MesoResult
from tailsafe.sim.network import ARC_STAIR_DOWN


def plot_run(res: MesoResult, title: str | None = None) -> Figure:
    """Cumulative occupants out over time, plus per-stair queue heatmaps.

    The heatmaps need a run with ``SimConfig(record_series=True)``.
    """
    net = res.net
    stairs = net.stair_ids if res.series is not None else []
    fig, axes = plt.subplots(
        1, 1 + len(stairs), figsize=(5 + 4.2 * len(stairs), 4.8), squeeze=False
    )
    ax = axes[0, 0]
    exits = np.sort(res.agent_exit[np.isfinite(res.agent_exit)]) / 60.0
    ax.step(exits, np.arange(1, exits.size + 1), where="post", color="#0072B2", lw=1.6)
    for q, style in ((0.5, ":"), (0.95, "--")):
        t = res.exit_time_quantile(q) / 60.0
        if np.isfinite(t):
            ax.axvline(t, color="#555555", ls=style, lw=1)
            ax.annotate(
                f"P{int(q * 100)}", (t, 0), fontsize=7, xytext=(2, 4), textcoords="offset points"
            )
    ax.set_xlabel("time since alarm (min)")
    ax.set_ylabel("occupants out of the building")
    ax.set_title(title or "Evacuation curve", fontsize=9)
    ax.grid(alpha=0.3)
    if res.series is not None and res.series_times is not None:
        times = res.series_times / 60.0
        vmax = max(float(res.series.max()), 1.0)
        for k, sid in enumerate(stairs):
            s = net.stair_ids.index(sid)
            flights = np.flatnonzero((net.arc_stair == s) & (net.arc_kind == ARC_STAIR_DOWN))
            levels = net.node_level[net.arc_src[flights]]
            order = np.argsort(levels)
            grid = res.series[flights[order]]
            axk = axes[0, k + 1]
            im = axk.imshow(
                grid,
                aspect="auto",
                origin="lower",
                cmap="cividis",
                vmin=0,
                vmax=vmax,
                extent=(
                    times[0],
                    times[-1] if times.size > 1 else 1.0,
                    levels.min() - 0.5,
                    levels.max() + 0.5,
                ),
                interpolation="nearest",
            )
            axk.set_title(f"Stair {sid}: people queueing at each flight", fontsize=9)
            axk.set_xlabel("time since alarm (min)")
            axk.set_ylabel("flight starting at level")
            fig.colorbar(im, ax=axk, fraction=0.046, pad=0.02, label="persons")
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=5.5, color="#555555", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def save_run_plot(res: MesoResult, out: Path, title: str | None = None) -> Path:
    """Render :func:`plot_run` to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_run(res, title)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
