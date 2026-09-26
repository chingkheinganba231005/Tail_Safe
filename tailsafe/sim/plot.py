"""Plots of a single simulation run: evacuation curve and stair congestion."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from tailsafe import DISCLAIMER
from tailsafe.sim.meso import MesoResult
from tailsafe.sim.micro import MicroResult
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


def plot_micro_frame(res: MicroResult, level: int, time: float, title: str | None = None) -> Figure:
    """Top-down snapshot of one floor in a micro replay at ``time`` (s)."""
    from tailsafe.sim._micro_kernel import S_FLIGHT, S_WAIT
    from tailsafe.sim.micro import compile_geometry

    net = res.net
    g = compile_geometry(net)
    k = int(np.clip(np.searchsorted(res.frame_times, time), 0, res.frame_times.size - 1))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for v in range(net.n_nodes):
        if net.node_level[v] != level or net.node_is_exit[v]:
            continue
        x0, x1, y0, y1 = g.n_x0[v], g.n_x1[v], g.n_y0[v], g.n_y1[v]
        shade = "#e8e7e1" if net.node_type[v].value != "unit" else "#fcfcfb"
        ax.add_patch(
            Rectangle((x0, y0), x1 - x0, y1 - y0, fc=shade, ec="#b0afa7", lw=0.6, zorder=1)
        )
    here = res.frame_level[k] == level
    st = res.frame_state[k]
    groups = (
        ("walking", here & (st != S_WAIT) & (st != S_FLIGHT), "#2a78d6", "o"),
        ("on the stairs", here & (st == S_FLIGHT), "#2a78d6", "^"),
        ("not yet moving", here & (st == S_WAIT), "#898781", "o"),
    )
    for label, sel, color, marker in groups:
        if sel.any():
            ax.scatter(
                res.frame_x[k, sel],
                res.frame_y[k, sel],
                s=14,
                c=color,
                marker=marker,
                label=f"{label} ({int(sel.sum())})",
                zorder=3,
                linewidths=0,
            )
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("m")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title(title or f"Level {level} at {res.frame_times[k] / 60:.1f} min", fontsize=10)
    fig.text(0.01, 0.005, DISCLAIMER, fontsize=5.5, color="#666666", wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def save_micro_frame(
    res: MicroResult, level: int, time: float, out: Path, title: str | None = None
) -> Path:
    """Write :func:`plot_micro_frame` to ``out``."""
    fig = plot_micro_frame(res, level, time, title)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
