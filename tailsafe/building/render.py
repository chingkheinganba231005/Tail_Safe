"""Static rendering of buildings with Matplotlib (plan view + 3D stack).

The web frontend renders interactively from the same JSON; this module is
for quick inspection, reports and tests. Colours follow the Okabe–Ito
colour-blind-safe palette.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tailsafe import DISCLAIMER
from tailsafe.building.model import Building, EdgeKind, LevelKind, NodeType

NODE_COLOURS: dict[NodeType, str] = {
    NodeType.UNIT: "#DDE5EE",
    NodeType.CORRIDOR: "#F4EBDD",
    NodeType.LOBBY: "#F4EBDD",
    NodeType.LIFT_LOBBY: "#E69F00",
    NodeType.PROTECTED_LOBBY: "#56B4E9",
    NodeType.STAIR_LANDING: "#0072B2",
    NodeType.REFUGE: "#009E73",
    NodeType.OPEN_AREA: "#F0E442",
    NodeType.EXIT: "#D55E00",
}
FEATURE_COLOURS = {"lift_shaft": "#555555", "service": "#BBBBBB", "void": "#FFFFFF"}
DOOR_COLOUR = "#CC79A7"


def default_plan_level(b: Building) -> int:
    """First typical level, else the ground floor."""
    for lv in b.levels:
        if lv.kind == LevelKind.TYPICAL:
            return lv.index
    return b.levels[0].index


def draw_plan(ax: Axes, b: Building, level: int, *, graph: bool = True) -> None:
    """Draw one storey: spaces, features, door openings and the egress graph."""
    lv = b.level_by_index[level]
    for f in lv.features:
        ax.add_patch(
            Polygon(
                f.polygon,
                closed=True,
                facecolor=FEATURE_COLOURS.get(f.kind, "#CCCCCC"),
                edgecolor="#777777",
                linewidth=0.4,
                hatch="//" if f.kind == "void" else None,
            )
        )
    nodes = [n for n in b.nodes if n.level == level]
    for n in nodes:
        if n.polygon:
            ax.add_patch(
                Polygon(
                    n.polygon,
                    closed=True,
                    facecolor=NODE_COLOURS[n.type],
                    edgecolor="#333333",
                    linewidth=0.6,
                    alpha=0.9,
                )
            )
    ids = {n.id for n in nodes}
    for e in b.edges:
        if e.source not in ids and e.target not in ids:
            continue
        if e.opening is not None:
            (x1, y1), (x2, y2) = e.opening
            ax.plot([x1, x2], [y1, y2], color=DOOR_COLOUR, linewidth=2.2, solid_capstyle="butt")
        if graph and e.kind != EdgeKind.STAIR:
            a, c = b.node_by_id[e.source], b.node_by_id[e.target]
            ax.plot([a.x, c.x], [a.y, c.y], color="#222222", linewidth=0.5, alpha=0.6)
    if graph:
        for n in nodes:
            ax.plot(n.x, n.y, "o", markersize=2.0, color="#222222")
    exits = [n for n in b.exits if n.level == level]
    for n in exits:
        ax.plot(n.x, n.y, marker="s", markersize=7, color=NODE_COLOURS[NodeType.EXIT])
        ax.annotate(
            n.label or n.id, (n.x, n.y), fontsize=6, xytext=(3, 3), textcoords="offset points"
        )
    for n in nodes:
        if n.type == NodeType.STAIR_LANDING and n.stair:
            ax.annotate(
                n.stair,
                (n.x, n.y),
                fontsize=8,
                color="white",
                ha="center",
                va="center",
                weight="bold",
            )
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(f"{b.name} — plan at {lv.label}", fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def draw_stack(ax: Axes, b: Building) -> None:
    """3D stack of translucent floors; refuge floors and stairs highlighted."""
    for lv in b.levels:
        z = lv.elevation
        polys, colours = [], []
        for n in b.nodes:
            if n.level != lv.index or not n.polygon:
                continue
            if n.type == NodeType.UNIT:
                colour = NODE_COLOURS[NodeType.UNIT]
            else:
                colour = NODE_COLOURS[n.type]
            polys.append([(x, y, z) for x, y in n.polygon])
            colours.append(colour)
        if not polys:
            continue
        alpha = 0.55 if lv.kind == LevelKind.REFUGE else 0.12
        coll = Poly3DCollection(polys, facecolors=colours, edgecolors="none", alpha=alpha)
        ax.add_collection3d(coll)  # type: ignore[attr-defined]
    # Stairs as vertical lines through their landings.
    by_stair: dict[str, list[tuple[float, float, float]]] = {}
    for n in b.nodes_of_type(NodeType.STAIR_LANDING):
        if n.stair:
            by_stair.setdefault(n.stair, []).append((n.x, n.y, b.level_by_index[n.level].elevation))
    for sid, pts in by_stair.items():
        pts.sort(key=lambda p: p[2])
        xs, ys, zs = zip(*pts, strict=True)
        ax.plot(xs, ys, zs, color=NODE_COLOURS[NodeType.STAIR_LANDING], linewidth=1.6)
        ax.text(xs[-1], ys[-1], zs[-1] + 2, f"Stair {sid}", fontsize=6)  # type: ignore[arg-type]
    all_x = [p[0] for n in b.nodes if n.polygon for p in n.polygon]
    all_y = [p[1] for n in b.nodes if n.polygon for p in n.polygon]
    top = b.levels[-1].elevation + b.levels[-1].height
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(0, top)  # type: ignore[attr-defined]
    span = max(max(all_x) - min(all_x), max(all_y) - min(all_y))
    ax.set_box_aspect((1, 1, min(3.0, top / max(span, 1.0))))  # type: ignore[arg-type]
    ax.view_init(elev=18, azim=-60)  # type: ignore[attr-defined]
    refuges = [lv.label for lv in b.levels if lv.kind == LevelKind.REFUGE]
    title = f"{b.n_storeys} storeys"
    if refuges:
        title += f"; refuge floor(s): {', '.join(refuges)}"
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def render_building(b: Building, *, level: int | None = None, dpi: int = 150) -> Figure:
    """Figure with the plan of ``level`` (default: first typical) and the 3D stack."""
    fig = plt.figure(figsize=(14, 7.5))
    ax_plan = fig.add_subplot(1, 2, 1)
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    draw_plan(ax_plan, b, default_plan_level(b) if level is None else level)
    draw_stack(ax_3d, b)
    handles = [
        Patch(facecolor=c, edgecolor="#333333", label=t.value.replace("_", " "))
        for t, c in NODE_COLOURS.items()
        if any(n.type == t for n in b.nodes)
    ]
    handles.append(Patch(facecolor=DOOR_COLOUR, label="door opening"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, frameon=False)
    fig.text(0.5, 0.005, DISCLAIMER, ha="center", fontsize=5.5, color="#555555", wrap=True)
    fig.set_dpi(dpi)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def save_render(b: Building, out: Path, *, level: int | None = None, dpi: int = 150) -> Path:
    """Render and save to ``out`` (PNG, SVG or PDF by extension)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = render_building(b, level=level, dpi=dpi)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out
