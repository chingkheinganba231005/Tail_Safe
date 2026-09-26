"""Analytical validation report for the mesoscopic engine.

Runs the idealised cases of :mod:`tailsafe.sim.cases` and compares them with
hydraulic hand calculations. The same checks are asserted in
``tests/validation/test_analytical.py``; this module renders them as a table
for ``docs/validation.md`` (``tailsafe validate``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tailsafe.config import Params, get_params
from tailsafe.sim.cases import corridor_building, stair_tower, uniform_population
from tailsafe.sim.meso import SimConfig, run_meso
from tailsafe.sim.network import ARC_STAIR_DOWN, compile_network


@dataclass(frozen=True)
class CheckResult:
    """One validation comparison."""

    name: str
    quantity: str
    expected: float
    simulated: float
    tolerance: float  # relative, or absolute when ``absolute`` is set
    absolute: bool = False

    @property
    def error(self) -> float:
        """Relative (or absolute) deviation of the simulation from the reference."""
        if self.absolute:
            return abs(self.simulated - self.expected)
        return abs(self.simulated - self.expected) / abs(self.expected)

    @property
    def passed(self) -> bool:
        """Within tolerance."""
        return self.error <= self.tolerance


def run_checks(params: Params | None = None) -> list[CheckResult]:
    """Run every analytical validation case."""
    p = params or get_params()
    fs_h = p.scalar("movement.hydraulic.max_specific_flow_horizontal")
    fs_s = p.scalar("movement.hydraulic.max_specific_flow_stair")
    bl_door = p.scalar("movement.boundary_layer.door")
    bl_stair = p.scalar("movement.boundary_layer.stair")
    v_h = 0.85 * p.scalar("movement.hydraulic.k_horizontal")
    v_s = 0.85 * p.scalar("movement.hydraulic.k_stair")
    out: list[CheckResult] = []

    for n, w in ((100, 1.0), (300, 0.9), (60, 1.6)):
        b = corridor_building(length=30.0, width=2.4, exit_width=w)
        res = run_meso(b, uniform_population(b, n, h_speed=v_h), params=p)
        expected = 31.0 / v_h + n / (fs_h * (w - 2 * bl_door))
        out.append(
            CheckResult(
                f"Corridor 30 m, {n} people, {w:.1f} m exit door",
                "evacuation time (s)",
                expected,
                res.total_time,
                0.03,
            )
        )

    for floors, per in ((10, 30), (20, 15), (6, 60)):
        b = stair_tower(floors=floors, stair_width=1.2, door_width=1.2)
        net = compile_network(b, p)
        res = run_meso(net, uniform_population(b, per, h_speed=v_h, down_speed=v_s), params=p)
        flight = float(net.arc_len[net.arc_kind == ARC_STAIR_DOWN][0])
        first = 2.0 / v_h + flight / v_s + 1.0 / v_h
        expected = first + floors * per / (fs_s * (1.2 - 2 * bl_stair))
        out.append(
            CheckResult(
                f"Single 1.2 m stair, {floors} floors × {per} people",
                "evacuation time (s)",
                expected,
                res.total_time,
                0.03,
            )
        )

    b = stair_tower(floors=12)
    net = compile_network(b, p)
    res = run_meso(net, uniform_population(b, 40, h_speed=v_h, down_speed=v_s), params=p)
    ex = np.sort(res.agent_exit)
    mid = ex[(ex > 0.2 * ex[-1]) & (ex < 0.8 * ex[-1])]
    out.append(
        CheckResult(
            "Saturated stair discharge",
            "flow (persons/s)",
            float(net.arc_cap[net.arc_kind == ARC_STAIR_DOWN][0]),
            float((mid.size - 1) / (mid[-1] - mid[0])),
            0.03,
        )
    )

    for ratio in (0.3, 0.5, 0.7):
        pr = p.with_overrides({"movement.merge.floor_deference_ratio": ratio})
        b = stair_tower(floors=2, door_width=1.6)
        pop = uniform_population(b, 150, h_speed=v_h, down_speed=v_s)
        res = run_meso(compile_network(b, pr), pop, params=pr)
        floor1 = pop.group_level == 1
        t_end = min(res.agent_exit[floor1].max(), res.agent_exit[~floor1].max())
        window = (res.agent_exit > 20.0) & (res.agent_exit < t_end - 5.0)
        share = float((window & floor1).sum() / window.sum())
        out.append(
            CheckResult(
                f"Stair-landing merge, deference ratio {ratio}",
                "floor share of stair flow",
                ratio,
                share,
                0.04,
                absolute=True,
            )
        )

    b = stair_tower(floors=8)
    pop = uniform_population(b, 25, h_speed=v_h, down_speed=v_s)
    fine = run_meso(b, pop, config=SimConfig(dt=0.25), params=p).total_time
    coarse = run_meso(b, pop, config=SimConfig(dt=1.0), params=p).total_time
    out.append(
        CheckResult(
            "Time step 1.0 s vs 0.25 s (8-floor stair)",
            "evacuation time (s)",
            fine,
            coarse,
            0.01,
        )
    )
    return out


def markdown_table(results: list[CheckResult]) -> str:
    """Render results as a Markdown table."""
    lines = [
        "| Case | Quantity | Reference | Meso | Deviation | Tolerance | |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        dev = f"{r.error:.3f}" if r.absolute else f"{100 * r.error:.1f}%"
        tol = f"±{r.tolerance:.2f}" if r.absolute else f"{100 * r.tolerance:.0f}%"
        lines.append(
            f"| {r.name} | {r.quantity} | {r.expected:.2f} | {r.simulated:.2f} | {dev} | {tol} "
            f"| {'✅' if r.passed else '❌'} |"
        )
    return "\n".join(lines)
