"""Meso–micro agreement: both engines on the same sampled scenarios.

For each scenario the meso engine runs first; the micro engine then replays
the same people with the same decisions (see :mod:`tailsafe.sim.micro`).
Only people the micro engine actually walks are compared (households waiting
for lifts or rescue are taken from the meso run, so they would agree
trivially). Reported per metric: mean bias (micro − meso) with a bootstrap
CI, mean relative bias, Pearson correlation, RMSE and the largest difference.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tailsafe.building.model import Building
from tailsafe.config import Params, get_params
from tailsafe.rng import stream
from tailsafe.scenarios.sampler import ScenarioSampler, scenario_uniforms
from tailsafe.scenarios.spec import ScenarioSpec
from tailsafe.sim.meso import run_meso
from tailsafe.sim.micro import MicroConfig, run_micro
from tailsafe.sim.network import compile_network
from tailsafe.sim.routing import Router

METRICS = {
    "p50_s": "Time for half of the walkers to get out",
    "p95_s": "Time for 95% of the walkers to get out",
    "last_s": "Time the last walker gets out",
}


def _quantile(t: NDArray[np.float64], q: float) -> float:
    t = np.sort(t)
    if t.size == 0:
        return float("nan")
    return float(t[min(int(np.ceil(q * t.size)) - 1, t.size - 1)])


def _stats(meso: NDArray[np.float64], micro: NDArray[np.float64], seed: int) -> dict[str, float]:
    ok = np.isfinite(meso) & np.isfinite(micro)
    a, b = meso[ok], micro[ok]
    if a.size == 0:
        return {"n": 0}
    d = b - a
    rng = stream(seed, "agreement-bootstrap")
    boot = rng.choice(d, size=(2000, d.size), replace=True).mean(axis=1)
    corr = (
        float(np.corrcoef(a, b)[0, 1])
        if a.size > 2 and a.std() > 0 and b.std() > 0
        else float("nan")
    )
    return {
        "n": int(a.size),
        "mean_meso": float(a.mean()),
        "mean_micro": float(b.mean()),
        "bias": float(d.mean()),
        "bias_lo": float(np.quantile(boot, 0.025)),
        "bias_hi": float(np.quantile(boot, 0.975)),
        "relative_bias": float((d / a).mean()),
        "correlation": corr,
        "rmse": float(np.sqrt((d**2).mean())),
        "max_abs_diff": float(np.abs(d).max()),
    }


def meso_micro_agreement(
    building: Building,
    spec: ScenarioSpec,
    n_runs: int = 20,
    *,
    seed: int = 0,
    batch_size: int = 100,
    params: Params | None = None,
    config: MicroConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run both engines on scenarios ``0 .. n_runs-1`` and compare."""
    p = params or get_params()
    net = compile_network(building, p)
    router = Router(net, p)
    sampler = ScenarioSampler(building, spec, p)
    rows: list[dict[str, Any]] = []
    for idx in range(n_runs):
        u = scenario_uniforms(seed, idx, 1, lhs=True, batch_size=batch_size)[0]
        sc = sampler.sample(seed, idx, u)
        me = run_meso(net, sc.population, sc.sim, params=p, router=router)
        mi = run_micro(
            net,
            sc.population,
            sc.sim,
            config,
            params=p,
            router=router,
            meso=me,
            seed=seed,
            index=idx,
        )
        walked = mi.agent_walked
        t_meso = me.group_exit[sc.population.agent_group][walked]
        t_micro = mi.agent_exit[walked]
        rows.append(
            {
                "index": idx,
                "occupants": int(walked.size),
                "walkers": int(walked.sum()),
                "not_out_micro": int((~np.isfinite(t_micro)).sum()),
                "forced_moves": mi.forced_moves,
                "wall_time_s": mi.wall_time,
                "meso": {
                    "p50_s": _quantile(t_meso, 0.5),
                    "p95_s": _quantile(t_meso, 0.95),
                    "last_s": float(t_meso.max()) if t_meso.size else float("nan"),
                },
                "micro": {
                    "p50_s": _quantile(t_micro, 0.5),
                    "p95_s": _quantile(t_micro, 0.95),
                    "last_s": float(t_micro.max()) if t_micro.size else float("nan"),
                },
            }
        )
        if progress is not None:
            progress(idx + 1, n_runs)
    summary = {
        m: _stats(
            np.array([r["meso"][m] for r in rows]),
            np.array([r["micro"][m] for r in rows]),
            seed,
        )
        for m in METRICS
    }
    return {
        "building": building.name,
        "scenario": spec.description or spec.name,
        "runs": n_runs,
        "seed": seed,
        "scenarios": rows,
        "summary": summary,
        "not_out_total": int(sum(r["not_out_micro"] for r in rows)),
    }


def agreement_markdown(result: dict[str, Any]) -> str:
    """Markdown table of an agreement result (minutes)."""
    lines = [
        f"{result['runs']} scenarios of *{result['scenario']}* in {result['building']} "
        f"(seed {result['seed']}); walkers only; people never out in the micro run: "
        f"{result['not_out_total']}.",
        "",
        "| Metric | Meso mean | Micro mean | Bias micro − meso (95% CI) | Relative bias "
        "| Correlation | RMSE | Largest difference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m, label in METRICS.items():
        s = result["summary"][m]
        if not s.get("n"):
            continue
        lines.append(
            f"| {label} | {s['mean_meso'] / 60:.1f} min | {s['mean_micro'] / 60:.1f} min | "
            f"{s['bias'] / 60:+.1f} min ({s['bias_lo'] / 60:+.1f} to {s['bias_hi'] / 60:+.1f}) | "
            f"{100 * s['relative_bias']:+.0f}% | {s['correlation']:.2f} | "
            f"{s['rmse'] / 60:.1f} min | {s['max_abs_diff'] / 60:.1f} min |"
        )
    return "\n".join(lines) + "\n"
