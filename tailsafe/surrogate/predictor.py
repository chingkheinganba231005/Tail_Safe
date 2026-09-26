"""Load the shipped surrogate and predict for a building and scenario."""

from __future__ import annotations

import inspect
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from tailsafe.building.model import Building
from tailsafe.building.templates import TEMPLATES
from tailsafe.config import Params, get_params
from tailsafe.scenarios.spec import ScenarioSpec
from tailsafe.surrogate import DEFAULT_WEIGHTS
from tailsafe.surrogate.data import SHARE_65_RANGE, TYPOLOGIES
from tailsafe.surrogate.features import StaticGraph, graph_features, static_graph
from tailsafe.surrogate.model import HORIZON, LOSS_NAMES, QUANTILES, load_model, predict


def coverage_notes(building: Building, spec: ScenarioSpec) -> list[str]:
    """Where a (building, scenario) lies outside what the surrogate was trained on.

    Empty when the case looks like the training draw in ``surrogate/data.py``;
    otherwise one plain sentence per departure, so the UI can say the estimate
    is an extrapolation and point to the full simulation.
    """
    notes: list[str] = []
    gen = building.metadata.get("generator")
    args = building.metadata.get("args", {})
    if gen not in TYPOLOGIES:
        notes.append(
            "The building is not one of the four generated templates (for example read "
            "from a floor plan or uploaded)."
        )
    else:
        for key, (lo, hi) in TYPOLOGIES[gen].items():
            v = args.get(key)
            if v is not None and not lo <= v <= hi:
                notes.append(f"{key.replace('_', ' ')} = {v} is outside the trained {lo}–{hi}.")
        defaults = {
            k: prm.default
            for k, prm in inspect.signature(TEMPLATES[gen]).parameters.items()
            if prm.default is not inspect.Parameter.empty and k not in TYPOLOGIES[gen]
        }
        for key, default in defaults.items():
            if key in args and args[key] != default:
                notes.append(f"{key.replace('_', ' ')} was not varied in training.")
    s_lo, s_hi = SHARE_65_RANGE
    if spec.share_65_plus is not None and not s_lo <= spec.share_65_plus <= s_hi:
        notes.append(f"Share aged 65+ outside the trained {s_lo:.0%}–{s_hi:.0%}.")
    if len(spec.stair_blockages) > 1:
        notes.append("More than one staircase lost (training lost at most one).")
    if any(bk.time.value is None for bk in spec.stair_blockages):
        notes.append("A staircase lost at a random time (training used fixed times).")
    if spec.lifts_out_of_service > 1:
        notes.append("More than one lift out of service (training had at most one).")
    unseen = {
        "a random staircase loss": spec.random_stair_blockage is not None,
        "phased release": bool(spec.phased_release),
        "stair assignment": bool(spec.stair_assignment),
        "fire-service rescue settings": spec.rescue_start is not None
        or spec.rescue_teams is not None,
        "counter-flow": spec.counter_flow_probability is not None,
        "vacancy": spec.vacancy_rate is not None,
        "the share aged 80+": spec.share_80_plus_of_65_plus is not None,
        "capacity changes": bool(spec.capacity_multipliers),
        "fire and door settings": spec.hazard is not None
        and bool(spec.hazard.model_dump(exclude_defaults=True, exclude={"enabled"})),
    }
    missing = [k for k, v in unseen.items() if v]
    if missing:
        notes.append("Not varied in training: " + ", ".join(missing) + ".")
    return notes


class Surrogate:
    """The trained network with a small cache of building graphs."""

    def __init__(self, path: Path | None = None, params: Params | None = None) -> None:
        self.path = path or DEFAULT_WEIGHTS
        if not self.path.exists():
            raise FileNotFoundError(
                f"no surrogate weights at {self.path}; train them with `tailsafe surrogate train`"
            )
        self.net, self.stats, self.config, self.meta = load_model(self.path)
        self.params = params or get_params()
        self._graphs: OrderedDict[str, StaticGraph] = OrderedDict()

    def _static(self, building: Building) -> StaticGraph:
        key = building.digest()
        if key in self._graphs:
            self._graphs.move_to_end(key)
            return self._graphs[key]
        g = static_graph(building, self.params)
        self._graphs[key] = g
        while len(self._graphs) > 8:
            self._graphs.popitem(last=False)
        return g

    def predict(self, building: Building, spec: ScenarioSpec, top_edges: int = 8) -> dict[str, Any]:
        """Quantiles (seconds) of each loss, CVaR95, and the most congested edges.

        Values are capped at the simulation horizon (``horizon_s``); a value at
        the cap means "not within the horizon".
        """
        t0 = time.perf_counter()
        g = graph_features(self._static(building), spec, self.params)
        q, e = predict(self.net, self.stats, [g])
        elapsed = time.perf_counter() - t0
        out: dict[str, Any] = {"losses": {}, "elapsed_ms": elapsed * 1000.0, "horizon_s": HORIZON}
        for li, name in enumerate(LOSS_NAMES):
            vals = np.minimum(q[0, li], HORIZON)
            out["losses"][name] = {
                **{f"p{int(qq * 100)}": float(vals[k]) for k, qq in enumerate(QUANTILES)},
                "cvar95": float(vals[-1]),
            }
        per = e[0]
        fwd = [i for i, f in enumerate(g.edge_forward) if f]
        ranked = sorted(fwd, key=lambda i: -per[i])[:top_edges]
        out["edges"] = [
            {
                "edge": g.edge_ids[i],
                "label": building.describe_edge(g.edge_ids[i]),
                "person_minutes": float(per[i]) / 60.0,
            }
            for i in ranked
        ]
        out["coverage_notes"] = coverage_notes(building, spec)
        out["model"] = {
            "trained_on": self.meta.get("trained_on"),
            "evaluation": self.meta.get("evaluation"),
        }
        return out
