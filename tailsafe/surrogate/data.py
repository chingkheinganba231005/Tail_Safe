"""Training data for the surrogate: simulated (building, scenario) cases.

Each case is a procedurally generated building (template and options drawn
at random) with a random scenario specification. The simulator runs a short
Monte Carlo (``n_runs`` scenarios, Latin Hypercube) and we keep the sample of
each loss and the mean queueing (person-seconds) on every edge. Buildings are
stored as template + options (they are deterministic), so the data set stays
small; features are computed from them when training.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tailsafe.building.model import Building
from tailsafe.building.templates import generate
from tailsafe.config import get_params
from tailsafe.rng import stream
from tailsafe.scenarios.montecarlo import LOSSES, MCConfig, run_monte_carlo
from tailsafe.scenarios.spec import Dist, HazardSpec, ScenarioSpec, StairBlockage

# Option ranges per typology (inclusive integer ranges).
TYPOLOGIES: dict[str, dict[str, tuple[int, int]]] = {
    "cruciform": {"storeys": (8, 40), "flats_per_wing": (2, 5), "lifts": (2, 3)},
    "slab": {"storeys": (6, 30), "flats_per_side": (4, 10), "lifts": (1, 2)},
    "twin_core": {
        "storeys": (10, 40),
        "podium_levels": (1, 3),
        "flats_per_floor": (4, 8),
        "lifts": (2, 3),
    },
    "care_home": {"storeys": (2, 6), "rooms_per_side": (4, 8), "lifts": (1, 2)},
}
TIME_SLOTS = ("weekday_day", "weekday_night", "weekend_day", "weekend_night")
SHARE_65_RANGE = (0.05, 0.45)


@dataclass(frozen=True)
class Case:
    """One training case: a building (template + options) and a scenario."""

    index: int
    template: str
    options: dict[str, int]
    spec: ScenarioSpec

    def building(self) -> Building:
        """Generate the building (deterministic)."""
        return generate(self.template, **self.options)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "index": self.index,
            "template": self.template,
            "options": self.options,
            "spec": self.spec.model_dump(mode="json"),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Case:
        """Inverse of :meth:`as_dict`."""
        return cls(
            index=int(d["index"]),
            template=d["template"],
            options={k: int(v) for k, v in d["options"].items()},
            spec=ScenarioSpec.model_validate(d["spec"]),
        )


def sample_case(index: int, seed: int = 0, typologies: tuple[str, ...] | None = None) -> Case:
    """Draw a building and a scenario for case ``index`` (deterministic)."""
    rng = stream(seed, "surrogate-case", index)
    names = typologies or tuple(TYPOLOGIES)
    template = names[int(rng.integers(len(names)))]
    options = {k: int(rng.integers(lo, hi + 1)) for k, (lo, hi) in TYPOLOGIES[template].items()}
    if template == "twin_core":  # the twin core needs an even number of flats per floor
        options["flats_per_floor"] = 2 * (options["flats_per_floor"] // 2)
    b = generate(template, **options)
    stairs = [s.id for s in b.stairs]
    upper = [lv.index for lv in b.levels if lv.index > 0]
    n_lifts = len(b.lifts)

    blockages = []
    if stairs and rng.random() < 0.5:
        sid = stairs[int(rng.integers(len(stairs)))]
        blockages.append(StairBlockage(stair=sid, time=Dist.fixed(float(rng.uniform(60, 900)))))
    evac = bool(rng.random() < 0.3) and any(not lf.firefighting for lf in b.lifts)
    wardens: list[int] = []
    if upper and rng.random() < 0.25:
        k = int(rng.integers(1, 3))
        wardens = sorted(
            {int(x) for x in rng.choice(upper, size=min(k, len(upper)), replace=False)}
        )
    spec = ScenarioSpec(
        name=f"surrogate-{index}",
        time_slot=TIME_SLOTS[int(rng.integers(4))],
        share_65_plus=None if template == "care_home" else float(rng.uniform(*SHARE_65_RANGE)),
        fire_level=int(upper[int(rng.integers(len(upper)))]) if upper else None,
        stair_blockages=blockages,
        lifts_out_of_service=int(rng.integers(0, 2)) if n_lifts > 1 else 0,
        evacuation_lifts=evac,
        lift_priority=("top_down", "nearest", "bottom_up")[int(rng.integers(3))],
        lift_eligibility=("mobility_impaired", "wheelchair_users")[int(rng.integers(2))],
        hazard=HazardSpec() if rng.random() < 0.6 else None,
        warden_levels=wardens,
    )
    return Case(index=index, template=template, options=options, spec=spec)


def run_case(case: Case, n_runs: int = 64, seed: int = 0) -> dict[str, Any]:
    """Simulate a case: loss samples and mean queueing per edge."""
    b = case.building()
    t0 = time.perf_counter()
    res = run_monte_carlo(
        b,
        case.spec,
        MCConfig(n_runs=n_runs, seed=seed, workers=1, batch_size=n_runs),
        params=get_params(),
    )
    wall = time.perf_counter() - t0
    qint = res.arc_qint.mean(axis=0)
    by_edge: dict[str, float] = {}
    for a, eid in enumerate(res.arc_edge_ids):
        by_edge[eid] = by_edge.get(eid, 0.0) + float(qint[a])
    return {
        **case.as_dict(),
        "n_runs": n_runs,
        "seed": seed,
        "wall_time_s": wall,
        "losses": {name: [float(x) for x in res.loss(name)] for name in LOSSES},
        "edge_queue": {k: round(v, 2) for k, v in by_edge.items() if v >= 0.01},
    }


def _work(args: tuple[dict[str, Any], int, int]) -> dict[str, Any]:
    case, n_runs, seed = args
    return run_case(Case.from_dict(case), n_runs, seed)


def generate_dataset(
    n_cases: int,
    *,
    n_runs: int = 64,
    seed: int = 0,
    start: int = 0,
    workers: int | None = None,
    typologies: tuple[str, ...] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Simulate cases ``start .. start + n_cases - 1`` in parallel (one process per case)."""
    import os

    cases = [sample_case(i, seed, typologies).as_dict() for i in range(start, start + n_cases)]
    out: list[dict[str, Any]] = []
    n_workers = max(1, workers or (os.cpu_count() or 1))
    if n_workers == 1:
        for k, c in enumerate(cases):
            out.append(_work((c, n_runs, seed)))
            if progress:
                progress(k + 1, n_cases)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_work, (c, n_runs, seed)) for c in cases]
            for k, fut in enumerate(as_completed(futures)):
                out.append(fut.result())
                if progress:
                    progress(k + 1, n_cases)
    out.sort(key=lambda r: int(r["index"]))
    return out


def save_dataset(records: list[dict[str, Any]], path: Path) -> Path:
    """Write records as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return path


def load_dataset(path: Path) -> list[dict[str, Any]]:
    """Read JSON-lines records."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def loss_quantiles(values: list[float], qs: tuple[float, ...]) -> np.ndarray:
    """Empirical quantiles (linear interpolation) of a loss sample."""
    return np.quantile(np.asarray(values, dtype=np.float64), qs)
