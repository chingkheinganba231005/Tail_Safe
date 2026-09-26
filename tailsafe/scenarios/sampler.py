"""Draw concrete evacuation scenarios from a :class:`ScenarioSpec`.

Scenario-level random variables (rescue start, blockage times, which lifts are
out, ...) use a fixed 16-slot vector of uniforms per scenario. Slots never move,
whatever the spec, so a baseline and an intervention evaluated on scenario ``i``
see the same draws (common random numbers). With ``lhs=True`` the vectors of
each batch form a Latin Hypercube design; otherwise they are i.i.d. from the
scenario's own stream. Occupant-level randomness comes from the population
sampler's streams for ``(seed, index)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

from tailsafe.building.model import Building, LevelKind
from tailsafe.config import Params, get_params
from tailsafe.population.synth import Population, PopulationConfig, sample_population
from tailsafe.rng import stream, stream_id
from tailsafe.scenarios.spec import ScenarioSpec
from tailsafe.sim.meso import Blockage, SimScenario, stair_blockage

N_DIMS = 16
DIM_RESCUE = 0
DIM_FIRE_LEVEL = 1
DIM_RANDOM_BLOCK = 2  # occurrence, which stair, time: slots 2, 3, 4
DIM_LIFTS_OUT = 5  # up to 3 lifts: slots 5, 6, 7
DIM_BLOCK_TIMES = 8  # up to 4 named blockages: slots 8..11
DIM_HAZARD = 12  # reserved for the hazard model: slots 12..15
MAX_LIFTS_OUT = 3
MAX_NAMED_BLOCKAGES = 4


def scenario_uniforms(
    seed: int, start: int, n: int, *, lhs: bool = True, batch_size: int = 100
) -> NDArray[np.float64]:
    """Uniform vectors for scenarios ``start .. start + n - 1`` (shape ``[n, N_DIMS]``).

    With ``lhs`` each block of ``batch_size`` consecutive indices is one Latin
    Hypercube design, so rows depend only on ``(seed, index, batch_size)``.
    """
    out = np.empty((n, N_DIMS))
    if not lhs:
        for k in range(n):
            out[k] = stream(seed, "scenario", start + k).random(N_DIMS)
        return out
    designs: dict[int, NDArray[np.float64]] = {}
    for k in range(n):
        i = start + k
        b, r = divmod(i, batch_size)
        if b not in designs:
            ss = np.random.SeedSequence(entropy=seed, spawn_key=(b, stream_id("lhs")))
            sampler = qmc.LatinHypercube(d=N_DIMS, seed=np.random.default_rng(ss))
            designs[b] = sampler.random(batch_size)
        out[k] = designs[b][r]
    return out


@dataclass
class SampledScenario:
    """One concrete scenario: who is where, and what goes wrong when."""

    index: int
    population: Population
    sim: SimScenario
    info: dict[str, Any] = field(default_factory=dict)


class ScenarioSampler:
    """Turns ``(seed, index, uniforms)`` into a :class:`SampledScenario`."""

    def __init__(
        self, building: Building, spec: ScenarioSpec, params: Params | None = None
    ) -> None:
        self.building = building
        self.spec = spec
        self.params = params or get_params()
        stairs = {s.id for s in building.stairs}
        for blk in spec.stair_blockages:
            if blk.stair not in stairs:
                raise ValueError(f"spec blocks unknown stair {blk.stair!r}")
        for lv, sid in spec.stair_assignment.items():
            if sid not in stairs:
                raise ValueError(f"level {lv} assigned to unknown stair {sid!r}")
        if len(spec.stair_blockages) > MAX_NAMED_BLOCKAGES:
            raise ValueError(f"at most {MAX_NAMED_BLOCKAGES} named stair blockages")
        if spec.lifts_out_of_service > MAX_LIFTS_OUT:
            raise ValueError(f"at most {MAX_LIFTS_OUT} lifts out of service")
        self._stairs = [s.id for s in building.stairs]
        self._lifts = [lf.id for lf in building.lifts]
        self._evac_candidates = [lf.id for lf in building.lifts if not lf.firefighting]
        self._occupied_levels = sorted(
            {n.level for n in building.units}
            - {lv.index for lv in building.levels if lv.kind == LevelKind.REFUGE}
        )

    def sample(self, seed: int, index: int, u: NDArray[np.float64]) -> SampledScenario:
        """Draw scenario ``index`` using the scenario-level uniforms ``u``."""
        spec = self.spec
        p = self.params
        info: dict[str, Any] = {}

        rescue = spec.rescue_start.as_param() if spec.rescue_start else p["rescue.operations_start"]
        rescue_start = float(np.asarray(rescue.ppf(np.array([u[DIM_RESCUE]])))[0])
        info["rescue_start"] = rescue_start

        if spec.fire_level is not None:
            fire_level = spec.fire_level
        elif self._occupied_levels:
            k = min(
                int(u[DIM_FIRE_LEVEL] * len(self._occupied_levels)), len(self._occupied_levels) - 1
            )
            fire_level = self._occupied_levels[k]
        else:
            fire_level = 0
        info["fire_level"] = fire_level

        blockages: list[Blockage] = []
        blocked: dict[str, float] = {}
        for k, blk in enumerate(spec.stair_blockages):
            t = float(blk.time.ppf(np.array([u[DIM_BLOCK_TIMES + k]]))[0])
            blocked[blk.stair] = min(t, blocked.get(blk.stair, np.inf))
        rsb = spec.random_stair_blockage
        if rsb is not None and self._stairs and u[DIM_RANDOM_BLOCK] < rsb.probability:
            k = min(int(u[DIM_RANDOM_BLOCK + 1] * len(self._stairs)), len(self._stairs) - 1)
            t = float(rsb.time.ppf(np.array([u[DIM_RANDOM_BLOCK + 2]]))[0])
            sid = self._stairs[k]
            blocked[sid] = min(t, blocked.get(sid, np.inf))
        for sid, t in sorted(blocked.items()):
            blockages.extend(stair_blockage(self.building, sid, t))
        info["blocked_stairs"] = {sid: t for sid, t in sorted(blocked.items())}

        remaining = list(self._lifts)
        out: list[str] = []
        for j in range(min(spec.lifts_out_of_service, len(remaining))):
            k = min(int(u[DIM_LIFTS_OUT + j] * len(remaining)), len(remaining) - 1)
            out.append(remaining.pop(k))
        info["lifts_out"] = out
        evac = (
            tuple(lid for lid in self._evac_candidates if lid not in out)
            if spec.evacuation_lifts
            else ()
        )
        info["evacuation_lifts"] = list(evac)

        pop = sample_population(
            self.building,
            PopulationConfig(
                time_slot=spec.time_slot,
                share_65_plus=spec.share_65_plus,
                share_80_plus_of_65_plus=spec.share_80_plus_of_65_plus,
                evacuation_lifts=bool(evac),
                counter_flow_probability=spec.counter_flow_probability,
                vacancy_rate=spec.vacancy_rate,
            ),
            seed=seed,
            index=index,
            params=p,
        )
        sim = SimScenario(
            blockages=tuple(blockages),
            evacuation_lifts=evac,
            lift_priority=spec.lift_priority,
            phased_release=dict(spec.phased_release),
            stair_assignment=dict(spec.stair_assignment),
            rescue_start=rescue_start,
            rescue_teams=spec.rescue_teams,
        )
        return SampledScenario(index=index, population=pop, sim=sim, info=info)
