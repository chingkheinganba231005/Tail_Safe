"""RSET versus ASET for one simulated evacuation.

* **Per floor.** RSET_floor is when the last household left that floor (entered
  a stair or a lift, or was carried out); ASET_floor is the earliest ASET of the
  floor's circulation spaces (corridors, lobbies, lift lobbies, protected
  lobbies). The floor fails if RSET_floor > ASET_floor.
* **Per occupant.** An occupant fails if their accumulated FED reaches the
  susceptible-population limit (0.3 by default) before they get out.
* **Building.** The scenario fails (``RSET > ASET``) if any floor or any
  occupant fails.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tailsafe.building.model import NodeType
from tailsafe.config import Params, get_params
from tailsafe.hazard.model import HazardResult
from tailsafe.sim.meso import MesoResult

CIRCULATION = frozenset(
    {NodeType.CORRIDOR, NodeType.LOBBY, NodeType.LIFT_LOBBY, NodeType.PROTECTED_LOBBY}
)


@dataclass(frozen=True)
class RsetAset:
    """Tenability outcome of one run."""

    floor_levels: NDArray[np.int16]
    floor_rset: NDArray[np.float32]
    floor_aset: NDArray[np.float32]
    occupants_over_fed_limit: int
    occupants_incapacitated: int
    max_fed: float

    @property
    def floor_fails(self) -> NDArray[np.bool_]:
        """Floors where RSET exceeds ASET."""
        out: NDArray[np.bool_] = self.floor_rset > self.floor_aset
        return out

    @property
    def fails(self) -> bool:
        """RSET > ASET anywhere (a floor, or any occupant over the FED limit)."""
        return bool(self.floor_fails.any() or self.occupants_over_fed_limit > 0)


def floor_aset(hazard: HazardResult, res: MesoResult) -> dict[int, float]:
    """ASET of each level: earliest ASET of its circulation spaces."""
    net = res.net
    out: dict[int, float] = {}
    for i, t in enumerate(net.node_type):
        if t in CIRCULATION:
            lv = int(net.node_level[i])
            out[lv] = min(out.get(lv, np.inf), float(hazard.aset[i]))
    return out


def rset_aset(res: MesoResult, params: Params | None = None) -> RsetAset:
    """Compare evacuation (RSET) with tenability (ASET) for a finished run."""
    p = params or get_params()
    limit = p.scalar("hazard.tenability.fed_limit_susceptible")
    clear = res.floor_clearance()
    levels = np.array(sorted(clear), dtype=np.int16)
    rset = np.array([clear[int(lv)] for lv in levels], dtype=np.float32)
    hazard = res.scenario.hazard
    if isinstance(hazard, HazardResult):
        fa = floor_aset(hazard, res)
        aset = np.array([fa.get(int(lv), np.inf) for lv in levels], dtype=np.float32)
    else:
        aset = np.full(levels.size, np.inf, dtype=np.float32)
    sizes = res.population.group_size
    fed = res.group_fed
    return RsetAset(
        floor_levels=levels,
        floor_rset=rset,
        floor_aset=aset,
        occupants_over_fed_limit=int(sizes[fed >= limit].sum()),
        occupants_incapacitated=res.n_incapacitated,
        max_fed=float(fed.max()) if fed.size else 0.0,
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
