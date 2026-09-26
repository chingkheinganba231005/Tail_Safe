"""Hazard model driver: fire → smoke spread → fields for the evacuation simulator.

A simplified engineering approximation (see ``docs/assumptions.md``): a
t-squared fire in one zone, a linear multi-zone transport network, and ISO 13571
style tenability. It produces, on a regular time grid,

* a walking-speed multiplier and an FED rate per node (consumed by the meso
  kernel through :class:`tailsafe.sim.meso.HazardField`);
* per-node **ASET**: the first time visibility or temperature breaches its
  limit, or a person standing there from the alarm would reach the FED limit.

Fields from a CFD tool can be used instead through :mod:`tailsafe.hazard.external`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tailsafe.building.model import Building, EdgeKind
from tailsafe.config import Params, get_params
from tailsafe.hazard import _kernel as hazard_kernel
from tailsafe.hazard.tenability import M_AIR, M_CO, M_CO2, RHO_AIR, RHO_CP, Tenability
from tailsafe.hazard.zones import ZoneNetwork, build_zones, open_door_rate


@dataclass(frozen=True)
class FireSpec:
    """One fire: where it is, how fast it grows, and whether its door stays open."""

    node: str
    growth: float  # kW/s²
    peak: float  # kW
    door_open: bool = False

    def hrr(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """Heat release rate (kW): t-squared growth capped at the peak."""
        out: NDArray[np.float64] = np.minimum(self.growth * np.maximum(t, 0.0) ** 2, self.peak)
        return out


@dataclass
class HazardResult:
    """Hazard fields on a regular time grid (rows = times, columns = nodes)."""

    dt: float
    times: NDArray[np.float64]
    speed_multiplier: NDArray[np.float32]
    fed_rate: NDArray[np.float32]
    aset: NDArray[np.float64]
    fire: FireSpec
    hrr: NDArray[np.float64]
    node_ids: list[str]
    visibility: NDArray[np.float32] | None = None
    temperature: NDArray[np.float32] | None = None
    co_ppm: NDArray[np.float32] | None = None
    info: dict[str, float] = field(default_factory=dict)

    def aset_of(self, node_id: str) -> float:
        """ASET of a node (s; ``inf`` if it stays tenable)."""
        return float(self.aset[self.node_ids.index(node_id)])


class HazardModel:
    """Runs fires in one building, reusing the zone network across scenarios."""

    def __init__(
        self,
        building: Building,
        params: Params | None = None,
        *,
        held_open: Iterable[str] = (),
    ) -> None:
        self.building = building
        self.params = params or get_params()
        self.base = build_zones(building, self.params, held_open=held_open)
        self.tenability = Tenability.from_params(self.params)
        self._index = {nid: i for i, nid in enumerate(self.base.node_ids)}
        f = "hazard.fire."
        self.heat_of_combustion = self.params.scalar(f + "heat_of_combustion")
        self.convective_fraction = self.params.scalar(f + "convective_fraction")
        self.heat_loss = self.params.scalar("hazard.transport.wall_heat_loss_rate")
        self.fire_vent = self.params.scalar("hazard.transport.fire_room_vent_rate")

    def zones_for(self, fire: FireSpec) -> ZoneNetwork:
        """Base network plus the fire room's vent and, if left open, its door."""
        z = self.base
        i = self._index[fire.node]
        vent = z.vent_rate.copy()
        vent[i] = max(vent[i], self.fire_vent)
        rate = z.ex_rate
        if fire.door_open:
            edges = self.building.edges
            door = next(
                (
                    k
                    for k, e in enumerate(edges)
                    if e.kind == EdgeKind.DOOR and fire.node in (e.source, e.target)
                ),
                None,
            )
            if door is not None:
                rate = rate.copy()
                rate[z.ex_edge == door] = open_door_rate(edges[door].width, self.params)
        return ZoneNetwork(
            node_ids=z.node_ids,
            volume=z.volume,
            is_sink=z.is_sink,
            ex_src=z.ex_src,
            ex_dst=z.ex_dst,
            ex_rate=rate,
            ex_edge=z.ex_edge,
            vent_rate=vent,
        )

    def run(
        self,
        fire: FireSpec,
        *,
        horizon: float = 7200.0,
        record_dt: float = 10.0,
        keep_fields: bool = False,
    ) -> HazardResult:
        """Simulate smoke spread for ``horizon`` seconds and derive the fields."""
        z = self.zones_for(fire)
        rec_every = max(1, int(np.ceil(record_dt / min(z.stable_dt(), record_dt))))
        dt = record_dt / rec_every
        n_steps = int(np.ceil(horizon / dt))
        t_mid = (np.arange(n_steps, dtype=np.float64) + 0.5) * dt
        q = fire.hrr(t_mid)
        n_rec = n_steps // rec_every + 1
        mass = np.zeros((n_rec, z.n_zones))
        heat = np.zeros((n_rec, z.n_zones))
        hazard_kernel.integrate_zones(
            z.volume,
            z.is_sink,
            z.ex_src,
            z.ex_dst,
            z.ex_rate,
            z.vent_rate,
            self.heat_loss,
            self._index[fire.node],
            q / self.heat_of_combustion,
            q * self.convective_fraction,
            dt,
            rec_every,
            mass,
            heat,
        )
        times = np.arange(n_rec, dtype=np.float64) * record_dt
        ten = self.tenability
        speed = np.empty((n_rec, z.n_zones), dtype=np.float32)
        fed = np.empty((n_rec, z.n_zones), dtype=np.float32)
        aset = np.empty(z.n_zones)
        hazard_kernel.derive_fields(
            mass,
            heat,
            z.volume,
            z.is_sink,
            ten.km * ten.soot_yield,
            ten.c_vis / ten.vis_limit,
            ten.beta / ten.alpha,
            ten.vmin_frac,
            ten.co_yield / RHO_AIR * (M_AIR / M_CO) * 1e6,
            ten.co2_yield / RHO_AIR * (M_AIR / M_CO2) * 100.0,
            ten.co_dose * 60.0,
            ten.co2_scale,
            RHO_CP,
            ten.max_rise,
            ten.t_amb,
            ten.t_limit,
            ten.heat_c * 60.0,
            ten.fed_limit,
            record_dt,
            speed,
            fed,
            aset,
        )
        vis = temp = co = None
        if keep_fields:
            dens = mass / z.volume
            vis = np.minimum(ten.visibility(ten.extinction(dens)), 1e3).astype(np.float32)
            temp = ten.temperature(heat / z.volume).astype(np.float32)
            co = ten.co_ppm(dens).astype(np.float32)
        return HazardResult(
            dt=record_dt,
            times=times,
            speed_multiplier=speed,
            fed_rate=fed,
            aset=aset,
            fire=fire,
            hrr=fire.hrr(times),
            node_ids=z.node_ids,
            visibility=vis,
            temperature=temp,
            co_ppm=co,
            info={
                "products_released_kg": float(np.sum(q / self.heat_of_combustion) * dt),
                "products_inside_kg": float(mass[-1].sum()),
                "heat_capacity_kj_per_k": float(RHO_CP * z.volume.sum()),
            },
        )
