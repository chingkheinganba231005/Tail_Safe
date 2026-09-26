"""From smoke contents to visibility, walking speed, FED rates and tenability.

Conversions (all per zone, per time):

* soot density ``ρ_s = y_soot · m / V``; extinction ``K = K_m ρ_s`` (1/m);
  visibility ``S = C / K``;
* walking-speed multiplier in smoke ``max(v_min, 1 + (β/α) K)`` (Frantzich &
  Nilsson, as in FDS+Evac);
* CO (ppm) and CO₂ (%) from their yields; asphyxiant FED rate
  ``[CO] / (D · 60 s) · exp(%CO₂ / s)`` (ISO 13571 simplified form);
* convective-heat FED rate ``1 / (60 · c · T^-3.4)`` in excess of the ambient
  value, T in °C (ISO 13571).

O₂ depletion, HCN and radiant heat are ignored. All constants come from the
``hazard`` section of the parameter registry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tailsafe.config import Params, get_params

RHO_AIR = 1.2  # kg/m³
RHO_CP = 1.2  # kJ/(m³·K), air
M_AIR, M_CO, M_CO2 = 28.97, 28.01, 44.01

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]


@dataclass(frozen=True)
class Tenability:
    """Constants of the tenability conversions."""

    km: float
    c_vis: float
    alpha: float
    beta: float
    vmin_frac: float
    vis_limit: float
    t_limit: float
    t_amb: float
    co_dose: float
    co2_scale: float
    heat_c: float
    fed_limit: float
    soot_yield: float
    co_yield: float
    co2_yield: float
    max_rise: float

    @classmethod
    def from_params(cls, params: Params | None = None) -> Tenability:
        """Read constants from the registry."""
        p = params or get_params()
        t = "hazard.tenability."
        f = "hazard.fire."
        return cls(
            km=p.scalar(t + "mass_extinction_coefficient"),
            c_vis=p.scalar(t + "visibility_constant"),
            alpha=p.scalar(t + "smoke_speed_alpha"),
            beta=p.scalar(t + "smoke_speed_beta"),
            vmin_frac=p.scalar(t + "smoke_speed_min_fraction"),
            vis_limit=p.scalar(t + "visibility_limit"),
            t_limit=p.scalar(t + "temperature_limit"),
            t_amb=p.scalar(t + "ambient_temperature"),
            co_dose=p.scalar(t + "co_fed_dose"),
            co2_scale=p.scalar(t + "co2_hyperventilation_scale"),
            heat_c=p.scalar(t + "heat_fed_constant"),
            fed_limit=p.scalar(t + "fed_limit_susceptible"),
            soot_yield=p.scalar(f + "soot_yield"),
            co_yield=p.scalar(f + "co_yield"),
            co2_yield=p.scalar(f + "co2_yield"),
            max_rise=p.scalar(f + "fire_room_max_temperature_rise"),
        )

    def extinction(self, products_density: F64) -> F64:
        """Extinction coefficient K (1/m) from product density (kg fuel-equivalent / m³)."""
        out: F64 = self.km * self.soot_yield * products_density
        return out

    def visibility(self, extinction: F64) -> F64:
        """Visibility (m); infinite in clear air."""
        with np.errstate(divide="ignore"):
            out: F64 = np.where(extinction > 0, self.c_vis / extinction, np.inf)
        return out

    def speed_multiplier(self, extinction: F64) -> F64:
        """Walking-speed factor in smoke."""
        out: F64 = np.maximum(self.vmin_frac, 1.0 + (self.beta / self.alpha) * extinction)
        return np.minimum(out, 1.0)

    def co_ppm(self, products_density: F64) -> F64:
        """CO concentration (ppm by volume)."""
        out: F64 = self.co_yield * products_density / RHO_AIR * (M_AIR / M_CO) * 1e6
        return out

    def co2_percent(self, products_density: F64) -> F64:
        """CO₂ from combustion (% by volume, above ambient)."""
        out: F64 = self.co2_yield * products_density / RHO_AIR * (M_AIR / M_CO2) * 100.0
        return out

    def temperature(self, heat_density: F64) -> F64:
        """Gas temperature (°C) from smoke heat content (kJ/m³), capped."""
        rise = np.minimum(heat_density / RHO_CP, self.max_rise)
        out: F64 = self.t_amb + rise
        return out

    def _heat_rate(self, temp: F64) -> F64:
        t = np.maximum(temp, 1.0)
        out: F64 = 1.0 / (60.0 * self.heat_c * t ** (-3.4))
        return out

    def fed_rate(self, co_ppm: F64, co2_pct: F64, temp: F64) -> F64:
        """FED accumulated per second of exposure (asphyxiant + convective heat)."""
        asphyx = co_ppm / (self.co_dose * 60.0) * np.exp(co2_pct / self.co2_scale)
        heat = np.maximum(self._heat_rate(temp) - self._heat_rate(np.full(1, self.t_amb)), 0.0)
        out: F64 = asphyx + heat
        return out

    def untenable(self, visibility: F64, temp: F64) -> NDArray[np.bool_]:
        """Where visibility or temperature breaches its limit."""
        out: NDArray[np.bool_] = (visibility < self.vis_limit) | (temp > self.t_limit)
        return out
