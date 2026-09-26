"""Numba integrator of the zone smoke network (explicit, conservative)."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def integrate_zones(  # type: ignore[no-untyped-def]
    volume,
    is_sink,
    ex_src,
    ex_dst,
    ex_rate,
    vent_rate,
    heat_loss,
    source,
    q_mass,
    q_heat,
    dt,
    rec_every,
    out_mass,
    out_heat,
):
    """Advance fuel-equivalent product mass and smoke heat in every zone.

    ``q_mass[k]`` (kg/s) and ``q_heat[k]`` (kW) are injected into zone ``source``
    during step ``k``. Every ``rec_every`` steps the zone contents are written to
    ``out_mass`` / ``out_heat`` (rows = records). Mass leaving through sinks
    (exits) and vents is lost to the outside; nothing else is created or lost.
    """
    N = volume.size
    X = ex_src.size
    n_steps = q_mass.size
    m = np.zeros(N)
    e = np.zeros(N)
    dm = np.zeros(N)
    de = np.zeros(N)
    coef = np.empty(X)
    for x in range(X):
        coef[x] = ex_rate[x] * dt / volume[ex_src[x]]
    keep_m = np.empty(N)
    keep_e = np.empty(N)
    for i in range(N):
        keep_m[i] = 1.0 - vent_rate[i] * dt
        keep_e[i] = 1.0 - (vent_rate[i] + heat_loss) * dt
    rec = 0
    for k in range(n_steps):
        if k % rec_every == 0 and rec < out_mass.shape[0]:
            for i in range(N):
                out_mass[rec, i] = m[i]
                out_heat[rec, i] = e[i]
            rec += 1
        for i in range(N):
            dm[i] = 0.0
            de[i] = 0.0
        for x in range(X):
            i = ex_src[x]
            j = ex_dst[x]
            fm = coef[x] * m[i]
            fe = coef[x] * e[i]
            dm[i] -= fm
            de[i] -= fe
            if not is_sink[j]:
                dm[j] += fm
                de[j] += fe
        dm[source] += q_mass[k] * dt
        de[source] += q_heat[k] * dt
        for i in range(N):
            m[i] = m[i] * keep_m[i] + dm[i]
            e[i] = e[i] * keep_e[i] + de[i]
            if m[i] < 0.0:
                m[i] = 0.0
            if e[i] < 0.0:
                e[i] = 0.0
    while rec < out_mass.shape[0]:
        for i in range(N):
            out_mass[rec, i] = m[i]
            out_heat[rec, i] = e[i]
        rec += 1


@njit(cache=True)
def derive_fields(  # type: ignore[no-untyped-def]
    mass,
    heat,
    volume,
    is_sink,
    k_ext,
    ext_limit,
    speed_slope,
    vmin_frac,
    k_co,
    k_co2,
    co_dose_s,
    co2_scale,
    rho_cp,
    max_rise,
    t_amb,
    t_limit,
    heat_c_s,
    fed_limit,
    record_dt,
    out_speed,
    out_fed,
    out_aset,
):
    """Speed multiplier, FED rate and ASET from recorded zone contents (one pass).

    See :mod:`tailsafe.hazard.tenability` for the formulas; this fused version
    avoids large temporary arrays in Monte Carlo runs.
    """
    H, N = mass.shape
    base_heat = 1.0 / (heat_c_s * max(t_amb, 1.0) ** (-3.4))
    cum = np.zeros(N)
    for i in range(N):
        out_aset[i] = np.inf
    for r in range(H):
        for i in range(N):
            if is_sink[i] or mass[r, i] <= 1e-15:
                out_speed[r, i] = 1.0
                out_fed[r, i] = 0.0
                continue
            dens = mass[r, i] / volume[i]
            ext = k_ext * dens
            sp = 1.0 + speed_slope * ext
            if sp < vmin_frac:
                sp = vmin_frac
            if sp > 1.0:
                sp = 1.0
            out_speed[r, i] = sp
            rise = heat[r, i] / volume[i] / rho_cp
            if rise > max_rise:
                rise = max_rise
            fed = k_co * dens / co_dose_s * np.exp(k_co2 * dens / co2_scale)
            if rise > 0.01:
                temp = t_amb + rise
                h = 1.0 / (heat_c_s * temp ** (-3.4)) - base_heat
                if h > 0.0:
                    fed += h
            out_fed[r, i] = fed
            cum[i] += fed * record_dt
            if out_aset[i] == np.inf and (
                ext > ext_limit or t_amb + rise > t_limit or cum[i] >= fed_limit
            ):
                out_aset[i] = r * record_dt
