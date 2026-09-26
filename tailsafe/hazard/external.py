"""Import hazard time series from an external tool (e.g. FDS, CONTAM).

TailSafe's own smoke model is a coarse approximation. When CFD or network-model
output is available, save it per building node and load it here; the result
plugs into the simulator exactly like :class:`~tailsafe.hazard.model.HazardResult`.

File format (``.npz``)
----------------------
``times``        float [H]      seconds since ignition, increasing, regular spacing
``node_ids``     str   [N]      building node ids the columns refer to
``visibility``   float [H, N]   metres (optional)
``temperature``  float [H, N]   °C (optional)
``co_ppm``       float [H, N]   ppm (optional)
``co2_percent``  float [H, N]   % (optional)

Nodes not listed are treated as clean air. At least one field must be present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tailsafe.building.model import Building
from tailsafe.config import Params
from tailsafe.hazard.model import FireSpec, HazardResult
from tailsafe.hazard.tenability import Tenability

FIELDS = ("visibility", "temperature", "co_ppm", "co2_percent")


def hazard_from_arrays(
    building: Building,
    times: NDArray[np.float64],
    node_ids: list[str],
    fields: dict[str, NDArray[np.float64]],
    *,
    params: Params | None = None,
    fire_node: str | None = None,
) -> HazardResult:
    """Build a :class:`HazardResult` from externally computed node time series."""
    if not any(k in fields for k in FIELDS):
        raise ValueError(f"need at least one of {FIELDS}")
    unknown = set(fields) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown fields {sorted(unknown)}")
    t = np.asarray(times, dtype=np.float64)
    steps = np.diff(t)
    if t.size < 2 or (steps <= 0).any() or not np.allclose(steps, steps[0], rtol=1e-6):
        raise ValueError("times must be regularly spaced and increasing")
    ten = Tenability.from_params(params)
    all_ids = [n.id for n in building.nodes]
    col = {nid: i for i, nid in enumerate(all_ids)}
    missing = [n for n in node_ids if n not in col]
    if missing:
        raise ValueError(f"node ids not in the building: {missing[:5]}")
    H, N = t.size, len(all_ids)
    cols = np.array([col[n] for n in node_ids], dtype=np.intp)

    def full(name: str, clean: float) -> NDArray[np.float64]:
        out = np.full((H, N), clean)
        if name in fields:
            arr = np.asarray(fields[name], dtype=np.float64)
            if arr.shape != (H, len(node_ids)):
                raise ValueError(f"{name} must have shape {(H, len(node_ids))}")
            out[:, cols] = arr
        return out

    vis = full("visibility", np.inf)
    temp = full("temperature", ten.t_amb)
    co = full("co_ppm", 0.0)
    co2 = full("co2_percent", 0.0)
    with np.errstate(divide="ignore"):
        ext = np.where(np.isfinite(vis), ten.c_vis / np.maximum(vis, 1e-6), 0.0)
    speed = ten.speed_multiplier(ext)
    fed = ten.fed_rate(co, co2, temp)
    dt = float(steps[0])
    bad = ten.untenable(vis, temp) | (np.cumsum(fed * dt, axis=0) >= ten.fed_limit)
    first = np.where(bad.any(axis=0), bad.argmax(axis=0), -1)
    aset = np.where(first >= 0, t[np.maximum(first, 0)], np.inf)
    return HazardResult(
        dt=dt,
        times=t - t[0],
        speed_multiplier=speed.astype(np.float32),
        fed_rate=fed.astype(np.float32),
        aset=aset,
        fire=FireSpec(node=fire_node or node_ids[0], growth=0.0, peak=0.0),
        hrr=np.zeros(H),
        node_ids=all_ids,
        visibility=np.minimum(vis, 1e3).astype(np.float32),
        temperature=temp.astype(np.float32),
        co_ppm=co.astype(np.float32),
        info={"source": 1.0},
    )


def load_hazard_npz(
    building: Building, path: Path, *, params: Params | None = None
) -> HazardResult:
    """Load an external hazard file (format in the module docstring)."""
    with np.load(path, allow_pickle=False) as data:
        fields = {k: data[k] for k in FIELDS if k in data.files}
        return hazard_from_arrays(
            building,
            data["times"],
            [str(x) for x in data["node_ids"]],
            fields,
            params=params,
        )
