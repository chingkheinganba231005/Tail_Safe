from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tailsafe.building.model import Building, EdgeKind
from tailsafe.building.templates import generate
from tailsafe.hazard import _kernel as hk
from tailsafe.hazard.external import hazard_from_arrays, load_hazard_npz
from tailsafe.hazard.model import FireSpec, HazardModel
from tailsafe.hazard.tenability import Tenability
from tailsafe.hazard.zones import build_zones


@pytest.fixture(scope="module")
def tower() -> Building:
    return generate("cruciform", storeys=16, flats_per_wing=2)


@pytest.fixture(scope="module")
def model(tower: Building) -> HazardModel:
    return HazardModel(tower)


FIRE = dict(node="L08.unit.N1", growth=0.0469, peak=3000.0)


def test_hrr_curve() -> None:
    f = FireSpec(node="x", growth=0.01, peak=100.0)
    np.testing.assert_allclose(f.hrr(np.array([0.0, 50.0, 1000.0])), [0.0, 25.0, 100.0])


def test_integrator_conserves_mass() -> None:
    # Two closed zones exchanging, no sinks or vents: everything injected stays.
    vol = np.array([10.0, 30.0])
    out_m = np.zeros((11, 2))
    out_e = np.zeros((11, 2))
    q = np.zeros(100)
    q[:10] = 0.2  # a short puff, then let it mix
    hk.integrate_zones(
        vol,
        np.zeros(2, dtype=bool),
        np.array([0, 1], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
        np.array([1.0, 1.0]),
        np.zeros(2),
        0.0,
        0,
        q,
        q,
        1.0,
        10,
        out_m,
        out_e,
    )
    assert out_m[-1].sum() == pytest.approx(0.2 * 10)
    # Well mixed at the end: equal concentrations.
    c = out_m[-1] / vol
    assert c[0] == pytest.approx(c[1], rel=0.01)


def test_zone_network_structure(tower: Building) -> None:
    z = build_zones(tower)
    assert z.n_zones == len(tower.nodes)
    assert not z.is_sink[z.ex_src].any(), "exits never send smoke back"
    assert z.stable_dt() > 1.0
    # Stack effect: every stair exchange upwards is larger than downwards.
    stairs = [k for k, e in enumerate(tower.edges) if e.kind == EdgeKind.STAIR]
    idx = {n.id: i for i, n in enumerate(tower.nodes)}
    for k in stairs[:5]:
        e = tower.edges[k]
        up = z.ex_rate[(z.ex_edge == k) & (z.ex_src == idx[e.target])]
        down = z.ex_rate[(z.ex_edge == k) & (z.ex_src == idx[e.source])]
        assert up[0] > down[0]


def test_open_fire_door_brings_smoke_into_corridor(model: HazardModel) -> None:
    closed = model.run(FireSpec(**FIRE, door_open=False), horizon=1800)
    opened = model.run(FireSpec(**FIRE, door_open=True), horizon=1800)
    assert closed.aset_of("L08.unit.N1") < 120
    assert opened.aset_of("L08.cor.N1") < closed.aset_of("L08.cor.N1")
    assert np.isfinite(opened.aset_of("L08.cor.N1"))
    # Smoke rises: the floor above is worse than the floor below.
    t = -1
    i_up = opened.node_ids.index("L09.lobby")
    i_dn = opened.node_ids.index("L07.lobby")
    assert opened.fed_rate[t, i_up] > opened.fed_rate[t, i_dn]
    # Exits stay clean.
    ex = [opened.node_ids.index(n.id) for n in model.building.exits]
    assert (opened.fed_rate[:, ex] == 0).all() and (opened.speed_multiplier[:, ex] == 1).all()


def test_holding_stair_doors_open_lets_smoke_into_stairs(tower: Building) -> None:
    stair_doors = [
        e.id
        for e in tower.edges
        if e.kind == EdgeKind.DOOR and ".stair." in e.target and "exit" not in e.target
    ]
    shut = HazardModel(tower)
    held = HazardModel(tower, held_open=stair_doors)
    fire = FireSpec(**FIRE, door_open=True)
    a = shut.run(fire, horizon=1800)
    b = held.run(fire, horizon=1800)
    i = a.node_ids.index("L10.stair.A")
    assert b.fed_rate[-1, i] > a.fed_rate[-1, i]


def test_fields_match_reference_conversions(model: HazardModel) -> None:
    hz = model.run(FireSpec(**FIRE, door_open=True), horizon=900, keep_fields=True)
    ten = model.tenability
    assert hz.visibility is not None and hz.temperature is not None and hz.co_ppm is not None
    i = hz.node_ids.index("L08.cor.N1")
    vis = float(hz.visibility[-1, i])
    ext = ten.c_vis / vis
    expected_speed = max(ten.vmin_frac, 1 + ten.beta / ten.alpha * ext)
    assert hz.speed_multiplier[-1, i] == pytest.approx(expected_speed, rel=1e-3)
    assert (hz.speed_multiplier >= ten.vmin_frac - 1e-6).all()
    assert (hz.speed_multiplier <= 1.0).all()
    assert (np.diff(hz.hrr) >= 0).all()


def test_tenability_formulas() -> None:
    ten = Tenability.from_params()
    ext = np.array([0.0, 1.0, 100.0])
    np.testing.assert_allclose(ten.visibility(ext)[1:], ten.c_vis / ext[1:])
    assert ten.speed_multiplier(ext)[0] == 1.0
    assert ten.speed_multiplier(ext)[2] == pytest.approx(ten.vmin_frac)
    # ISO 13571 simplified: 35,000 ppm·min of CO gives FED 1 (no CO2, ambient).
    rate = ten.fed_rate(np.array([1000.0]), np.array([0.0]), np.array([ten.t_amb]))
    assert rate[0] * 35 * 60 == pytest.approx(1.0)
    # Ambient air contributes no heat dose.
    assert ten.fed_rate(np.zeros(1), np.zeros(1), np.array([ten.t_amb]))[0] == 0.0
    assert ten.fed_rate(np.zeros(1), np.zeros(1), np.array([120.0]))[0] > 0


def test_external_fields(tmp_path: Path, tower: Building) -> None:
    times = np.arange(0, 610, 10.0)
    ids = ["L08.cor.N1", "L08.lobby"]
    vis = np.full((times.size, 2), 30.0)
    vis[30:, 0] = 2.0  # corridor smoke-logged from 300 s
    hz = hazard_from_arrays(tower, times, ids, {"visibility": vis})
    assert hz.aset_of("L08.cor.N1") == 300.0
    assert np.isinf(hz.aset_of("L08.lobby"))
    path = tmp_path / "cfd.npz"
    np.savez(path, times=times, node_ids=np.array(ids), visibility=vis)
    again = load_hazard_npz(tower, path)
    np.testing.assert_array_equal(again.aset, hz.aset)
    with pytest.raises(ValueError):
        hazard_from_arrays(tower, times, ["nope"], {"visibility": vis[:, :1]})
    with pytest.raises(ValueError):
        hazard_from_arrays(tower, times, ids, {"smell": vis})
