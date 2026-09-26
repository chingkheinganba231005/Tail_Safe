from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tailsafe.building.model import Building
from tailsafe.building.templates import generate
from tailsafe.risk.breakdown import floor_band, tail_breakdown
from tailsafe.scenarios.montecarlo import LOSSES, MCConfig, MCResult, run_monte_carlo
from tailsafe.scenarios.spec import ScenarioSpec, reference_spec


@pytest.fixture(scope="module")
def small() -> Building:
    return generate("cruciform", storeys=16, flats_per_wing=2)


@pytest.fixture(scope="module")
def result(small: Building) -> MCResult:
    return run_monte_carlo(small, reference_spec(), MCConfig(n_runs=40, batch_size=20, workers=1))


def test_runs_and_losses(result: MCResult) -> None:
    assert result.n == 40
    assert [r.index for r in result.runs] == list(range(40))
    for name in LOSSES:
        x = result.loss(name)
        assert x.shape == (40,) and np.isfinite(x).all()
    assert (result.loss("total_time") >= result.loss("self_evacuation_time")).all()
    r = result.risk("total_time")
    assert r.mean.value <= r.cvar.value <= r.max
    with pytest.raises(ValueError):
        result.loss("average_vibes")


def test_parallel_equals_serial(small: Building, result: MCResult) -> None:
    par = run_monte_carlo(small, reference_spec(), MCConfig(n_runs=40, batch_size=20, workers=2))
    np.testing.assert_array_equal(par.loss("total_time"), result.loss("total_time"))
    np.testing.assert_array_equal(par.arc_qint, result.arc_qint)


def test_save_load_roundtrip(tmp_path: Path, result: MCResult) -> None:
    result.save(tmp_path / "r")
    again = MCResult.load(tmp_path / "r")
    np.testing.assert_array_equal(again.loss("total_time"), result.loss("total_time"))
    np.testing.assert_array_equal(again.arc_maxq, result.arc_maxq)
    assert again.runs[3].groups is not None and result.runs[3].groups is not None
    np.testing.assert_array_equal(again.runs[3].groups["exit"], result.runs[3].groups["exit"])
    assert again.spec == result.spec
    assert again.summary()["runs"] == 40


def test_convergence_stops_early(small: Building) -> None:
    res = run_monte_carlo(
        small,
        ScenarioSpec(),
        MCConfig(n_runs=200, batch_size=20, min_runs=40, target_halfwidth=1e9, workers=1),
    )
    assert res.converged is True and res.n == 40


def test_progress_callback(small: Building) -> None:
    seen: list[tuple[int, int]] = []
    run_monte_carlo(
        small,
        ScenarioSpec(),
        MCConfig(n_runs=10, batch_size=10, chunk_size=5, workers=1),
        progress=lambda d, t: seen.append((d, t)),
    )
    assert seen[-1] == (10, 10)


def test_tail_breakdown(result: MCResult) -> None:
    for loss in ("total_time", "self_evacuation_time"):
        bd = tail_breakdown(result, loss)
        assert bd["tail_scenarios"] >= 2
        for key in ("profiles", "floor_bands"):
            rows = bd[key]
            assert sum(r["occupant_share"] for r in rows) == pytest.approx(1.0)
            assert sum(r["straggler_share"] for r in rows) == pytest.approx(1.0)
            assert sum(r["last_out_share"] for r in rows) == pytest.approx(1.0)
        assert bd["headline"].startswith("Households whose most dependent member")


def test_floor_band_labels() -> None:
    assert floor_band(0) == "G/F–9/F"
    assert floor_band(34) == "30/F–39/F"
    assert floor_band(7, size=5) == "5/F–9/F"


@pytest.mark.slow
def test_performance_target() -> None:
    """Spec §4.3: 40 storeys, ~2,000 occupants, 1,000 runs in under 2 minutes."""
    b = generate("cruciform", storeys=40)
    res = run_monte_carlo(b, reference_spec(), MCConfig(n_runs=1000, keep_groups=False))
    assert res.array("n_agents").mean() > 1700
    assert res.elapsed < 120.0, f"{res.elapsed:.1f} s"


def test_worker_start_method_avoids_forking_jax(monkeypatch: pytest.MonkeyPatch) -> None:
    import multiprocessing as mp
    import sys
    import types

    from tailsafe.scenarios.montecarlo import _start_method

    monkeypatch.delitem(sys.modules, "jax", raising=False)
    if "fork" in mp.get_all_start_methods():
        assert _start_method() == "fork"
    monkeypatch.setitem(sys.modules, "jax", types.ModuleType("jax"))
    if "forkserver" in mp.get_all_start_methods():
        assert _start_method() == "forkserver"
