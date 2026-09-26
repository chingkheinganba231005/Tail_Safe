"""CVaR estimates stabilise as the number of Monte Carlo runs grows (spec §7)."""

from __future__ import annotations

import numpy as np

from tailsafe.building.templates import generate
from tailsafe.risk.metrics import cvar, cvar_halfwidth
from tailsafe.scenarios.montecarlo import MCConfig, run_monte_carlo
from tailsafe.scenarios.spec import demo_spec


def test_cvar_converges() -> None:
    b = generate("cruciform", storeys=16, flats_per_wing=2)
    res = run_monte_carlo(b, demo_spec(), MCConfig(n_runs=320, batch_size=40, workers=2))
    x = res.loss("self_evacuation_time")
    widths = [cvar_halfwidth(x[:n]) for n in (40, 80, 160, 320)]
    assert widths[-1] < widths[0]
    # The CI narrows substantially (tail CIs from few samples are noisy, so the
    # bound is looser than the asymptotic 1/sqrt(n)).
    assert widths[-1] <= 0.75 * widths[0]
    # The final estimate lies inside the 160-run confidence band.
    assert abs(cvar(x) - cvar(x[:160])) <= widths[2] * 1.5
    assert np.isfinite(x).all()
