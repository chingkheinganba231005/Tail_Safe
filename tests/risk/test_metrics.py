from __future__ import annotations

import numpy as np
import pytest

from tailsafe.risk.metrics import (
    bootstrap_ci,
    cvar,
    cvar_halfwidth,
    paired_difference,
    summarize,
    var,
)


def test_var_and_cvar_on_integers() -> None:
    x = np.arange(1, 101, dtype=float)  # 1..100
    assert var(x, 0.95) == 95.0
    assert cvar(x, 0.95) == pytest.approx(np.mean([96, 97, 98, 99, 100]))
    assert cvar(x, 0.9) == pytest.approx(np.mean(np.arange(91, 101)))


def test_cvar_is_order_invariant_and_bounds() -> None:
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=997)
    assert cvar(x) == pytest.approx(cvar(rng.permutation(x)))
    assert var(x) <= cvar(x) <= x.max()
    assert cvar(x) >= x.mean()


def test_cvar_fractional_tail() -> None:
    # n(1-alpha) = 2.5: RU formula weights the partial sample correctly.
    x = np.arange(1, 51, dtype=float)
    expected = (50 + 49 + 0.5 * 48) / 2.5  # worst 2.5 samples; VaR = 48 at alpha=0.95, n=50
    assert var(x, 0.95) == 48.0
    assert cvar(x, 0.95) == pytest.approx(expected)


def test_cvar_matches_normal_theory() -> None:
    from scipy.stats import norm

    x = np.random.default_rng(1).normal(size=400_000)
    theory = norm.pdf(norm.ppf(0.95)) / 0.05
    assert cvar(x) == pytest.approx(theory, rel=0.01)


def test_bootstrap_ci_covers_mean() -> None:
    rng = np.random.default_rng(2)
    hits = 0
    for s in range(60):
        x = rng.normal(10.0, 2.0, size=200)
        lo, hi = bootstrap_ci(x, lambda r: r.mean(axis=1), n_boot=400, seed=s)
        hits += lo <= 10.0 <= hi
    assert hits >= 50  # nominal 95% coverage, loose bound


def test_summarize_and_censoring() -> None:
    x = np.array([10.0, 20.0, 30.0, np.inf, 50.0] * 20)
    s = summarize(x, cap=100.0, n_boot=200)
    assert s.censored == 20 and s.n == 100
    assert s.max == 100.0
    assert s.median.lo <= s.median.value <= s.median.hi
    assert s.cvar.value >= s.p95.value
    d = s.as_dict()
    assert d["cvar"]["value"] == s.cvar.value


def test_convergence_halfwidth_shrinks() -> None:
    rng = np.random.default_rng(3)
    small = cvar_halfwidth(rng.lognormal(size=200))
    large = cvar_halfwidth(rng.lognormal(size=5000))
    assert large < small


def test_paired_difference_detects_small_improvement() -> None:
    rng = np.random.default_rng(4)
    base = rng.lognormal(7, 0.3, size=1000)
    better = base * 0.98  # 2% better in every scenario (common random numbers)
    d = paired_difference(base, better, n_boot=500)
    assert d.value < 0 and d.hi < 0
    with pytest.raises(ValueError):
        paired_difference(base, better[:10])
