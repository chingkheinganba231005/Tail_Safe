"""Tail-risk statistics of a loss sample (e.g. total evacuation time per scenario).

* ``VaR_α`` — the α-quantile (inverted empirical CDF).
* ``CVaR_α`` — expected loss in the worst ``1 - α`` of scenarios, via the
  Rockafellar–Uryasev form ``VaR + E[(X - VaR)+] / (1 - α)``. For ``n(1-α)``
  integer this is exactly the mean of the worst ``n(1-α)`` samples.

Confidence intervals are percentile bootstrap intervals, computed vectorised.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

Stat = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def _rows(x: ArrayLike) -> NDArray[np.float64]:
    a = np.asarray(x, dtype=np.float64)
    return a[None, :] if a.ndim == 1 else a


def var(x: ArrayLike, alpha: float = 0.95) -> float:
    """Value at risk: the α-quantile of the sample."""
    return float(_var_rows(_rows(x), alpha)[0])


def cvar(x: ArrayLike, alpha: float = 0.95) -> float:
    """Conditional value at risk: mean loss in the worst ``1 - α`` of the sample."""
    return float(_cvar_rows(_rows(x), alpha)[0])


def _var_rows(x: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    n = x.shape[1]
    k = min(max(int(np.ceil(alpha * n - 1e-9)), 1), n) - 1
    out: NDArray[np.float64] = np.partition(x, k, axis=1)[:, k]
    return out


def _cvar_rows(x: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    v = _var_rows(x, alpha)
    with np.errstate(invalid="ignore"):
        excess = np.maximum(x - v[:, None], 0.0)
        excess[np.isnan(excess)] = 0.0
        out: NDArray[np.float64] = v + excess.mean(axis=1) / (1.0 - alpha)
    return out


def _quantile_rows(q: float) -> Stat:
    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return _var_rows(x, q)

    return f


def bootstrap_ci(
    x: ArrayLike,
    stat: Stat,
    *,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval of a row-wise statistic."""
    a = np.asarray(x, dtype=np.float64)
    if a.size < 2:
        v = float(stat(_rows(a))[0]) if a.size else float("nan")
        return (v, v)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    vals = stat(a[idx])
    lo, hi = np.quantile(vals, [(1 - confidence) / 2, (1 + confidence) / 2])
    return (float(lo), float(hi))


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its confidence interval."""

    value: float
    lo: float
    hi: float

    @property
    def half_width(self) -> float:
        """Half the CI width."""
        return 0.5 * (self.hi - self.lo)


@dataclass(frozen=True)
class RiskSummary:
    """Distribution summary of one loss variable."""

    n: int
    alpha: float
    confidence: float
    mean: Estimate
    median: Estimate
    p95: Estimate
    p99: Estimate
    var: Estimate
    cvar: Estimate
    max: float
    censored: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly dict."""
        return asdict(self)


def summarize(
    x: ArrayLike,
    *,
    alpha: float = 0.95,
    confidence: float = 0.95,
    n_boot: int = 1000,
    seed: int = 0,
    cap: float | None = None,
) -> RiskSummary:
    """Mean, median, P95, P99, VaR_α and CVaR_α with bootstrap CIs.

    Non-finite losses (e.g. someone never got out) are replaced by ``cap`` when
    given and counted as ``censored``; the tail statistics are then lower bounds.
    """
    a = np.asarray(x, dtype=np.float64).copy()
    bad = ~np.isfinite(a)
    censored = int(bad.sum())
    if censored and cap is not None:
        a[bad] = cap

    def est(stat: Stat, offset: int) -> Estimate:
        lo, hi = bootstrap_ci(a, stat, n_boot=n_boot, confidence=confidence, seed=seed + offset)
        return Estimate(float(stat(_rows(a))[0]), lo, hi)

    def mean_rows(r: NDArray[np.float64]) -> NDArray[np.float64]:
        out: NDArray[np.float64] = r.mean(axis=1)
        return out

    def cvar_stat(r: NDArray[np.float64]) -> NDArray[np.float64]:
        return _cvar_rows(r, alpha)

    return RiskSummary(
        n=int(a.size),
        alpha=alpha,
        confidence=confidence,
        mean=est(mean_rows, 1),
        median=est(_quantile_rows(0.5), 2),
        p95=est(_quantile_rows(0.95), 3),
        p99=est(_quantile_rows(0.99), 4),
        var=est(_quantile_rows(alpha), 5),
        cvar=est(cvar_stat, 6),
        max=float(a.max()) if a.size else float("nan"),
        censored=censored,
    )


def cvar_halfwidth(x: ArrayLike, *, alpha: float = 0.95, n_boot: int = 500, seed: int = 0) -> float:
    """Half-width of the 95% bootstrap CI of CVaR_α (convergence criterion)."""
    lo, hi = bootstrap_ci(x, lambda r: _cvar_rows(r, alpha), n_boot=n_boot, seed=seed)
    return 0.5 * (hi - lo)


def paired_difference(
    base: ArrayLike,
    other: ArrayLike,
    *,
    alpha: float = 0.95,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """CVaR_α(other) - CVaR_α(base) with a *paired* bootstrap CI.

    Both samples must come from the same scenarios (common random numbers):
    resampling scenario indices jointly keeps the pairing, which is what makes
    small improvements detectable.
    """
    b = np.asarray(base, dtype=np.float64)
    o = np.asarray(other, dtype=np.float64)
    if b.shape != o.shape:
        raise ValueError("paired samples must have the same length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, b.size, size=(n_boot, b.size))
    diffs = _cvar_rows(o[idx], alpha) - _cvar_rows(b[idx], alpha)
    lo, hi = np.quantile(diffs, [(1 - confidence) / 2, (1 + confidence) / 2])
    return Estimate(cvar(o, alpha) - cvar(b, alpha), float(lo), float(hi))
