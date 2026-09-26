"""A small, deterministic CMA-ES for low-dimensional box-constrained problems.

(μ/μ_w, λ)-CMA-ES with cumulative step-size adaptation and rank-one plus
rank-μ covariance updates (Hansen's tutorial defaults). Candidates are clipped
to the box; the objective is expected to be noisy-but-coupled (common random
numbers), so no re-evaluation strategy is used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class CMAResult:
    """Best point found and the evaluation history."""

    x: NDArray[np.float64]
    f: float
    history: list[tuple[NDArray[np.float64], float]] = field(default_factory=list)


def cmaes(
    f: Callable[[NDArray[np.float64]], float],
    x0: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    *,
    sigma0: float = 0.3,
    iterations: int = 10,
    popsize: int | None = None,
    seed: int = 0,
) -> CMAResult:
    """Minimise ``f`` over the box ``[lower, upper]`` starting from ``x0``.

    The search runs in normalised coordinates (box mapped to ``[0, 1]^n``), so
    ``sigma0`` is a fraction of the box width.
    """
    rng = np.random.default_rng(seed)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    span = np.where(hi > lo, hi - lo, 1.0)
    n = lo.size
    lam = popsize or 4 + int(3 * np.log(n))
    mu = lam // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / np.sum(w**2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    mean = np.clip((np.asarray(x0, dtype=float) - lo) / span, 0.0, 1.0)
    sigma = sigma0
    C = np.eye(n)
    pc = np.zeros(n)
    ps = np.zeros(n)
    history: list[tuple[NDArray[np.float64], float]] = []

    def evaluate(z: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
        x = lo + np.clip(z, 0.0, 1.0) * span
        val = float(f(x))
        history.append((x, val))
        return x, val

    best_x, best_f = evaluate(mean)
    for gen in range(iterations):
        eigval, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(eigval, 1e-20))
        Z = rng.standard_normal((lam, n))
        Y = Z @ (B * D).T
        X = mean + sigma * Y
        vals = np.empty(lam)
        for k in range(lam):
            x, vals[k] = evaluate(X[k])
            if vals[k] < best_f:
                best_x, best_f = x, vals[k]
        order = np.argsort(vals)[:mu]
        y_w = w @ Y[order]
        mean = np.clip(mean + sigma * y_w, 0.0, 1.0)
        c_inv_sqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (c_inv_sqrt @ y_w)
        hsig = float(
            np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))
            < 1.4 + 2 / (n + 1) * chi_n
        )
        pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
        rank_mu = (Y[order].T * w) @ Y[order]
        C = (
            (1 - c1 - cmu) * C
            + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            + cmu * rank_mu
        )
        sigma *= float(np.exp((cs / damps) * (np.linalg.norm(ps) / chi_n - 1)))
        sigma = min(sigma, 1.0)
    return CMAResult(x=best_x, f=best_f, history=history)
