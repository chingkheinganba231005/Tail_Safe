"""Train/test splits and honest metrics for the surrogate.

* **Random split** (in-distribution): 80% of cases to train, 20% to test.
* **Leave one typology out**: train on three building typologies, test on
  the fourth — the question a new kind of building asks.

Metrics per test set, all against the held-out cases' own simulated outcomes:
mean pinball loss (minutes), absolute error of each quantile and of CVaR95
against the sample's value (minutes and relative), *coverage* — the share of
simulated outcomes below each predicted quantile, which should be close to
the quantile level for a calibrated model — the R² of P95 across cases, and
for edges the rank correlation of predicted and simulated queueing and the
overlap of the five most congested edges.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from tailsafe.config import Params, get_params
from tailsafe.surrogate.data import Case
from tailsafe.surrogate.features import GraphData, StaticGraph, graph_features, static_graph
from tailsafe.surrogate.model import (
    HORIZON,
    LOSS_NAMES,
    QUANTILES,
    ModelConfig,
    Stats,
    Target,
    TrainConfig,
    predict,
    train,
)
from tailsafe.surrogate.model import (
    Params as NetParams,
)


def prepare(
    records: Sequence[dict[str, Any]], params: Params | None = None
) -> tuple[list[GraphData], list[Target], list[dict[str, Any]]]:
    """Graphs, targets and case metadata from data-set records.

    Non-finite outcomes are censored at :data:`~tailsafe.surrogate.model.HORIZON`.
    """
    p = params or get_params()
    cache: dict[str, StaticGraph] = {}
    graphs: list[GraphData] = []
    targets: list[Target] = []
    meta: list[dict[str, Any]] = []
    for r in records:
        case = Case.from_dict(r)
        key = f"{case.template}:{sorted(case.options.items())}"
        if key not in cache:
            cache[key] = static_graph(case.building(), p)
        g = graph_features(cache[key], case.spec, p)
        eq = r.get("edge_queue", {})
        edge_q = np.array(
            [
                eq.get(eid, 0.0) if fwd else 0.0
                for eid, fwd in zip(g.edge_ids, g.edge_forward, strict=True)
            ]
        )
        graphs.append(g)
        targets.append(
            Target(
                samples=np.minimum(
                    np.array([r["losses"][n] for n in LOSS_NAMES], dtype=np.float64), HORIZON
                ),
                edge_queue=edge_q,
            )
        )
        meta.append(
            {
                "index": r["index"],
                "template": r["template"],
                "storeys": case.options["storeys"],
                "sim_seconds": float(r.get("wall_time_s", 0.0)),
                "runs": int(r.get("n_runs", 64)),
            }
        )
    return graphs, targets, meta


def metrics(
    net: NetParams,
    stats: Stats,
    graphs: Sequence[GraphData],
    targets: Sequence[Target],
) -> dict[str, Any]:
    """Accuracy and calibration on a test set."""
    preds: list[NDArray[np.float64]] = []
    edges: list[NDArray[np.float64]] = []
    for i in range(0, len(graphs), 16):
        qp, ep = predict(net, stats, graphs[i : i + 16])
        preds.append(qp)
        edges.extend(ep)
    pred = np.concatenate(preds)  # [n, L, Q+1] seconds
    out: dict[str, Any] = {"cases": len(graphs), "losses": {}}
    for li, name in enumerate(LOSS_NAMES):
        samples = [t.samples[li] for t in targets]
        emp = np.array([np.quantile(s, QUANTILES) for s in samples])
        emp_cvar = np.array([np.sort(s)[-max(1, round(0.05 * s.size)) :].mean() for s in samples])
        pq = pred[:, li, : len(QUANTILES)]
        pc = pred[:, li, -1]
        pin: list[float] = []
        for k, q in enumerate(QUANTILES):
            d = [smp - pq[i, k] for i, smp in enumerate(samples)]
            pin.append(float(np.mean([np.maximum(q * x, (q - 1) * x).mean() for x in d])))
        cover = {
            f"p{int(q * 100)}": float(
                np.mean([(s <= pq[i, k]).mean() for i, s in enumerate(samples)])
            )
            for k, q in enumerate(QUANTILES)
        }
        p95 = emp[:, -1]
        ss_res = float(((pq[:, -1] - p95) ** 2).sum())
        ss_tot = float(((p95 - p95.mean()) ** 2).sum()) or 1.0
        out["losses"][name] = {
            "pinball_min": float(np.mean(pin)) / 60.0,
            "mae_min": {
                f"p{int(q * 100)}": float(np.abs(pq[:, k] - emp[:, k]).mean()) / 60.0
                for k, q in enumerate(QUANTILES)
            },
            "mae_cvar95_min": float(np.abs(pc - emp_cvar).mean()) / 60.0,
            "relative_error_p95": float(np.mean(np.abs(pq[:, -1] - p95) / p95)),
            "r2_p95": 1.0 - ss_res / ss_tot,
            "coverage": cover,
        }
    rho: list[float] = []
    top5: list[float] = []
    for g, t, e_pred in zip(graphs, targets, edges, strict=True):
        fwd = g.edge_forward
        truth, guess = t.edge_queue[fwd], e_pred[fwd]
        if (truth > 0).sum() >= 5:
            r = spearmanr(truth, guess).statistic
            if np.isfinite(r):
                rho.append(float(r))
            a = set(np.argsort(-truth)[:5].tolist())
            b = set(np.argsort(-guess)[:5].tolist())
            top5.append(len(a & b) / 5.0)
    out["edges"] = {
        "spearman": float(np.mean(rho)) if rho else float("nan"),
        "top5_overlap": float(np.mean(top5)) if top5 else float("nan"),
    }
    return out


def speed(
    net: NetParams,
    stats: Stats,
    graphs: Sequence[GraphData],
    meta: Sequence[dict[str, Any]],
    stress_runs: int = 1000,
) -> dict[str, float]:
    """Prediction time against the simulator's time for a stress test of ``stress_runs``."""
    g = graphs[: min(20, len(graphs))]
    predict(net, stats, g[:1])  # compile
    t0 = time.perf_counter()
    for x in g:
        predict(net, stats, [x])
    per_pred = (time.perf_counter() - t0) / len(g)
    per_run = float(
        np.mean([m["sim_seconds"] / max(m["runs"], 1) for m in meta if m["sim_seconds"] > 0])
    )
    sim = per_run * stress_runs
    return {
        "prediction_ms": per_pred * 1000.0,
        "simulation_s_single_core": sim,
        "stress_runs": float(stress_runs),
        "speedup_single_core": sim / per_pred,
    }


def evaluate(
    records: Sequence[dict[str, Any]],
    *,
    mcfg: ModelConfig | None = None,
    tcfg: TrainConfig | None = None,
    seed: int = 0,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Random split and leave-one-typology-out."""
    graphs, targets, meta = prepare(records)
    rng = np.random.default_rng(seed)
    n = len(graphs)
    order = rng.permutation(n)
    n_test = max(1, n // 5)
    test, rest = order[:n_test], order[n_test:]
    n_val = max(1, len(rest) // 10)
    val, tr = rest[:n_val], rest[n_val:]

    def pick(idx: Sequence[int] | NDArray[np.int64]) -> tuple[list[GraphData], list[Target]]:
        return [graphs[i] for i in idx], [targets[i] for i in idx]

    result: dict[str, Any] = {"cases": n, "holdout": {}}
    if log:
        log(f"random split: {len(tr)} train, {len(val)} validation, {len(test)} test")
    net, stats, hist = train(*pick(tr), *pick(val), mcfg=mcfg, tcfg=tcfg, log=log)
    result["random"] = metrics(net, stats, *pick(test))
    result["random"]["epochs"] = len(hist)
    result["speed"] = speed(net, stats, [graphs[i] for i in test], [meta[i] for i in test])
    for typ in sorted({m["template"] for m in meta}):
        te = [i for i in range(n) if meta[i]["template"] == typ]
        others = [i for i in order if meta[i]["template"] != typ]
        n_val = max(1, len(others) // 10)
        if log:
            log(f"hold out {typ}: {len(others) - n_val} train, {len(te)} test")
        net_t, stats_t, _ = train(
            *pick(others[n_val:]), *pick(others[:n_val]), mcfg=mcfg, tcfg=tcfg, log=log
        )
        result["holdout"][typ] = metrics(net_t, stats_t, *pick(te))
    return result


def evaluation_markdown(res: dict[str, Any]) -> str:
    """Tables of :func:`evaluate` (P95 of each loss, coverage, edges)."""
    lines = [
        "| Test set | Cases | Loss | P95 error (min) | P95 relative error | R² of P95 "
        "| CVaR₉₅ error (min) | Coverage P50 / P75 / P90 / P95 |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    sets = [("Random 20%", res["random"])] + [
        (f"Unseen: {k}", v) for k, v in res["holdout"].items()
    ]
    short = {
        "total_time": "everyone out",
        "self_evacuation_time": "last self-evacuee",
        "p95_occupant_time": "95% out",
    }
    for label, m in sets:
        for name, s in m["losses"].items():
            c = s["coverage"]
            lines.append(
                f"| {label} | {m['cases']} | {short[name]} | {s['mae_min']['p95']:.1f} | "
                f"{100 * s['relative_error_p95']:.0f}% | {s['r2_p95']:.2f} | "
                f"{s['mae_cvar95_min']:.1f} | "
                f"{c['p50']:.2f} / {c['p75']:.2f} / {c['p90']:.2f} / {c['p95']:.2f} |"
            )
    lines += [
        "",
        "| Test set | Edge queueing rank correlation | Top-5 congested edges recovered |",
        "|---|---:|---:|",
    ]
    for label, m in sets:
        lines.append(
            f"| {label} | {m['edges']['spearman']:.2f} | {100 * m['edges']['top5_overlap']:.0f}% |"
        )
    sp = res.get("speed")
    if sp:
        lines += [
            "",
            f"Prediction takes {sp['prediction_ms']:.0f} ms; a {int(sp['stress_runs'])}-run "
            "stress test of "
            f"the same buildings takes about {sp['simulation_s_single_core']:.0f} s on one core "
            f"(≈ {sp['speedup_single_core']:.0f}× faster).",
        ]
    return "\n".join(lines) + "\n"
