"""Bottleneck attribution: where the tail comes from, and what would fix it.

Three complementary views, combined in one ranked table:

1. **Recurrence** — in the tail scenarios (loss ≥ VaR_α), how often does a
   queue of at least ``threshold`` people form at each place, and how many
   person-seconds of queueing does it hold?
2. **Structure** — min-cut membership and flow-weighted clearance time
   (:mod:`tailsafe.analysis.structural`), from the graph alone.
3. **Counterfactual criticality** — re-run scenarios (common random numbers)
   with one candidate element relaxed (capacity × ``factor``, or a blocked stair
   kept clear) and measure ΔCVaR_α with a paired bootstrap CI. Relaxing never
   slows a scenario (monotonicity, tested), so a scenario that is not re-run is
   bounded by its baseline loss. We start with the worst ``rerun_fraction`` and
   keep re-running any scenario whose baseline could still belong to the new
   tail; when none can, the counterfactual CVaR is exact.

Candidates are whole staircases, each staircase's doors, each final exit, the
elements downstream of the worst tail queues, and any staircase blocked in the
scenario specification.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tailsafe.analysis.structural import structural
from tailsafe.building.model import Building, EdgeKind, NodeType
from tailsafe.config import Params, get_params
from tailsafe.risk.metrics import Estimate, cvar, paired_difference, var
from tailsafe.scenarios.montecarlo import MCConfig, MCResult, run_monte_carlo
from tailsafe.scenarios.spec import Dist, ScenarioSpec, StairBlockage
from tailsafe.sim.network import SimNetwork, compile_network
from tailsafe.sim.routing import Router

NEVER = 1e12  # blockage time used to "unblock" while keeping random-number slots aligned


@dataclass(frozen=True)
class Candidate:
    """An element to relax in a counterfactual re-run."""

    key: str
    label: str
    kind: str  # stair | stair_doors | exit | edge | unblock
    edges: tuple[str, ...] = ()
    stair: str | None = None


@dataclass(frozen=True)
class QueueSpot:
    """Recurrence statistics of queueing on one arc."""

    arc: int
    where: str
    recurrence: float  # share of tail scenarios with max queue >= threshold
    tail_person_seconds: float
    all_person_seconds: float
    tail_max_queue: float


@dataclass
class BottleneckRow:
    """One ranked element."""

    rank: int
    key: str
    label: str
    kind: str
    delta_cvar: Estimate | None
    relative_change: float | None
    recurrence: float
    tail_person_seconds: float
    structural_clearance_s: float
    in_min_cut: bool


def queue_recurrence(
    result: MCResult,
    loss: str = "p95_occupant_time",
    *,
    alpha: float | None = None,
    threshold: float = 10.0,
    top: int = 15,
) -> list[QueueSpot]:
    """Where queues form in the tail scenarios, worst first."""
    a = alpha or result.config.alpha
    x = result.loss(loss)
    tail = x >= var(x, a)
    maxq = result.arc_maxq
    qint = result.arc_qint
    rec = (maxq[tail] >= threshold).mean(axis=0)
    tail_ps = qint[tail].mean(axis=0)
    all_ps = qint.mean(axis=0)
    order = np.argsort(-tail_ps)[:top]
    return [
        QueueSpot(
            arc=int(k),
            where=result.arc_labels[int(k)],
            recurrence=float(rec[k]),
            tail_person_seconds=float(tail_ps[k]),
            all_person_seconds=float(all_ps[k]),
            tail_max_queue=float(maxq[tail][:, k].mean()),
        )
        for k in order
        if tail_ps[k] > 0
    ]


def candidates(
    building: Building,
    spec: ScenarioSpec,
    net: SimNetwork,
    spots: list[QueueSpot],
    *,
    n_edges: int = 4,
    router: Router | None = None,
) -> list[Candidate]:
    """Elements worth a counterfactual re-run."""
    nodes = building.node_by_id
    stairish = {NodeType.STAIR_LANDING, NodeType.PROTECTED_LOBBY}
    out: list[Candidate] = []
    for s in building.stairs:
        flights = tuple(
            e.id for e in building.edges if e.kind == EdgeKind.STAIR and e.stair == s.id
        )
        out.append(Candidate(f"stair:{s.id}", f"{s.label} (all flights)", "stair", flights, s.id))
        doors = tuple(
            e.id
            for e in building.edges
            if e.kind == EdgeKind.DOOR
            and (nodes[e.target].stair == s.id or nodes[e.source].stair == s.id)
            and any(nodes[n].type in stairish for n in (e.source, e.target))
            and not any(nodes[n].type == NodeType.EXIT for n in (e.source, e.target))
        )
        if doors:
            out.append(
                Candidate(
                    f"stair_doors:{s.id}",
                    f"{s.label} doors on every floor",
                    "stair_doors",
                    doors,
                    s.id,
                )
            )
    for e in building.edges:
        if nodes[e.target].type == NodeType.EXIT or nodes[e.source].type == NodeType.EXIT:
            out.append(Candidate(f"exit:{e.id}", building.describe_edge(e.id), "exit", (e.id,)))
    for blk in spec.stair_blockages:
        out.append(
            Candidate(
                f"unblock:{blk.stair}",
                f"Keep Stair {blk.stair} usable (no blockage)",
                "unblock",
                stair=blk.stair,
            )
        )
    # The element each worst queue is waiting for (next arc towards the exits).
    tables = (router or Router(net)).tables()
    seen = {e for c in out for e in c.edges}
    for spot in spots:
        v = int(net.arc_dst[spot.arc])
        b = int(tables.next_arc[0, 0, v])
        if b < 0:
            continue
        eid = net.edge_ids[int(net.arc_edge[b])]
        if eid in seen:
            continue
        seen.add(eid)
        out.append(Candidate(f"edge:{eid}", building.describe_edge(eid), "edge", (eid,)))
        if sum(c.kind == "edge" for c in out) >= n_edges:
            break
    return out


def _counterfactual_spec(spec: ScenarioSpec, cand: Candidate, factor: float) -> ScenarioSpec:
    if cand.kind == "unblock":
        blocks = [
            StairBlockage(stair=b.stair, time=Dist.fixed(NEVER)) if b.stair == cand.stair else b
            for b in spec.stair_blockages
        ]
        return spec.model_copy(update={"stair_blockages": blocks})
    mult = dict(spec.capacity_multipliers)
    for e in cand.edges:
        mult[e] = mult.get(e, 1.0) * factor
    return spec.model_copy(update={"capacity_multipliers": mult})


def attribute_bottlenecks(
    result: MCResult,
    *,
    loss: str = "p95_occupant_time",
    factor: float = 1.5,
    rerun_fraction: float = 0.2,
    threshold: float = 10.0,
    n_edges: int = 4,
    params: Params | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Ranked, counterfactual bottleneck table for a finished stress test."""
    if result.building is None:
        raise ValueError("result has no building; re-run or load a result saved with it")
    p = params or get_params()
    building = result.building
    net = compile_network(building, p)
    router = Router(net, p)
    alpha = result.config.alpha
    base = result.loss(loss)
    spots = queue_recurrence(result, loss, threshold=threshold)
    cands = candidates(building, result.spec, net, spots, n_edges=n_edges, router=router)
    st = structural(net, params=p)

    k = max(2, math.ceil(rerun_fraction * result.n))
    first = [int(result.runs[j].index) for j in np.argsort(base)[-k:]]
    cfg = MCConfig(
        n_runs=result.config.n_runs,
        seed=result.config.seed,
        lhs=result.config.lhs,
        batch_size=result.config.batch_size,
        workers=workers if workers is not None else result.config.workers,
        keep_groups=False,
        sim=result.config.sim,
    )
    pos = {r.index: j for j, r in enumerate(result.runs)}
    edge_arcs = {e: np.flatnonzero(net.arc_edge == i) for i, e in enumerate(net.edge_ids)}
    maxq = result.arc_maxq
    qint = result.arc_qint
    tail = base >= var(base, alpha)
    base_cvar = cvar(base, alpha)

    rows: list[BottleneckRow] = []
    reruns: dict[str, int] = {}
    for cand in cands:
        spec2 = _counterfactual_spec(result.spec, cand, factor)
        new = base.copy()
        done: set[int] = set()
        todo = first
        while todo:
            rerun = run_monte_carlo(building, spec2, cfg, params=p, indices=sorted(todo))
            for run in rerun.runs:
                new[pos[run.index]] = min(getattr(run, loss), base[pos[run.index]])
            done.update(todo)
            threshold_now = var(new, alpha)
            todo = [
                int(r.index)
                for j, r in enumerate(result.runs)
                if r.index not in done and base[j] >= threshold_now
            ]
        reruns[cand.key] = len(done)
        delta = paired_difference(base, new, alpha=alpha, seed=result.config.seed)
        arcs = (
            np.concatenate([edge_arcs[e] for e in cand.edges])
            if cand.edges
            else np.zeros(0, dtype=np.int64)
        )
        rec = float((maxq[tail][:, arcs] >= threshold).any(axis=1).mean()) if arcs.size else 0.0
        rows.append(
            BottleneckRow(
                rank=0,
                key=cand.key,
                label=cand.label,
                kind=cand.kind,
                delta_cvar=delta,
                relative_change=delta.value / base_cvar if base_cvar else None,
                recurrence=rec,
                tail_person_seconds=float(qint[tail][:, arcs].sum(axis=1).mean())
                if arcs.size
                else 0.0,
                structural_clearance_s=float(st.clearance_time[arcs].max()) if arcs.size else 0.0,
                in_min_cut=bool(set(arcs.tolist()) & set(st.min_cut_arcs)),
            )
        )
    rows.sort(key=lambda r: r.delta_cvar.value if r.delta_cvar else 0.0)
    for i, r in enumerate(rows):
        r.rank = i + 1
    headline = ""
    if rows and rows[0].delta_cvar and rows[0].delta_cvar.value < 0:
        top = rows[0]
        assert top.delta_cvar is not None
        cut = (
            f"cuts CVaR{int(100 * alpha)} of {loss.replace('_', ' ')} by "
            f"{-top.delta_cvar.value / 60:.1f} min (95% CI {-top.delta_cvar.hi / 60:.1f} "
            f"to {-top.delta_cvar.lo / 60:.1f} min)"
        )
        if top.kind == "unblock":
            stair = top.key.split(":", 1)[1]
            headline = f"Losing Stair {stair} drives the tail: keeping it usable {cut}."
        else:
            headline = (
                f"The most critical element is {top.label}: "
                f"{100 * (factor - 1):.0f}% more capacity {cut}."
            )
    return {
        "loss": loss,
        "alpha": alpha,
        "baseline_cvar": base_cvar,
        "factor": factor,
        "rerun_scenarios": reruns,
        "max_flow_persons_per_s": st.max_flow,
        "min_cut": [result.arc_labels[a] for a in st.min_cut_arcs],
        "queues": [asdict(s) for s in spots],
        "ranking": [
            {**asdict(r), "delta_cvar": asdict(r.delta_cvar) if r.delta_cvar else None}
            for r in rows
        ],
        "headline": headline,
    }
