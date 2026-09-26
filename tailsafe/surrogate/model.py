"""Message-passing graph network (JAX) predicting evacuation-time quantiles.

Architecture: node, edge and global encoders; ``layers`` rounds of message
passing (messages from both endpoints and the edge; mean aggregation;
residual updates) with a global context vector that reads the mean and max
of the nodes and feeds every node update, so information travels across tall
buildings in a few layers; a graph head giving, per loss, monotone quantiles
(P50 < P75 < P90 < P95) and CVaR95 above P95; an edge head giving log mean
queueing per edge.

Training minimises the pinball loss of each quantile against *all* simulated
outcomes of a case (proper quantile regression), a squared error for CVaR95
against the sample's CVaR, and a squared error for edge queueing.

Outcomes that never happen within the simulation horizon (``inf``: e.g. more
than 5% of residents incapacitated, so "95% out" is never reached) are
censored at the horizon, as in the stress-test report; a prediction at the
horizon therefore means "not within the horizon".

The spec asked for PyTorch Geometric; its CPU wheels are not reachable from
this environment, so the network is written directly in JAX (same model
class — a message-passing GNN).
"""

from __future__ import annotations

import itertools
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import NDArray

from tailsafe.sim.meso import SimConfig
from tailsafe.surrogate.features import N_EDGE, N_GLOBAL, N_NODE, GraphData

QUANTILES = (0.5, 0.75, 0.9, 0.95)
LOSS_NAMES = ("total_time", "self_evacuation_time", "p95_occupant_time")
N_OUT = len(QUANTILES) + 1  # quantiles + CVaR95
TIME_SCALE = 3600.0  # s -> hours
HORIZON = SimConfig().t_max  # s; non-finite outcomes are censored here
Params = dict[str, Any]


@dataclass(frozen=True)
class ModelConfig:
    """Size of the network."""

    hidden: int = 48
    layers: int = 5
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings."""

    epochs: int = 120
    batch_graphs: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-4
    cvar_weight: float = 0.5
    edge_weight: float = 0.2
    patience: int | None = None  # epochs without improvement; None = full schedule


@dataclass
class Stats:
    """Feature standardisation (from the training set)."""

    node_mu: NDArray[np.float32]
    node_sd: NDArray[np.float32]
    edge_mu: NDArray[np.float32]
    edge_sd: NDArray[np.float32]
    glob_mu: NDArray[np.float32]
    glob_sd: NDArray[np.float32]

    @classmethod
    def fit(cls, graphs: Sequence[GraphData]) -> Stats:
        """Mean and standard deviation of each feature column."""

        def ms(x: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
            mu = x.mean(axis=0).astype(np.float32)
            sd = x.std(axis=0).astype(np.float32)
            return mu, np.where(sd > 1e-6, sd, 1.0).astype(np.float32)

        n = ms(np.concatenate([g.node_x for g in graphs]))
        e = ms(np.concatenate([g.edge_x for g in graphs]))
        gl = ms(np.stack([g.global_x for g in graphs]))
        return cls(*n, *e, *gl)

    def as_dict(self) -> dict[str, list[float]]:
        """JSON-friendly form."""
        return {k: np.asarray(v).tolist() for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> Stats:
        """Inverse of :meth:`as_dict`."""
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})


@dataclass
class Target:
    """Training target of one case: outcome samples (s) and edge queueing."""

    samples: NDArray[np.float64]  # [3, runs]
    edge_queue: NDArray[np.float64]  # [directed edges] person-seconds (forward copies only)


# ----------------------------------------------------------------------------- batching
def _bucket(n: int, step: int) -> int:
    return max(step, math.ceil(n / step) * step)


def make_batch(
    graphs: Sequence[GraphData],
    stats: Stats,
    targets: Sequence[Target] | None = None,
    runs: int = 64,
) -> dict[str, Any]:
    """Pack graphs into padded arrays (a dummy graph absorbs the padding)."""
    B = len(graphs)
    n_tot = sum(g.n_nodes for g in graphs)
    e_tot = sum(g.n_edges for g in graphs)
    n_pad, e_pad = _bucket(n_tot + 1, 512), _bucket(e_tot + 1, 1024)
    node_x = np.zeros((n_pad, N_NODE), np.float32)
    edge_x = np.zeros((e_pad, N_EDGE), np.float32)
    src = np.full(e_pad, n_pad - 1, np.int32)
    dst = np.full(e_pad, n_pad - 1, np.int32)
    node_graph = np.full(n_pad, B, np.int32)
    edge_mask = np.zeros(e_pad, np.float32)
    node_mask = np.zeros(n_pad, np.float32)
    edge_y = np.zeros(e_pad, np.float32)
    edge_ym = np.zeros(e_pad, np.float32)
    glob = np.zeros((B + 1, N_GLOBAL), np.float32)
    ys = np.zeros((B, len(LOSS_NAMES), runs), np.float32)
    n0 = e0 = 0
    for k, g in enumerate(graphs):
        n, e = g.n_nodes, g.n_edges
        node_x[n0 : n0 + n] = (g.node_x - stats.node_mu) / stats.node_sd
        edge_x[e0 : e0 + e] = (g.edge_x - stats.edge_mu) / stats.edge_sd
        src[e0 : e0 + e] = g.edge_src + n0
        dst[e0 : e0 + e] = g.edge_dst + n0
        node_graph[n0 : n0 + n] = k
        node_mask[n0 : n0 + n] = 1.0
        edge_mask[e0 : e0 + e] = 1.0
        glob[k] = (g.global_x - stats.glob_mu) / stats.glob_sd
        if targets is not None:
            t = targets[k]
            edge_y[e0 : e0 + e] = np.log1p(t.edge_queue / 60.0)
            edge_ym[e0 : e0 + e] = g.edge_forward.astype(np.float32)
            s = t.samples / TIME_SCALE
            ys[k] = s[:, :runs] if s.shape[1] >= runs else np.resize(s, (len(LOSS_NAMES), runs))
        n0 += n
        e0 += e
    return {
        "node_x": node_x,
        "edge_x": edge_x,
        "src": src,
        "dst": dst,
        "node_graph": node_graph,
        "node_mask": node_mask,
        "edge_mask": edge_mask,
        "glob": glob,
        "y": ys,
        "edge_y": edge_y,
        "edge_ym": edge_ym,
    }


# ----------------------------------------------------------------------------- network
def _mlp_init(key: Any, sizes: Sequence[int]) -> list[dict[str, Any]]:
    layers = []
    for a, b in itertools.pairwise(sizes):
        key, sub = jax.random.split(key)
        layers.append(
            {"W": jax.random.normal(sub, (a, b)) * jnp.sqrt(2.0 / (a + b)), "b": jnp.zeros(b)}
        )
    return layers


def _mlp(layers: list[dict[str, Any]], x: Any) -> Any:
    for i, lyr in enumerate(layers):
        x = x @ lyr["W"] + lyr["b"]
        if i < len(layers) - 1:
            x = jax.nn.silu(x)
    return x


def init_params(cfg: ModelConfig) -> Params:
    """Random initial weights."""
    h = cfg.hidden
    key = jax.random.PRNGKey(cfg.seed)
    ks = jax.random.split(key, 6 + 3 * cfg.layers)
    params: Params = {
        "enc_n": _mlp_init(ks[0], (N_NODE, h, h)),
        "enc_e": _mlp_init(ks[1], (N_EDGE, h, h)),
        "enc_g": _mlp_init(ks[2], (N_GLOBAL, h, h)),
        "layers": [
            {
                "msg": _mlp_init(ks[6 + 3 * i], (3 * h, h, h)),
                "upd": _mlp_init(ks[7 + 3 * i], (3 * h, h, h)),
                "glob": _mlp_init(ks[8 + 3 * i], (3 * h, h, h)),
            }
            for i in range(cfg.layers)
        ],
        "head_g": _mlp_init(ks[3], (3 * h, h, len(LOSS_NAMES) * N_OUT)),
        "head_e": _mlp_init(ks[4], (3 * h, h, 1)),
    }
    return params


def apply(params: Params, batch: dict[str, Any]) -> tuple[Any, Any]:
    """Predicted outcomes [graphs, losses, N_OUT] (hours) and edge log-queueing."""
    B = batch["glob"].shape[0] - 1  # one row per graph plus the padding graph
    ng = batch["node_graph"]
    nm = batch["node_mask"][:, None]
    em = batch["edge_mask"][:, None]
    src, dst = batch["src"], batch["dst"]
    n_nodes = batch["node_x"].shape[0]
    h = _mlp(params["enc_n"], batch["node_x"]) * nm
    e = _mlp(params["enc_e"], batch["edge_x"]) * em
    g = _mlp(params["enc_g"], batch["glob"])
    deg = jax.ops.segment_sum(em[:, 0], dst, num_segments=n_nodes)
    cnt = jax.ops.segment_sum(nm[:, 0], ng, num_segments=B + 1)
    for lyr in params["layers"]:
        m = _mlp(lyr["msg"], jnp.concatenate([h[src], h[dst], e], axis=-1)) * em
        agg = jax.ops.segment_sum(m, dst, num_segments=n_nodes) / jnp.maximum(deg, 1.0)[:, None]
        h = h + _mlp(lyr["upd"], jnp.concatenate([h, agg, g[ng]], axis=-1)) * nm
        e = e + m
        mean = jax.ops.segment_sum(h, ng, num_segments=B + 1) / jnp.maximum(cnt, 1.0)[:, None]
        mx = jax.ops.segment_max(jnp.where(nm > 0, h, -1e9), ng, num_segments=B + 1)
        mx = jnp.where(cnt[:, None] > 0, mx, 0.0)
        g = g + _mlp(lyr["glob"], jnp.concatenate([g, mean, mx], axis=-1))
    mean = jax.ops.segment_sum(h, ng, num_segments=B + 1) / jnp.maximum(cnt, 1.0)[:, None]
    mx = jax.ops.segment_max(jnp.where(nm > 0, h, -1e9), ng, num_segments=B + 1)
    mx = jnp.where(cnt[:, None] > 0, mx, 0.0)
    raw = _mlp(params["head_g"], jnp.concatenate([g, mean, mx], axis=-1))[:B]
    raw = raw.reshape(B, len(LOSS_NAMES), N_OUT)
    base = jax.nn.softplus(raw[..., :1])
    steps = jax.nn.softplus(raw[..., 1:])
    out = jnp.concatenate([base, base + jnp.cumsum(steps, axis=-1)], axis=-1)
    edge = _mlp(params["head_e"], jnp.concatenate([h[src], h[dst], e], axis=-1))[:, 0]
    return out, edge


def loss_fn(params: Params, batch: dict[str, Any], tcfg: TrainConfig) -> Any:
    """Pinball (quantiles) + squared error (CVaR95, edge queueing)."""
    pred, edge = apply(params, batch)
    y = batch["y"]  # [B, L, R]
    qs = jnp.asarray(QUANTILES)
    diff = y[:, :, None, :] - pred[:, :, : len(QUANTILES), None]  # [B, L, Q, R]
    pin = jnp.maximum(qs[None, None, :, None] * diff, (qs[None, None, :, None] - 1.0) * diff)
    pinball = pin.mean()
    ysort = jnp.sort(y, axis=-1)
    k = max(1, round(0.05 * y.shape[-1]))
    cvar = ysort[..., -k:].mean(axis=-1)
    cvar_err = ((pred[..., -1] - cvar) ** 2).mean()
    em = batch["edge_ym"]
    edge_err = (em * (edge - batch["edge_y"]) ** 2).sum() / jnp.maximum(em.sum(), 1.0)
    return pinball + tcfg.cvar_weight * cvar_err + tcfg.edge_weight * edge_err


# ----------------------------------------------------------------------------- training
def train(
    graphs: Sequence[GraphData],
    targets: Sequence[Target],
    val_graphs: Sequence[GraphData] = (),
    val_targets: Sequence[Target] = (),
    *,
    mcfg: ModelConfig | None = None,
    tcfg: TrainConfig | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[Params, Stats, list[dict[str, float]]]:
    """Fit the network; keeps the weights with the best validation loss."""
    mcfg = mcfg or ModelConfig()
    tcfg = tcfg or TrainConfig()
    stats = Stats.fit(graphs)
    params = init_params(mcfg)
    steps_per_epoch = max(1, math.ceil(len(graphs) / tcfg.batch_graphs))
    sched = optax.cosine_decay_schedule(tcfg.lr, tcfg.epochs * steps_per_epoch, alpha=0.05)
    opt = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=tcfg.weight_decay)
    )
    state = opt.init(params)

    def prep(batch: dict[str, Any]) -> dict[str, Any]:
        return {k: jnp.asarray(v) for k, v in batch.items()}

    @jax.jit
    def step(p: Params, s: Any, batch: dict[str, Any]) -> tuple[Params, Any, Any]:
        loss, grads = jax.value_and_grad(loss_fn)(p, batch, tcfg)
        updates, s = opt.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    step_fn: Any = step
    eval_fn: Any = jax.jit(lambda p, b: loss_fn(p, b, tcfg))
    rng = np.random.default_rng(mcfg.seed)
    val_batches = [
        prep(make_batch(val_graphs[i : i + 16], stats, val_targets[i : i + 16]))
        for i in range(0, len(val_graphs), 16)
    ]
    history: list[dict[str, float]] = []
    best = (math.inf, params)
    bad = 0
    for epoch in range(tcfg.epochs):
        t0 = time.perf_counter()
        order = rng.permutation(len(graphs))
        losses = []
        for i in range(0, len(order), tcfg.batch_graphs):
            idx = order[i : i + tcfg.batch_graphs]
            batch = prep(make_batch([graphs[j] for j in idx], stats, [targets[j] for j in idx]))
            params, state, loss = step_fn(params, state, batch)
            losses.append(float(loss))
        val = (
            float(np.mean([float(eval_fn(params, vb)) for vb in val_batches]))
            if val_batches
            else float(np.mean(losses))
        )
        history.append({"epoch": epoch, "train": float(np.mean(losses)), "val": val})
        if log and (epoch % 10 == 0 or epoch == tcfg.epochs - 1):
            log(
                f"epoch {epoch:3d}  train {np.mean(losses):.4f}  val {val:.4f}  "
                f"({time.perf_counter() - t0:.1f} s)"
            )
        if val < best[0] - 1e-5:
            best = (val, jax.tree_util.tree_map(lambda x: x, params))
            bad = 0
        else:
            bad += 1
            if tcfg.patience is not None and bad >= tcfg.patience:
                break
    return best[1], stats, history


# ----------------------------------------------------------------------------- persistence
def _flatten(params: Params) -> dict[str, NDArray[np.float32]]:
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    out = {}
    for path, leaf in leaves:
        key = "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)
        out[key] = np.asarray(leaf, dtype=np.float32)
    return out


def save_model(
    path: Path, params: Params, stats: Stats, mcfg: ModelConfig, meta: dict[str, Any]
) -> Path:
    """Weights (npz, float32) plus config, feature statistics and metadata (json)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save: Any = np.savez_compressed
    save(path, **_flatten(params))
    info = {"model": asdict(mcfg), "stats": stats.as_dict(), "meta": meta}
    path.with_suffix(".json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return path


def load_model(path: Path) -> tuple[Params, Stats, ModelConfig, dict[str, Any]]:
    """Inverse of :func:`save_model`."""
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    mcfg = ModelConfig(**info["model"])
    template = init_params(mcfg)
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    with np.load(path) as z:
        values = []
        for p, _ in leaves:
            key = "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in p)
            values.append(jnp.asarray(z[key]))
    params = jax.tree_util.tree_unflatten(treedef, values)
    return params, Stats.from_dict(info["stats"]), mcfg, info.get("meta", {})


def predict(
    params: Params, stats: Stats, graphs: Sequence[GraphData]
) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
    """Outcomes in seconds [graphs, losses, N_OUT] and edge queueing (person-s) per graph."""
    batch = make_batch(graphs, stats)
    jb = {k: jnp.asarray(v) for k, v in batch.items()}
    out, edge = _apply_jit(params, jb)
    q = np.asarray(out, dtype=np.float64) * TIME_SCALE
    e = np.expm1(np.asarray(edge, dtype=np.float64)).clip(min=0.0) * 60.0
    per: list[NDArray[np.float64]] = []
    e0 = 0
    for g in graphs:
        per.append(e[e0 : e0 + g.n_edges])
        e0 += g.n_edges
    return q, per


_apply_jit: Any = jax.jit(apply)
