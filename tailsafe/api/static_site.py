"""Export a static copy of the web app's data for the browser version.

The browser version is the same web UI served as static files (GitHub Pages),
without a Python server. This module drives the real API in-process with
exactly the requests the UI sends when a visitor follows the standard path
for each building type — default building, reference scenario, stress test,
bottlenecks, optimised plan, replays, briefing — and stores every response
under a hash of a canonical form of the request. The browser computes the
same hash (``web/src/static/key.ts``) and reads the file instead of calling
the server. Requests outside the recorded path get a friendly "run it
locally" message in the UI.

The surrogate is not recorded: its weights and each building's graph
features are exported so the what-if screen runs the network in the
browser, for any settings.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from tailsafe.building.model import Building
from tailsafe.config import get_params

LOSSES = ("total_time", "self_evacuation_time", "p95_occupant_time")
LEVERS = ["lifts", "hold_open", "stair_assignment", "phasing", "wardens"]


# ----------------------------------------------------------------------------- request keys
def canon(x: Any) -> str:
    """Canonical JSON (sorted keys, integral numbers without a decimal point).

    Mirrors ``canon`` in ``web/src/static/key.ts``: both sides must produce
    the same string for the same request body.
    """
    if x is None:
        return "null"
    if x is True:
        return "true"
    if x is False:
        return "false"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            return "null"
        if x.is_integer() and abs(x) < 1e15:
            return str(int(x))
        return repr(x)
    if isinstance(x, str):
        return json.dumps(x, ensure_ascii=False)
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(canon(v) for v in x) + "]"
    if isinstance(x, dict):
        items = sorted(x.items())
        return (
            "{"
            + ",".join(json.dumps(k, ensure_ascii=False) + ":" + canon(v) for k, v in items)
            + "}"
        )
    raise TypeError(f"cannot canonicalise {type(x).__name__}")


def _fnv(data: bytes, h: int) -> int:
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def body_for_key(path: str, body: Any) -> Any:
    """The part of a request body that identifies it (large bodies are reduced)."""
    if path == "/api/briefing":
        return {
            "building_id": body["building_id"],
            "bottlenecks": body.get("bottlenecks") is not None,
            "optimization": body.get("optimization") is not None,
            "llm": body.get("llm"),
        }
    if path == "/api/briefing/pdf":
        return {"markdown": body["markdown"]}
    return body


def request_key(method: str, path: str, body: Any = None) -> str:
    """Hash of a request (16 hex digits), as computed by the browser."""
    text = f"{method} {path} {canon(body_for_key(path, body)) if body is not None else ''}"
    data = text.encode("utf-8")
    return f"{_fnv(data, 0x811C9DC5):08x}{_fnv(data, 0x050C5D1F):08x}"


def fit_spec(spec: dict[str, Any], view: dict[str, Any]) -> dict[str, Any]:
    """The UI's ``fitSpec``: the reference scenario adapted to a building."""
    b = view["building"]
    top = max(lv["index"] for lv in b["levels"])
    stairs = {s["id"] for s in b["stairs"]}
    non_ff = sum(1 for lf in b["lifts"] if not lf.get("firefighting", False))
    return {
        **spec,
        "fire_level": min(spec["fire_level"], top) if spec["fire_level"] is not None else None,
        "stair_blockages": [x for x in spec["stair_blockages"] if x["stair"] in stairs],
        "lifts_out_of_service": min(spec["lifts_out_of_service"], len(b["lifts"])),
        "evacuation_lifts": bool(spec["evacuation_lifts"] and non_ff > 0),
        "stair_assignment": {},
        "phased_release": {},
        "warden_levels": [lv for lv in spec["warden_levels"] if lv <= top],
        "capacity_multipliers": {},
    }


# ----------------------------------------------------------------------------- recorder
class Recorder:
    """Calls the API in-process and writes each response under its request key."""

    def __init__(self, out: Path, log: Callable[[str], None] | None = None) -> None:
        from fastapi.testclient import TestClient

        from tailsafe.api import app as api

        self.api = api
        self.client = TestClient(api.app)
        self.out = out
        (out / "r").mkdir(parents=True, exist_ok=True)
        self.log = log or (lambda _m: None)
        self.count = 0

    def save(self, key: str, data: Any, suffix: str = "json") -> None:
        """Write one response (JSON, or bytes for other suffixes)."""
        path = self.out / "r" / f"{key}.{suffix}"
        if suffix == "json":
            path.write_text(
                json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8"
            )
        else:
            path.write_bytes(data)
        self.count += 1

    def get(self, path: str, record_path: str | None = None) -> Any:
        """GET and record (under ``record_path`` when the browser asks a different path)."""
        res = self.client.get(path)
        res.raise_for_status()
        data = res.json()
        self.save(request_key("GET", record_path or path), data)
        return data

    def post(self, path: str, body: dict[str, Any], key_body: dict[str, Any] | None = None) -> Any:
        """POST and record (keyed by ``key_body`` when the browser sends different ids)."""
        res = self.client.post(path, json=body)
        res.raise_for_status()
        data = res.json()
        self.save(request_key("POST", path, key_body if key_body is not None else body), data)
        return data

    def job(
        self, path: str, body: dict[str, Any], key_body: dict[str, Any] | None = None
    ) -> tuple[str, str, Any]:
        """Run a job and record the submission and its result.

        Returns the server's job id, the recorded job id and the result.
        """
        kb = key_body if key_body is not None else body
        res = self.client.post(path, json=body)
        res.raise_for_status()
        job = res.json()
        done = self.api.JOBS.wait(job["id"], timeout=3600)
        if done.status != "done":
            raise RuntimeError(f"{path} failed: {done.error}")
        rec_id = "r" + request_key("POST", path, kb)
        public = {**done.public(), "id": rec_id, "cached": True}
        self.save(request_key("POST", path, kb), public)
        result = self.client.get(f"/api/jobs/{job['id']}/result").json()
        self.save(request_key("GET", f"/api/jobs/{rec_id}/result"), result)
        return job["id"], rec_id, result


def _replay_body(
    bid: str, spec: dict[str, Any], index: int, seed: int, runs: int, batch: int = 100
) -> dict[str, Any]:
    return {
        "building_id": bid,
        "spec": spec,
        "index": index,
        "seed": seed,
        "runs": runs,
        "batch_size": batch,
    }


def record_building(
    rec: Recorder,
    template: dict[str, Any],
    reference: dict[str, Any],
    *,
    runs: int = 300,
    micro: bool = True,
) -> dict[str, Any]:
    """The standard path through the UI for one building type."""
    log = rec.log
    name = template["name"]
    log(f"[{name}] building")
    view = rec.post("/api/buildings", {"template": name, "options": template["options"]})
    bid = view["id"]
    spec = fit_spec(reference, view)

    log(f"[{name}] stress test ({runs} runs)")
    stress_body = {"building_id": bid, "spec": spec, "runs": runs, "seed": 0}
    stress_real, stress_rec, stress = rec.job("/api/stress", stress_body)

    log(f"[{name}] bottlenecks")
    bn_real, _, bottlenecks = rec.job(
        "/api/bottlenecks",
        {"stress_job_id": stress_real, "loss": "p95_occupant_time", "factor": 1.5},
        key_body={"stress_job_id": stress_rec, "loss": "p95_occupant_time", "factor": 1.5},
    )
    del bn_real

    log(f"[{name}] optimisation")
    opt_body = {
        "building_id": bid,
        "spec": spec,
        "objective": {"kind": "cvar", "loss": "total_time"},
        "n_scenarios": 60,
        "confirm_scenarios": 200,
        "levers": LEVERS,
        "max_wardens": 2,
        "cmaes_iterations": 3,
        "seed": 1,
    }
    _, _, optimization = rec.job("/api/optimize", opt_body)

    worst = int(stress["worst_scenarios"][0]) if stress["worst_scenarios"] else 0
    log(f"[{name}] replays (scenario {worst} and the median)")
    for idx in {worst, int(stress["median_scenario"])}:
        rec.job("/api/replay", _replay_body(bid, spec, idx, 0, runs))
    rp = optimization.get("replay")
    if rp:
        for s in (rp["baseline_spec"], rp["plan_spec"]):
            rec.job(
                "/api/replay",
                _replay_body(bid, s, rp["worst_index"], rp["seed"], rp["runs"], rp["batch_size"]),
            )

    if micro:
        log(f"[{name}] person-by-person replay of scenario {worst}")
        mbody = {"building_id": bid, "spec": spec, "index": worst, "seed": 0, "batch_size": 100}
        m_real, m_rec, mres = rec.job("/api/micro", mbody)
        for lv in mres.get("levels", []):
            rec.get(f"/api/micro/{m_real}/level/{lv}", record_path=f"/api/micro/{m_rec}/level/{lv}")

    log(f"[{name}] briefings")
    for with_bn in (False, True):
        for with_opt in (False, True):
            for llm in (None, False):
                body = {
                    "building_id": bid,
                    "stress": stress,
                    "bottlenecks": bottlenecks if with_bn else None,
                    "optimization": optimization if with_opt else None,
                    "llm": llm,
                }
                brief = rec.post("/api/briefing", body)
                dist = (
                    {
                        "Baseline": optimization["before_after"]["total_time"]["before"],
                        "With plan": optimization["before_after"]["total_time"]["after"],
                    }
                    if with_opt
                    else {"Baseline": stress["losses"]["total_time"]}
                )
                pdf_body = {"markdown": brief["markdown"], "distributions": dist}
                res = rec.client.post("/api/briefing/pdf", json=pdf_body)
                res.raise_for_status()
                rec.save(request_key("POST", "/api/briefing/pdf", pdf_body), res.content, "pdf")
    out: dict[str, Any] = view
    return out


# ----------------------------------------------------------------------------- surrogate
def model_payload(weights: Path) -> tuple[dict[str, Any], bytes]:
    """The network for the browser: a JSON description and float32 weights."""
    import inspect

    from tailsafe.building.templates import TEMPLATES
    from tailsafe.sim.meso import SimConfig
    from tailsafe.surrogate.data import SHARE_65_RANGE, TYPOLOGIES

    info = json.loads(weights.with_suffix(".json").read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    with np.load(weights) as z:
        for key in sorted(z.files):
            arr = np.ascontiguousarray(z[key], dtype="<f4")
            manifest[key] = {"offset": offset, "shape": list(arr.shape)}
            chunks.append(arr.tobytes())
            offset += arr.size
    p = get_params()
    mix = p["population.age_mix.residential"].categories
    older = float(sum(pr for k, pr in zip(mix[0], mix[1], strict=True) if k in ("older", "frail")))
    defaults = {
        name: {
            k: v.default
            for k, v in inspect.signature(fn).parameters.items()
            if k != "params" and v.default is not inspect.Parameter.empty
        }
        for name, fn in TEMPLATES.items()
    }
    model = {
        "config": info["model"],
        "stats": info["stats"],
        "meta": info["meta"],
        "weights": manifest,
        "horizon_s": SimConfig().t_max,
        "defaults": {
            "share_65_plus": older,
            "share_80_plus_of_65_plus": p.scalar("population.frail_share_of_65_plus"),
            "vacancy_rate": p.scalar("population.vacancy_rate"),
        },
        "typologies": TYPOLOGIES,
        "template_defaults": defaults,
        "share_65_range": list(SHARE_65_RANGE),
    }
    return model, b"".join(chunks)


def graph_payload(b: Building) -> dict[str, Any]:
    """One building's graph features for the browser (see ``surrogate/features.py``)."""
    from tailsafe.surrogate.features import static_graph

    g = static_graph(b, get_params())
    forward = [eid for eid, fwd in zip(g.edge_ids, g.edge_forward, strict=True) if fwd]
    labels = {eid: b.describe_edge(eid) for eid in forward}

    def r5(a: Any) -> Any:
        return np.round(np.asarray(a, dtype=np.float64), 5).tolist()

    return {
        "node_level": g.node_level.tolist(),
        "node_base": r5(g.node_base),
        "edge_src": g.edge_src.tolist(),
        "edge_dst": g.edge_dst.tolist(),
        "edge_base": r5(g.edge_base),
        "edge_ids": g.edge_ids,
        "edge_forward": g.edge_forward.tolist(),
        "edge_stair": g.edge_stair,
        "landing_stair": g.landing_stair,
        "lift_lobby": g.lift_lobby.tolist(),
        "n_levels": g.n_levels,
        "n_units": g.n_units,
        "occupants": g.occupants,
        "care_share": g.care_share,
        "n_lifts": g.n_lifts,
        "n_ff_lifts": g.n_ff_lifts,
        "n_stairs": g.n_stairs,
        "labels": labels,
        "metadata": {
            "generator": b.metadata.get("generator"),
            "args": b.metadata.get("args", {}),
        },
    }


def export_surrogate(out: Path, views: list[dict[str, Any]]) -> None:
    """Weights (float32 binary + description) and each building's graph features."""
    from tailsafe.surrogate.predictor import DEFAULT_WEIGHTS

    sdir = out / "surrogate"
    sdir.mkdir(parents=True, exist_ok=True)
    model, blob = model_payload(DEFAULT_WEIGHTS)
    (sdir / "weights.bin").write_bytes(blob)
    (sdir / "model.json").write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    for view in views:
        b = Building.model_validate(view["building"])
        (sdir / f"{view['id']}.json").write_text(
            json.dumps(graph_payload(b), separators=(",", ":")), encoding="utf-8"
        )


def export_static_site(
    out: Path,
    *,
    templates: list[str] | None = None,
    runs: int = 300,
    micro: bool = True,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Record the standard path for each building type and export the surrogate."""
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(out, log)
    rec.get("/api/health")
    all_templates = rec.get("/api/templates")
    reference = rec.get("/api/specs/reference")
    chosen = [t for t in all_templates if templates is None or t["name"] in templates]
    views = [record_building(rec, t, reference, runs=runs, micro=micro) for t in chosen]
    export_surrogate(out, views)
    manifest = {
        "buildings": {v["id"]: v["summary"]["name"] for v in views},
        "templates": [t["name"] for t in chosen],
        "runs": runs,
        "responses": rec.count,
        "source_commit": os.environ.get("GITHUB_SHA"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest
