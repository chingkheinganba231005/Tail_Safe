"""FastAPI application exposing TailSafe to the web frontend.

Long computations run as background jobs (:mod:`tailsafe.api.jobs`): submit,
then follow progress with ``GET /api/jobs/{id}`` or the server-sent events
stream ``GET /api/jobs/{id}/events``, and fetch ``/api/jobs/{id}/result``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from tailsafe import DISCLAIMER, __version__
from tailsafe.api.jobs import JobManager, cache_dir, cache_key, finite
from tailsafe.api.views import (
    building_view,
    micro_level_view,
    micro_view,
    replay_view,
    stress_view,
)
from tailsafe.building.model import Building
from tailsafe.building.templates import TEMPLATES, generate
from tailsafe.building.validate import check_building
from tailsafe.config import get_params
from tailsafe.optimize.search import LEVERS, Objective, OptimizeConfig, optimize
from tailsafe.scenarios.montecarlo import MCConfig, MCResult, run_monte_carlo
from tailsafe.scenarios.sampler import ScenarioSampler, scenario_uniforms
from tailsafe.scenarios.spec import ScenarioSpec, demo_spec
from tailsafe.sim.meso import SimConfig, run_meso
from tailsafe.sim.network import compile_network
from tailsafe.vision.detect import PlanDetection, Scale

app = FastAPI(
    title="TailSafe API",
    version=__version__,
    description="Tail-risk evacuation stress-testing for high-rise Hong Kong. " + DISCLAIMER,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS = JobManager()
_BUILDINGS: dict[str, Building] = {}
MAX_RUNS = 5000


# ============================================================================ helpers
def _building_dir() -> Path:
    return cache_dir().parent / "buildings"


def _store(b: Building) -> str:
    bid = b.digest()[:16]
    _BUILDINGS[bid] = b
    path = _building_dir() / f"{bid}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(b.model_dump_json(exclude_none=True), encoding="utf-8")
    return bid


def _building(bid: str) -> Building:
    if bid not in _BUILDINGS:
        path = _building_dir() / f"{bid}.json"
        if not path.exists():
            raise HTTPException(404, f"unknown building {bid}")
        _BUILDINGS[bid] = Building.model_validate_json(path.read_text(encoding="utf-8"))
    return _BUILDINGS[bid]


def _check_spec(b: Building, spec: ScenarioSpec) -> None:
    try:
        ScenarioSampler(b, spec)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


# ============================================================================ models
class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BuildingRequest(_Req):
    """Generate a building from a procedural template."""

    template: str = "cruciform"
    options: dict[str, Any] = Field(default_factory=dict)


class StressRequest(_Req):
    """Monte Carlo stress test."""

    building_id: str
    spec: ScenarioSpec = Field(default_factory=demo_spec)
    runs: int = Field(default=300, ge=10, le=MAX_RUNS)
    seed: int = Field(default=0, ge=0)


class BottleneckRequest(_Req):
    """Counterfactual bottleneck ranking of a finished stress test."""

    stress_job_id: str
    loss: str = "p95_occupant_time"
    factor: float = Field(default=1.5, gt=1.0, le=5.0)


class OptimizeRequest(_Req):
    """Search for an operational plan."""

    building_id: str
    spec: ScenarioSpec = Field(default_factory=demo_spec)
    objective: Objective = Field(default_factory=Objective)
    n_scenarios: int = Field(default=60, ge=10, le=1000)
    confirm_scenarios: int = Field(default=200, ge=10, le=MAX_RUNS)
    levers: list[str] = Field(default_factory=lambda: list(LEVERS))
    max_wardens: int = Field(default=2, ge=0, le=10)
    cmaes_iterations: int = Field(default=3, ge=0, le=20)
    seed: int = Field(default=1, ge=0)


class ReplayRequest(_Req):
    """Re-simulate one scenario with full time series."""

    building_id: str
    spec: ScenarioSpec = Field(default_factory=demo_spec)
    index: int = Field(default=0, ge=0)
    seed: int = Field(default=0, ge=0)
    runs: int = Field(default=300, ge=1, le=MAX_RUNS, description="Runs of the stress test")
    batch_size: int = Field(
        default=100, ge=1, le=1000, description="LHS batch size of the stress test"
    )


# ============================================================================ basics
@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe with version and the responsible-use notice."""
    return {"status": "ok", "version": __version__, "disclaimer": DISCLAIMER}


@app.get("/api/params")
def list_params() -> list[dict[str, Any]]:
    """Every registry parameter with its unit, distribution family and source."""
    return [
        {
            "path": p.path,
            "dist": p.dist,
            "unit": p.unit,
            "source": p.source,
            "assumption": p.is_assumption,
        }
        for p in get_params().leaves()
    ]


@app.get("/api/templates")
def templates() -> list[dict[str, Any]]:
    """Procedural templates and their options (with defaults)."""
    out = []
    for name, fn in TEMPLATES.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        opts = {
            k: v.default
            for k, v in inspect.signature(fn).parameters.items()
            if k != "params" and v.default is not inspect.Parameter.empty
        }
        out.append({"name": name, "description": doc, "options": opts})
    return out


@app.get("/api/specs/demo")
def get_demo_spec() -> dict[str, Any]:
    """The pitch scenario specification."""
    return demo_spec().model_dump(mode="json")


# ============================================================================ buildings
@app.post("/api/buildings")
def create_building(req: BuildingRequest) -> dict[str, Any]:
    """Generate a building from a template."""
    try:
        b = generate(req.template, **req.options)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc
    _store(b)
    return finite(building_view(b))  # type: ignore[no-any-return]


@app.post("/api/buildings/upload")
def upload_building(building: dict[str, Any]) -> dict[str, Any]:
    """Register a building JSON (validated against the schema and semantic rules)."""
    try:
        b = Building.model_validate(building)
        check_building(b)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    _store(b)
    return finite(building_view(b))  # type: ignore[no-any-return]


@app.get("/api/buildings/{bid}")
def get_building(bid: str) -> dict[str, Any]:
    """A registered building."""
    return finite(building_view(_building(bid)))  # type: ignore[no-any-return]


# ============================================================================ jobs
def _mc_dir(key: str) -> Path:
    return cache_dir() / f"mc-{key}"


@app.post("/api/stress")
def submit_stress(req: StressRequest) -> dict[str, Any]:
    """Start a Monte Carlo stress test."""
    b = _building(req.building_id)
    _check_spec(b, req.spec)
    p = get_params()
    key = cache_key(b.digest(), req.spec.model_dump(mode="json"), req.runs, req.seed, p.digest)

    def work(progress: Any) -> dict[str, Any]:
        res = run_monte_carlo(
            b, req.spec, MCConfig(n_runs=req.runs, seed=req.seed), params=p, progress=progress
        )
        res.save(_mc_dir(key))
        JOBS.objects[job.id] = res
        return finite(stress_view(res))  # type: ignore[no-any-return]

    job = JOBS.submit("stress", key, work)
    return job.public()


def _stress_result(job_id: str) -> MCResult:
    job = JOBS.get(job_id)
    if job is None or job.kind != "stress":
        raise HTTPException(404, f"unknown stress job {job_id}")
    if job.status != "done":
        raise HTTPException(409, "stress test not finished")
    res = JOBS.objects.get(job_id)
    if res is None:
        path = _mc_dir(job.key)
        if not path.exists():
            raise HTTPException(410, "stress result no longer available; re-run it")
        res = MCResult.load(path)
        JOBS.objects[job_id] = res
    assert isinstance(res, MCResult)
    return res


@app.post("/api/bottlenecks")
def submit_bottlenecks(req: BottleneckRequest) -> dict[str, Any]:
    """Rank bottlenecks of a finished stress test (re-runs scenarios)."""
    from tailsafe.analysis.bottlenecks import attribute_bottlenecks

    res = _stress_result(req.stress_job_id)
    stress_key = JOBS.get(req.stress_job_id).key  # type: ignore[union-attr]
    key = cache_key(stress_key, req.loss, req.factor)

    def work(progress: Any) -> dict[str, Any]:
        progress(0, 1)
        table = attribute_bottlenecks(res, loss=req.loss, factor=req.factor)
        progress(1, 1)
        return finite(table)  # type: ignore[no-any-return]

    return JOBS.submit("bottlenecks", key, work).public()


@app.post("/api/optimize")
def submit_optimize(req: OptimizeRequest) -> dict[str, Any]:
    """Search for the plan that shrinks the tail most, and confirm it."""
    b = _building(req.building_id)
    _check_spec(b, req.spec)
    unknown = set(req.levers) - set(LEVERS)
    if unknown:
        raise HTTPException(422, f"unknown levers {sorted(unknown)}")
    p = get_params()
    key = cache_key(b.digest(), req.model_dump(mode="json", exclude={"building_id"}), p.digest)

    def work(progress: Any) -> dict[str, Any]:
        count = {"n": 0}

        def log(msg: str) -> None:
            count["n"] += 1
            progress(count["n"], 0)
            job.message = msg

        cfg = OptimizeConfig(
            n_scenarios=req.n_scenarios,
            confirm_scenarios=req.confirm_scenarios,
            levers=tuple(req.levers),
            max_wardens=req.max_wardens,
            cmaes_iterations=req.cmaes_iterations,
            seed=req.seed,
        )
        out = optimize(b, req.spec, req.objective, cfg, params=p, log=log)
        summary = out.summary(b)
        summary["before_after"] = {
            loss: {
                "before": [float(x) for x in out.confirm_baseline.loss(loss)],
                "after": [float(x) for x in out.confirm_best.loss(loss)],
            }
            for loss in ("total_time", "self_evacuation_time", "p95_occupant_time")
        }
        # Everything needed to replay the worst confirmation scenario with and
        # without the plan (same seed and batch layout, so identical draws).
        worst = int(np.argmax(out.confirm_baseline.loss("total_time")))
        summary["replay"] = {
            "baseline_spec": req.spec.model_dump(mode="json"),
            "plan_spec": out.best.plan.apply(req.spec, b).model_dump(mode="json"),
            "seed": cfg.confirm_seed,
            "runs": cfg.confirm_scenarios,
            "batch_size": min(100, cfg.confirm_scenarios),
            "worst_index": int(out.confirm_baseline.runs[worst].index),
        }
        summary["disclaimer"] = DISCLAIMER
        return finite(summary)  # type: ignore[no-any-return]

    job = JOBS.submit("optimize", key, work)
    return job.public()


@app.post("/api/replay")
def submit_replay(req: ReplayRequest) -> dict[str, Any]:
    """Re-simulate one scenario with time series for the 3D view."""
    b = _building(req.building_id)
    _check_spec(b, req.spec)
    p = get_params()
    key = cache_key(
        b.digest(),
        req.spec.model_dump(mode="json"),
        req.index,
        req.seed,
        req.batch_size,
        p.digest,
    )

    def work(progress: Any) -> dict[str, Any]:
        progress(0, 1)
        cfg = MCConfig(n_runs=req.runs, seed=req.seed, batch_size=req.batch_size)
        u = scenario_uniforms(req.seed, req.index, 1, lhs=cfg.lhs, batch_size=cfg.batch_size)[0]
        sc = ScenarioSampler(b, req.spec, p).sample(req.seed, req.index, u, keep_hazard_fields=True)
        net = compile_network(b, p)
        res = run_meso(
            net,
            sc.population,
            sc.sim,
            SimConfig(record_series=True, record_interval=10.0),
            params=p,
        )
        progress(1, 1)
        return finite(replay_view(res, sc.info))  # type: ignore[no-any-return]

    return JOBS.submit("replay", key, work).public()


class MicroRequest(_Req):
    """Replay one scenario person by person (and compare with the meso engine)."""

    building_id: str
    spec: ScenarioSpec = Field(default_factory=demo_spec)
    index: int = Field(default=0, ge=0)
    seed: int = Field(default=0, ge=0)
    batch_size: int = Field(default=100, ge=1, le=1000)


def _micro_path(key: str) -> Path:
    return cache_dir() / f"micro-{key}.npz"


@app.post("/api/micro")
def submit_micro(req: MicroRequest) -> dict[str, Any]:
    """Micro replay of one scenario; frames are served per floor afterwards."""
    from tailsafe.sim.micro import micro_problems, run_micro

    b = _building(req.building_id)
    _check_spec(b, req.spec)
    problems = micro_problems(b)
    if problems:
        raise HTTPException(422, "; ".join(problems))
    p = get_params()
    key = cache_key(
        "micro",
        b.digest(),
        req.spec.model_dump(mode="json"),
        req.index,
        req.seed,
        req.batch_size,
        p.digest,
    )

    def work(progress: Any) -> dict[str, Any]:
        progress(0, 2)
        u = scenario_uniforms(req.seed, req.index, 1, batch_size=req.batch_size)[0]
        sc = ScenarioSampler(b, req.spec, p).sample(req.seed, req.index, u)
        net = compile_network(b, p)
        me = run_meso(net, sc.population, sc.sim, params=p)
        progress(1, 2)
        mi = run_micro(
            net, sc.population, sc.sim, params=p, meso=me, seed=req.seed, index=req.index
        )
        _micro_path(key).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            _micro_path(key),
            times=mi.frame_times,
            level=mi.frame_level,
            x=np.round(mi.frame_x * 10).astype(np.int16),
            y=np.round(mi.frame_y * 10).astype(np.int16),
            state=mi.frame_state,
        )
        progress(2, 2)
        return finite(micro_view(mi, me, sc.info))  # type: ignore[no-any-return]

    return JOBS.submit("micro", key, work).public()


@app.get("/api/micro/{job_id}/level/{level}")
def micro_level(job_id: str, level: int) -> dict[str, Any]:
    """Positions on one floor for every frame of a finished micro replay."""
    job = JOBS.get(job_id)
    if job is None or job.kind != "micro":
        raise HTTPException(404, f"unknown micro job {job_id}")
    if job.status != "done":
        raise HTTPException(409, "micro replay not finished")
    path = _micro_path(job.key)
    if not path.exists():
        raise HTTPException(410, "replay frames no longer available; run it again")
    cached = JOBS.objects.get(f"micro-frames-{job_id}")
    if cached is None:
        with np.load(path) as z:
            cached = {k: z[k] for k in z.files}
        JOBS.objects[f"micro-frames-{job_id}"] = cached
    return micro_level_view(cached, level)


# ============================================================================ floor plans
MAX_IMAGE_CHARS = 30_000_000  # ~22 MB of image data as base64


class VisionDetectRequest(_Req):
    """A plan image (data URL) and, optionally, a reference line for the scale."""

    image: str = Field(max_length=MAX_IMAGE_CHARS)
    scale: Scale | None = None


class VisionBuildRequest(_Req):
    """A (corrected) detection and how many storeys to stack."""

    detection: PlanDetection
    storeys: int = Field(default=10, ge=1, le=80)
    floor_height: float | None = Field(default=None, gt=2.0, lt=8.0)
    name: str = Field(default="Building from floor plan", max_length=200)


@app.post("/api/vision/detect")
def vision_detect(req: VisionDetectRequest) -> dict[str, Any]:
    """Read walls, doorways, rooms and stairs from a plan image."""
    from tailsafe.vision.detect import detect_plan
    from tailsafe.vision.graph import detection_summary
    from tailsafe.vision.raster import load_image

    try:
        img = load_image(req.image)
        det = detect_plan(img, req.scale)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    out: dict[str, Any] = {
        "detection": det.model_dump(mode="json"),
        "summary": detection_summary(det),
    }
    if req.image.startswith("data:application/pdf"):
        from tailsafe.vision.raster import to_data_url
        from tailsafe.vision.synth import to_png_bytes

        out["preview"] = to_data_url(to_png_bytes(img))  # browsers can't draw PDF pages in <img>
    return out


@app.post("/api/vision/build")
def vision_build(req: VisionBuildRequest) -> dict[str, Any]:
    """Stack the corrected floor into a building and register it."""
    from tailsafe.vision.graph import plan_to_building

    try:
        b = plan_to_building(
            req.detection, storeys=req.storeys, floor_height=req.floor_height, name=req.name
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    _store(b)
    return finite(building_view(b))  # type: ignore[no-any-return]


@app.get("/api/vision/sample")
def vision_sample(template: str = "cruciform", level: int = 1) -> dict[str, Any]:
    """A synthetic plan rendered from a template, to try the reader without a drawing."""
    from tailsafe.vision.raster import to_data_url
    from tailsafe.vision.synth import render_plan, to_png_bytes

    if template not in TEMPLATES:
        raise HTTPException(422, f"unknown template {template!r}")
    b = generate(template)
    if level not in {lv.index for lv in b.levels}:
        raise HTTPException(422, f"{template} has no level {level}")
    img, truth = render_plan(b, level, noise=0.02, blur=0.4, seed=level)
    return {
        "image": to_data_url(to_png_bytes(img)),
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "m_per_px": truth.m_per_px,
        "note": "Synthetic plan rendered from the procedural template (for trying the reader).",
    }


# ============================================================================ surrogate
class SurrogateRequest(_Req):
    """Instant estimate for a building and scenario."""

    building_id: str
    spec: ScenarioSpec = Field(default_factory=demo_spec)


_SURROGATE: dict[str, Any] = {}


def _surrogate() -> Any:
    if "model" not in _SURROGATE:
        try:
            from tailsafe.surrogate.predictor import Surrogate

            _SURROGATE["model"] = Surrogate()
        except (ImportError, FileNotFoundError) as exc:
            raise HTTPException(
                503,
                "the surrogate is not available here: "
                f"{exc}. Install the optional extra with pip install 'tailsafe[surrogate]'.",
            ) from exc
    return _SURROGATE["model"]


@app.post("/api/surrogate/predict")
def surrogate_predict(req: SurrogateRequest) -> dict[str, Any]:
    """Evacuation-time quantiles from the graph surrogate (milliseconds, not minutes)."""
    b = _building(req.building_id)
    _check_spec(b, req.spec)
    out = _surrogate().predict(b, req.spec)
    out["disclaimer"] = DISCLAIMER
    return finite(out)  # type: ignore[no-any-return]


@app.get("/api/jobs")
def list_jobs() -> list[dict[str, Any]]:
    """Recent jobs (without results)."""
    return [j.public() for j in JOBS.list()]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    """Status and progress of a job."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id}")
    return job.public()


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> dict[str, Any]:
    """Result of a finished job."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id}")
    if job.status == "error":
        raise HTTPException(500, job.error or "job failed")
    if job.status != "done" or job.result is None:
        raise HTTPException(409, "job not finished")
    return job.result


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """Server-sent events with job progress until it finishes."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job {job_id}")

    async def stream() -> AsyncIterator[str]:
        last = None
        while True:
            state = json.dumps(job.public())
            if state != last:
                yield f"data: {state}\n\n"
                last = state
            if job.status in ("done", "error"):
                break
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ============================================================================ frontend
_WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
if _WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=_WEB_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the built web app."""
        return FileResponse(_WEB_DIST / "index.html")
