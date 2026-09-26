from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tailsafe.api import app as api
from tailsafe.api.jobs import finite
from tailsafe.sim.cases import corridor_building

client = TestClient(api.app)

SMALL = {"template": "cruciform", "options": {"storeys": 12, "flats_per_wing": 2}}
SPEC: dict[str, Any] = {
    "name": "api-test",
    "time_slot": "weekend_night",
    "fire_level": 5,
    "stair_blockages": [{"stair": "A", "time": {"dist": "constant", "value": 120.0}}],
    "hazard": {"enabled": True},
}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("TAILSAFE_CACHE_DIR", str(tmp_path / "cache"))
    yield


def wait(job: dict[str, Any]) -> dict[str, Any]:
    done = api.JOBS.wait(job["id"], timeout=300)
    assert done.status == "done", done.error
    res = client.get(f"/api/jobs/{job['id']}/result")
    assert res.status_code == 200
    return res.json()  # type: ignore[no-any-return]


def building_id() -> str:
    res = client.post("/api/buildings", json=SMALL)
    assert res.status_code == 200, res.text
    return res.json()["id"]  # type: ignore[no-any-return]


def test_health() -> None:
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "not a substitute" in body["disclaimer"]


def test_params_endpoint_lists_sources() -> None:
    rows = client.get("/api/params").json()
    assert rows and all(r["source"] for r in rows)


def test_templates_and_buildings() -> None:
    names = {t["name"] for t in client.get("/api/templates").json()}
    assert names == {"cruciform", "slab", "twin_core", "care_home"}
    bid = building_id()
    got = client.get(f"/api/buildings/{bid}").json()
    assert got["summary"]["storeys"] == 12
    assert got["building"]["nodes"]
    assert client.get("/api/buildings/nope").status_code == 404
    bad = client.post("/api/buildings", json={"template": "pagoda"})
    assert bad.status_code == 422
    up = client.post("/api/buildings/upload", json=got["building"])
    assert up.status_code == 200 and up.json()["id"] == bid


def test_demo_spec() -> None:
    spec = client.get("/api/specs/demo").json()
    assert spec["fire_level"] == 14 and spec["hazard"]["enabled"]


def test_stress_bottlenecks_replay_flow() -> None:
    bid = building_id()
    job = client.post("/api/stress", json={"building_id": bid, "spec": SPEC, "runs": 20}).json()
    stress = wait(job)
    assert stress["runs"] == 20
    assert set(stress["risk"]) == {"total_time", "self_evacuation_time", "p95_occupant_time"}
    assert len(stress["losses"]["total_time"]) == 20
    assert "p_rset_exceeds_aset" in stress["tenability"]
    assert stress["stair_congestion"]
    json.dumps(stress, allow_nan=False)  # strict JSON for browsers

    again = client.post("/api/stress", json={"building_id": bid, "spec": SPEC, "runs": 20}).json()
    assert again["cached"] and again["status"] == "done"

    # SSE progress stream ends with the final state.
    with client.stream("GET", f"/api/jobs/{job['id']}/events") as r:
        lines = [ln for ln in r.iter_lines() if ln.startswith("data:")]
    assert json.loads(lines[-1][5:])["status"] == "done"

    bn = wait(client.post("/api/bottlenecks", json={"stress_job_id": job["id"]}).json())
    assert bn["ranking"] and bn["ranking"][0]["rank"] == 1
    edges = {e["id"] for e in client.get(f"/api/buildings/{bid}").json()["building"]["edges"]}
    stair_rows = [r for r in bn["ranking"] if r["kind"] == "stair"]
    assert stair_rows and all(r["edges"] and set(r["edges"]) <= edges for r in stair_rows)
    assert all(q["edge"] in edges for q in bn["queues"])

    rp = wait(
        client.post(
            "/api/replay",
            json={
                "building_id": bid,
                "spec": SPEC,
                "index": stress["worst_scenarios"][0],
                "runs": 20,
            },
        ).json()
    )
    assert rp["times"] and len(rp["remaining"]) == len(rp["times"])
    assert set(rp["stair_queues"]) == {"A", "B"}
    assert rp["visibility"] is not None
    assert rp["evacuated"][-1] == rp["summary"]["occupants"] - rp["summary"]["not_evacuated"]
    # The replay is the same scenario as in the stress test (same draws).
    worst = max(stress["losses"]["total_time"])
    assert rp["summary"]["total_time_s"] == pytest.approx(worst, rel=1e-9)


def test_optimize_job() -> None:
    bid = building_id()
    body = {
        "building_id": bid,
        "spec": SPEC,
        "n_scenarios": 10,
        "confirm_scenarios": 12,
        "levers": ["lifts", "wardens"],
        "max_wardens": 1,
        "cmaes_iterations": 0,
    }
    out = wait(client.post("/api/optimize", json=body).json())
    assert out["plan_description"]
    assert len(out["before_after"]["total_time"]["before"]) == 12
    # Replaying the worst confirmation scenario reproduces it exactly, with and
    # without the plan (same seed and batch layout -> identical draws).
    rp = out["replay"]
    before = out["before_after"]["total_time"]["before"]
    k = before.index(max(before))
    for spec_key, series in (("baseline_spec", "before"), ("plan_spec", "after")):
        replay = wait(
            client.post(
                "/api/replay",
                json={
                    "building_id": bid,
                    "spec": rp[spec_key],
                    "index": rp["worst_index"],
                    "seed": rp["seed"],
                    "runs": rp["runs"],
                    "batch_size": rp["batch_size"],
                },
            ).json()
        )
        expected = out["before_after"]["total_time"][series][k]
        assert replay["summary"]["total_time_s"] == pytest.approx(expected, rel=1e-9)
    bad = client.post("/api/optimize", json={**body, "levers": ["teleport"]})
    assert bad.status_code == 422


def test_errors() -> None:
    assert client.get("/api/jobs/nope").status_code == 404
    bid = building_id()
    bad_spec = {**SPEC, "stair_blockages": [{"stair": "Z"}]}
    res = client.post("/api/stress", json={"building_id": bid, "spec": bad_spec, "runs": 10})
    assert res.status_code == 422
    assert client.post("/api/bottlenecks", json={"stress_job_id": "nope"}).status_code == 404


def test_finite_replaces_infinities() -> None:
    assert finite({"a": [1.0, float("inf"), float("nan")], "b": "x"}) == {
        "a": [1.0, None, None],
        "b": "x",
    }


def test_micro_replay_and_floor_frames() -> None:
    bid = building_id()
    body = {"building_id": bid, "spec": SPEC, "index": 2}
    job = client.post("/api/micro", json=body).json()
    out = wait(job)
    assert out["summary"]["not_out"] == 0
    cmp = out["comparison"]
    assert cmp["walkers"] > 0 and cmp["micro"]["last_s"] > cmp["micro"]["p50_s"] > 0
    assert len(out["people_on_level"]) == out["frames"]
    assert out["rooms"] and all(r["polygon"] for r in out["rooms"])
    json.dumps(out, allow_nan=False)

    level = client.get(f"/api/micro/{job['id']}/level/3").json()
    assert len(level["frames"]) == len(level["times"]) == out["frames"]
    first = level["frames"][0]
    assert first and all(len(row) == 4 for row in first)
    # everyone on 3/F at the start is accounted for in the per-level counts
    k = out["levels"].index(3)
    assert len(first) == out["people_on_level"][0][k]
    assert client.get("/api/micro/nope/level/3").status_code == 404

    corridor = corridor_building().model_dump(mode="json", exclude_none=True)
    graph_only = client.post("/api/buildings/upload", json=corridor).json()
    bad = client.post("/api/micro", json={"building_id": graph_only["id"], "spec": {"name": "x"}})
    assert bad.status_code == 422
