from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tailsafe.api.static_site import canon, export_static_site, fit_spec, request_key

# The same reference hashes are checked in web/test/key.test.ts: the browser
# and the recorder must agree on every request key.
CASES = [
    ("GET", "/api/health", None, "6946f38b29590789"),
    (
        "POST",
        "/api/stress",
        {
            "building_id": "abc",
            "spec": {
                "name": "reference",
                "share_65_plus": 0.22,
                "fire_level": 14,
                "stair_blockages": [
                    {"stair": "A", "time": {"dist": "constant", "value": 240.0, "min": None}}
                ],
                "hazard": {},
                "note": "Stair A → G/F ₉₅",
            },
            "runs": 300,
            "seed": 0,
        },
        "37311fbae03474f0",
    ),
    (
        "POST",
        "/api/briefing",
        {
            "building_id": "abc",
            "stress": {"x": 1},
            "bottlenecks": None,
            "optimization": {"a": 1},
            "llm": False,
        },
        "967d0cd1bf6aec77",
    ),
    (
        "POST",
        "/api/briefing/pdf",
        {"markdown": "# Title\n- 219.2 min", "distributions": {"Baseline": [1.5, 2.0]}},
        "e4da5465ff86ff67",
    ),
]


@pytest.mark.parametrize(("method", "path", "body", "expected"), CASES)
def test_request_keys_match_the_browser(
    method: str, path: str, body: object, expected: str
) -> None:
    assert request_key(method, path, body) == expected


def test_canon() -> None:
    assert canon({"z": 2, "a": [1.0, 0.5, None, True]}) == '{"a":[1,0.5,null,true],"z":2}'
    assert canon(-0.0) == "0"
    assert canon(float("inf")) == "null"


def test_fit_spec_matches_the_building() -> None:
    view = {
        "building": {
            "levels": [{"index": i} for i in range(5)],
            "stairs": [{"id": "A"}],
            "lifts": [{"firefighting": True}],
        }
    }
    spec = {
        "fire_level": 14,
        "stair_blockages": [{"stair": "A"}, {"stair": "B"}],
        "lifts_out_of_service": 3,
        "evacuation_lifts": True,
        "warden_levels": [2, 9],
    }
    out = fit_spec(spec, view)
    assert out["fire_level"] == 4
    assert [b["stair"] for b in out["stair_blockages"]] == ["A"]
    assert out["lifts_out_of_service"] == 1
    assert out["evacuation_lifts"] is False  # only a firefighting lift
    assert out["warden_levels"] == [2]
    assert out["phased_release"] == {} and out["capacity_multipliers"] == {}


@pytest.mark.slow
def test_record_a_small_building(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAILSAFE_CACHE_DIR", str(tmp_path / "cache"))
    out = tmp_path / "site"
    manifest = export_static_site(out, templates=["care_home"], runs=40, micro=False)
    assert len(manifest["buildings"]) == 1
    (bid,) = manifest["buildings"]
    files = {p.name for p in (out / "r").iterdir()}
    assert f"{request_key('GET', '/api/health')}.json" in files
    assert any(name.endswith(".pdf") for name in files)
    graph = json.loads((out / "surrogate" / f"{bid}.json").read_text())
    assert graph["edge_src"] and graph["metadata"]["generator"] == "care_home"
    model = json.loads((out / "surrogate" / "model.json").read_text())
    n_weights = sum(math.prod(w["shape"]) for w in model["weights"].values())
    assert (out / "surrogate" / "weights.bin").stat().st_size == 4 * n_weights
