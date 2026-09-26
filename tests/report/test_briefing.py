from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tailsafe.analysis.bottlenecks import attribute_bottlenecks
from tailsafe.api.views import stress_view
from tailsafe.building.templates import generate
from tailsafe.optimize.plan import InterventionPlan
from tailsafe.optimize.search import Objective, confirm
from tailsafe.report import briefing as br
from tailsafe.report.briefing import (
    briefing_facts,
    briefing_pdf,
    make_briefing,
    template_briefing,
    unknown_numbers,
)
from tailsafe.scenarios.montecarlo import MCConfig, run_monte_carlo
from tailsafe.scenarios.spec import demo_spec


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    b = generate("cruciform", storeys=16, flats_per_wing=2)
    spec = demo_spec().model_copy(update={"fire_level": 6})
    cfg = MCConfig(n_runs=20, batch_size=20, workers=1)
    res = run_monte_carlo(b, spec, cfg)
    plan = InterventionPlan(evacuation_lifts=True)
    after = run_monte_carlo(b, plan.apply(spec, b), cfg)
    optimization = {
        "objective": Objective().model_dump() | {"label": Objective().label()},
        "evaluations": 2,
        "plan_description": plan.describe(),
        "confirmation": confirm(res, after, Objective()),
    }
    roundtrip = json.loads(json.dumps(stress_view(res), default=float))
    return {
        "building": b,
        "stress": roundtrip,
        "bottlenecks": json.loads(json.dumps(attribute_bottlenecks(res), default=float)),
        "optimization": json.loads(json.dumps(optimization, default=float)),
        "before": res.loss("total_time"),
        "after": after.loss("total_time"),
    }


def test_template_uses_only_numbers_from_the_facts(results: dict[str, Any]) -> None:
    facts = briefing_facts(
        results["stress"],
        results["bottlenecks"],
        results["optimization"],
        building_name=results["building"].name,
    )
    assert facts["simulated_scenarios"] == 20
    # The scenario sentence comes from the settings, not the demo's free text.
    assert facts["scenario"].startswith("Weekend night, 22% of residents aged 65+, fire on 6/F")
    assert "40-storey" not in facts["scenario"]
    assert facts["plan"]["confirmation_scenarios"] == 20
    text = template_briefing(facts)
    assert unknown_numbers(text, facts) == []
    for heading in ("## What we found", "## Why", "## What to do", "## Limits"):
        assert heading in text
    assert "not a substitute" in text
    # Without bottlenecks or a plan the briefing is shorter but still grounded.
    small = briefing_facts(results["stress"])
    text = template_briefing(small)
    assert "## What to do" not in text and unknown_numbers(text, small) == []


def test_unknown_numbers() -> None:
    facts = {"a": 219.2, "b": [-12.8, 0.60], "floor": "14/F", "n": 1825}
    assert unknown_numbers("219.2 min, cut by 12.8 min, 0.6, on 14/F, 1,825 people", facts) == []
    assert unknown_numbers("81.3 min and 219.20", facts) == ["81.3"]
    assert unknown_numbers("P95 of 219.2", facts) == ["95"]


def test_llm_draft_is_checked(results: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    facts = briefing_facts(results["stress"], results["bottlenecks"], results["optimization"])
    monkeypatch.delenv(br.KEY_ENV, raising=False)
    monkeypatch.delenv(br.MODEL_ENV, raising=False)
    assert make_briefing(facts).source == "template"
    assert br.MODEL_ENV in (make_briefing(facts, use_llm=True).note or "")

    monkeypatch.setenv(br.KEY_ENV, "test-key")
    monkeypatch.setenv(br.MODEL_ENV, "test-model")
    everyone = facts["outcomes"]["total_time"]["worst_5pct_average_min"]
    drafts = {
        "grounded": f"# Briefing\n\nIn the worst 5% of runs: {everyone} min.\n",
        "invented": f"# Briefing\n\n{everyone} min, or 12345.6 min in total.\n",
    }
    for kind, draft in drafts.items():
        monkeypatch.setattr(br, "llm_briefing", lambda f, model, api_key=None, d=draft: d)
        out = make_briefing(facts)
        if kind == "grounded":
            assert out.source == "llm" and out.unknown_numbers == []
        else:
            assert out.source == "template" and out.unknown_numbers == ["12345.6"]
            assert "12345.6" in (out.note or "")

    def fail(*_: Any, **__: Any) -> str:
        raise ConnectionError("offline")

    monkeypatch.setattr(br, "llm_briefing", fail)
    out = make_briefing(facts)
    assert out.source == "template" and "offline" in (out.note or "")
    assert make_briefing(facts, use_llm=False).note is None


def test_pdf_page(results: dict[str, Any], tmp_path: Path) -> None:
    facts = briefing_facts(results["stress"], results["bottlenecks"], results["optimization"])
    text = make_briefing(facts, use_llm=False).markdown
    pdf = briefing_pdf(
        text,
        tmp_path / "brief.pdf",
        {"Baseline": results["before"], "With plan": results["after"]},
    )
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and data.count(b"/Type /Page\n") <= 1
    png = briefing_pdf(text, tmp_path / "brief.png")
    assert png.read_bytes()[:4] == b"\x89PNG"


def test_brief_cli(results: dict[str, Any], tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tailsafe.cli import app

    stress = tmp_path / "stress"
    stress.mkdir()
    (stress / "metrics.json").write_text(json.dumps(results["stress"]))
    (stress / "bottlenecks.json").write_text(json.dumps(results["bottlenecks"]))
    opt = tmp_path / "opt"
    opt.mkdir()
    (opt / "optimization.json").write_text(json.dumps(results["optimization"]))
    out = tmp_path / "brief.md"
    r = CliRunner().invoke(
        app,
        [
            "brief",
            "--stress",
            str(stress),
            "--optimization",
            str(opt),
            "--writer",
            "template",
            "--out",
            str(out),
            "--pdf",
            str(tmp_path / "brief.pdf"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert out.read_text().startswith("# Evacuation stress test")
    assert (tmp_path / "brief.pdf").read_bytes().startswith(b"%PDF")
    facts = json.loads((tmp_path / "brief.facts.json").read_text())
    assert unknown_numbers(out.read_text(), facts) == []


@pytest.mark.slow
def test_demo_cli_end_to_end(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from tailsafe.cli import app

    r = CliRunner().invoke(
        app,
        [
            "demo",
            "--out",
            str(tmp_path),
            "--runs",
            "60",
            "--scenarios",
            "20",
            "--confirm",
            "40",
            "--no-replay",
        ],
    )
    assert r.exit_code == 0, r.output
    for name in ("building.png", "stress/metrics.json", "plan/optimization.json"):
        assert (tmp_path / name).exists(), name
    assert (tmp_path / "briefing.pdf").read_bytes().startswith(b"%PDF")
    assert json.loads((tmp_path / "timings.json").read_text())["total"] > 0
