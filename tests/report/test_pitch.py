from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tailsafe.building.templates import generate
from tailsafe.cli import app
from tailsafe.optimize.plan import InterventionPlan
from tailsafe.optimize.search import Objective, confirm
from tailsafe.report.pitch import pitch_markdown
from tailsafe.risk.breakdown import tail_breakdown
from tailsafe.scenarios.montecarlo import MCConfig, run_monte_carlo
from tailsafe.scenarios.spec import demo_spec


def test_pitch_page_from_real_results(tmp_path: Path) -> None:
    b = generate("cruciform", storeys=16, flats_per_wing=2)
    spec = demo_spec().model_copy(update={"fire_level": 6})
    cfg = MCConfig(n_runs=20, batch_size=20, workers=1)
    res = run_monte_carlo(b, spec, cfg)
    metrics = {
        **res.summary(),
        "breakdown": {"total_time": tail_breakdown(res, "total_time")},
    }
    plan = InterventionPlan(evacuation_lifts=True)
    after = run_monte_carlo(b, plan.apply(spec, b), cfg)
    optimization = {
        "objective": Objective().model_dump() | {"label": Objective().label()},
        "evaluations": 2,
        "plan_description": plan.describe(),
        "confirmation": confirm(res, after, Objective()),
    }
    text = pitch_markdown(json.loads(json.dumps(metrics, default=float)), None, optimization)
    assert "# Pitch metrics" in text and "CVaR₉₅" in text
    assert "not a substitute" in text
    assert "evacuate mobility-impaired residents" in text
    assert "P(RSET > ASET)" in text

    # CLI: writes the file from saved results.
    stress = tmp_path / "stress"
    res.save(stress)
    (stress / "metrics.json").write_text(json.dumps(metrics, default=float))
    out = tmp_path / "pitch.md"
    r = CliRunner().invoke(
        app,
        [
            "pitch",
            "--stress",
            str(stress),
            "--optimization",
            str(tmp_path / "none"),
            "--out",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.stdout
    assert out.read_text().startswith("# Pitch metrics")


def test_pitch_flags_trade_offs() -> None:
    est = {"value": 600.0, "lo": 500.0, "hi": 700.0}
    risk = {k: {"mean": est, "p95": est, "cvar": est} for k in _LOSSES}
    metrics = {
        "building": {"name": "Test block"},
        "scenario": {"name": "demo", "description": ""},
        "runs": 10,
        "seed": 0,
        "occupants_mean": 100.0,
        "risk": risk,
    }

    def row(delta: float) -> dict[str, object]:
        return {
            "before_cvar": 600.0,
            "after_cvar": 600.0 + delta,
            "delta_cvar": {"value": delta, "lo": delta - 10, "hi": delta + 10},
            "significant": True,
        }

    optimization = {
        "objective": {"label": "CVaR95 of total time"},
        "evaluations": 3,
        "plan_description": ["Something."],
        "confirmation": {
            "scenarios": 10,
            "seed": 1,
            "losses": {
                "total_time": row(-300.0),
                "self_evacuation_time": row(-60.0),
                "p95_occupant_time": row(+120.0),
            },
            "p_rset_exceeds_aset": {
                "before": 0.5,
                "after": 0.7,
                "delta": {"value": 0.2, "lo": 0.1, "hi": 0.3},
            },
        },
    }
    text = pitch_markdown(metrics, None, optimization)
    assert "**Trade-off:**" in text
    assert "time for 95% of occupants" in text and "P(RSET > ASET)" in text
    assert "time until everyone" not in text.split("**Trade-off:**")[1]
    rows = {line.split("|")[1].strip(): line for line in text.splitlines() if line.startswith("| ")}
    assert rows["Time until everyone is out (incl. fire-service rescue)"].endswith("✔ |")
    assert rows["Time for 95% of occupants to get out"].endswith("▲ |")
    assert rows["P(RSET > ASET)"].endswith("▲ |")


_LOSSES = ("total_time", "self_evacuation_time", "p95_occupant_time")
