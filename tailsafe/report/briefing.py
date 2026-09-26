"""One-page briefing for a building manager, grounded in computed results.

The briefing is written from a *facts* dictionary built only from saved
results (stress test, bottlenecks, optimisation), with every number already
rounded the way it may appear. Two writers:

* :func:`template_briefing` — deterministic text from the facts (always
  available);
* :func:`llm_briefing` — an Anthropic model, used when ``ANTHROPIC_API_KEY``
  and ``TAILSAFE_BRIEFING_MODEL`` are set and the optional ``anthropic``
  package is installed. The prompt forbids numbers that are not in the facts,
  and :func:`unknown_numbers` checks the draft afterwards: any number that does
  not appear in the facts rejects the draft and the template is used instead.

:func:`briefing_pdf` renders the Markdown on one A4 page with the evacuation
time distribution (before and after the plan when there is one).
"""

from __future__ import annotations

import json
import math
import os
import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tailsafe import DISCLAIMER
from tailsafe.building.builder import hk_level_label

OUTCOMES = {
    "total_time": "time until everyone is out, including fire-service rescue",
    "self_evacuation_time": "time until the last person who can leave unaided is out",
    "p95_occupant_time": "time for 95% of occupants to get out",
}
MODEL_ENV = "TAILSAFE_BRIEFING_MODEL"
KEY_ENV = "ANTHROPIC_API_KEY"

_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _m(seconds: float | None) -> float | None:
    """Seconds to minutes, rounded as displayed; ``None`` if never reached.

    Results arrive as JSON, where an outcome that never happens within the
    simulated period (``inf``) is ``null``.
    """
    if seconds is None or not math.isfinite(float(seconds)):
        return None
    return round(float(seconds) / 60.0, 1)


def _ci_min(est: dict[str, float | None]) -> list[float | None]:
    return [_m(est["lo"]), _m(est["hi"])]


def _verdict(delta: dict[str, float | None]) -> str:
    hi, lo = delta.get("hi"), delta.get("lo")
    if hi is not None and hi < 0:
        return "better"
    if lo is not None and lo > 0:
        return "worse"
    return "no clear change"


_SLOTS = {
    "weekday_day": "Weekday daytime",
    "weekday_night": "Weekday night",
    "weekend_day": "Weekend daytime",
    "weekend_night": "Weekend night",
}


def describe_scenario(spec: dict[str, Any]) -> str:
    """One sentence from the scenario's settings (never from free text that may be stale)."""
    parts = [_SLOTS.get(str(spec.get("time_slot")), "Scenario")]
    if spec.get("share_65_plus") is not None:
        parts.append(f"{round(100 * float(spec['share_65_plus']))}% of residents aged 65+")
    if spec.get("fire_level") is not None:
        parts.append(f"fire on {hk_level_label(int(spec['fire_level']))}")
    for bk in spec.get("stair_blockages") or []:
        t = (bk.get("time") or {}).get("value")
        when = f"from {round(float(t) / 60)} min" if t is not None else "at a random time"
        parts.append(f"Stair {bk['stair']} smoke-logged {when}")
    if spec.get("random_stair_blockage"):
        parts.append("a staircase may be lost at random")
    n_out = int(spec.get("lifts_out_of_service") or 0)
    if n_out:
        parts.append(f"{n_out} lift{'s' if n_out > 1 else ''} out of service")
    if spec.get("evacuation_lifts"):
        parts.append("lifts used to evacuate residents who need them")
    wardens = spec.get("warden_levels") or []
    if wardens:
        parts.append("wardens on " + ", ".join(hk_level_label(int(lv)) for lv in wardens))
    return ", ".join(parts) + "."


def briefing_facts(
    stress: dict[str, Any],
    bottlenecks: dict[str, Any] | None = None,
    optimization: dict[str, Any] | None = None,
    *,
    building_name: str | None = None,
    scenario: str | None = None,
) -> dict[str, Any]:
    """Everything a briefing may say, with numbers rounded as they may appear.

    ``stress`` is either ``metrics.json`` from ``tailsafe stress run`` or the
    result of an API stress job; ``bottlenecks`` and ``optimization`` are the
    corresponding JSON results (optional).
    """
    risk = stress["risk"]
    spec = stress.get("spec") or {}
    scen = stress.get("scenario") or {}
    facts: dict[str, Any] = {
        "building": building_name or (stress.get("building") or {}).get("name", "the building"),
        "scenario": scenario or describe_scenario(scen or spec),
        "simulated_scenarios": int(stress["runs"]),
        "occupants_on_average": round(float(stress.get("occupants_mean", 0.0))),
        "confidence_level_percent": 95,
        "tail": "the worst 5% of simulated scenarios",
        "outcomes": {},
    }
    for key, label in OUTCOMES.items():
        r = risk[key]
        facts["outcomes"][key] = {
            "label": label,
            "average_min": _m(r["mean"]["value"]),
            "p95_min": _m(r["p95"]["value"]),
            "worst_5pct_average_min": _m(r["cvar"]["value"]),
            "worst_5pct_average_ci_min": _ci_min(r["cvar"]),
        }
    ten = stress.get("tenability") or {}
    if ten.get("p_rset_exceeds_aset"):
        p = ten["p_rset_exceeds_aset"]
        facts["someone_caught_by_smoke"] = {
            "meaning": "share of scenarios in which someone is still on a floor after its "
            "corridors become untenable, or breathes a harmful dose of smoke",
            "percent": round(100 * p["value"]),
            "ci_percent": [round(100 * p["lo"]), round(100 * p["hi"])],
        }
        inc = ten.get("p_any_incapacitated")
        if inc:
            facts["someone_incapacitated_percent"] = round(100 * inc["value"], 1)
        facts["floors_most_at_risk"] = [
            {
                "floor": hk_level_label(int(w["level"])),
                "percent_of_scenarios": round(100 * w["value"]),
            }
            for w in (ten.get("worst_floors") or [])[:3]
        ]
    heads = [
        b["headline"]
        for b in (stress.get("breakdown") or {}).values()
        if isinstance(b, dict) and b.get("headline")
    ]
    if heads:
        facts["who_is_most_at_risk"] = heads
    if bottlenecks:
        facts["bottlenecks"] = {
            "outcome": OUTCOMES.get(bottlenecks["loss"], bottlenecks["loss"].replace("_", " ")),
            "headline": bottlenecks.get("headline"),
            "top": [
                {
                    "element": row["label"],
                    "change_in_worst_5pct_average_min": _m(row["delta_cvar"]["value"]),
                    "ci_min": _ci_min(row["delta_cvar"]),
                }
                for row in bottlenecks["ranking"][:3]
            ],
        }
    if optimization:
        conf = optimization["confirmation"]
        effects = []
        for key, label in OUTCOMES.items():
            row = conf["losses"][key]
            d = row["delta_cvar"]
            effects.append(
                {
                    "outcome": label,
                    "worst_5pct_average_before_min": _m(row["before_cvar"]),
                    "worst_5pct_average_after_min": _m(row["after_cvar"]),
                    "change_min": _m(d["value"]),
                    "ci_min": _ci_min(d),
                    "verdict": _verdict(d),
                }
            )
        pr = conf["p_rset_exceeds_aset"]
        facts["plan"] = {
            "objective": optimization["objective"]["label"],
            "actions": list(optimization["plan_description"]),
            "plans_compared": int(optimization["evaluations"]),
            "confirmation_scenarios": int(conf["scenarios"]),
            "effects": effects,
            "someone_caught_by_smoke": {
                "before_percent": round(100 * pr["before"]),
                "after_percent": round(100 * pr["after"]),
                "verdict": _verdict(pr["delta"]),
            },
            "made_worse": [e["outcome"] for e in effects if e["verdict"] == "worse"],
        }
    facts["disclaimer"] = DISCLAIMER
    return facts


# ----------------------------------------------------------------------------- checking
def _numbers(text: str) -> list[str]:
    return [m.group(0) for m in _NUMBER.finditer(text)]


def _value(token: str) -> float:
    return float(token.replace(",", ""))


def unknown_numbers(text: str, facts: dict[str, Any]) -> list[str]:
    """Numbers in ``text`` that do not appear anywhere in ``facts``.

    Signs are ignored (the text may say "cuts 12.8 min" for a change of −12.8)
    and trailing zeros are not significant (0.60 = 0.6). Numbers written as
    words are not checked.
    """
    allowed = {_value(t) for t in _numbers(json.dumps(facts, ensure_ascii=False))}
    return [t for t in _numbers(text) if _value(t) not in allowed]


# ----------------------------------------------------------------------------- writers
BEYOND = "beyond the simulated period"


def _mins(x: float | None) -> str:
    """ "12.3 min", or a phrase when the outcome was never reached."""
    return BEYOND if x is None else f"{x:.1f} min"


def _ci(pair: list[float | None]) -> str:
    lo, hi = pair
    if lo is None or hi is None:
        return ""
    return f" (95% confidence {lo:.1f} to {hi:.1f} min)"


def _lower_first(s: str) -> str:
    return s[:1].lower() + s[1:]


def template_briefing(facts: dict[str, Any]) -> str:
    """Deterministic one-page briefing (Markdown) from the facts only."""
    out = facts["outcomes"]
    tot, selfe = out["total_time"], out["self_evacuation_time"]
    lines = [
        f"# Evacuation stress test: {facts['building']}",
        "",
        f"**Scenario:** {facts['scenario']}",
        f"Based on {facts['simulated_scenarios']} simulated runs of this scenario, with about "
        f"{facts['occupants_on_average']} synthetic residents in each.",
        "",
        "## What we found",
        "",
        f"- On an average run, everyone is out after {_mins(tot['average_min'])} "
        "(including fire-service rescue). In the worst 5% of runs it takes "
        f"{_mins(tot['worst_5pct_average_min'])} on average"
        f"{_ci(tot['worst_5pct_average_ci_min'])}.",
        f"- The last person who can leave unaided is out after {_mins(selfe['average_min'])} "
        f"on average, {_mins(selfe['worst_5pct_average_min'])} in the worst 5%.",
    ]
    smoke = facts.get("someone_caught_by_smoke")
    if smoke:
        lines.append(
            f"- In {smoke['percent']}% of runs someone is still on a floor after its corridors "
            "become untenable, or breathes a harmful dose of smoke."
        )
        floors = facts.get("floors_most_at_risk") or []
        if floors:
            lines.append(
                "- Floors most often affected: "
                + ", ".join(f"{f['floor']} ({f['percent_of_scenarios']}% of runs)" for f in floors)
                + "."
            )
    who = facts.get("who_is_most_at_risk")
    if who:
        lines += ["", "## Who is most at risk", ""]
        lines += [f"- {h}" for h in who]
    bn = facts.get("bottlenecks")
    if bn:
        lines += ["", "## Why", ""]
        if bn.get("headline"):
            lines.append(f"- {bn['headline']}")
        for row in bn["top"][1:]:
            change = row["change_in_worst_5pct_average_min"]
            if change is not None and change < 0:
                lines.append(
                    f"- More capacity at {row['element']} would cut the worst-5% average of the "
                    f"{bn['outcome']} by {_mins(-change)}."
                )
    plan = facts.get("plan")
    if plan:
        lines += ["", "## What to do", ""]
        lines += [f"- {a}" for a in plan["actions"]]
        lines.append("")
        lines.append(
            f"Tested on {plan['confirmation_scenarios']} fresh runs against the same "
            "runs without the plan:"
        )
        lines.append("")
        for e in plan["effects"]:
            if e["verdict"] == "no clear change" or e["change_min"] is None:
                what = e["verdict"]
            else:
                what = f"{e['verdict']} by {_mins(abs(e['change_min']))}"
            lines.append(
                f"- {e['outcome'][:1].upper() + e['outcome'][1:]} (worst 5%): "
                f"{_mins(e['worst_5pct_average_before_min'])} → "
                f"{_mins(e['worst_5pct_average_after_min'])}, {what}."
            )
        s = plan["someone_caught_by_smoke"]
        lines.append(
            f"- Runs where someone is caught by smoke: {s['before_percent']}% → "
            f"{s['after_percent']}% ({s['verdict']})."
        )
        if plan["made_worse"]:
            lines += [
                "",
                "**Trade-off:** the plan makes the "
                + " and the ".join(plan["made_worse"])
                + " worse. Weigh this before adopting it.",
            ]
    lines += [
        "",
        "## Limits",
        "",
        "- These results come from a simulation with many assumed parameters; they show "
        "where the risk concentrates and what helps, not exact times.",
        f"- {facts['disclaimer']}",
    ]
    return "\n".join(lines) + "\n"


SYSTEM_PROMPT = """You write one-page evacuation-risk briefings for the manager of a residential \
building in Hong Kong. You are given FACTS as JSON, computed by a simulator.

Rules:
- Use only the facts. Every number you write must appear in the JSON exactly as \
given (you may drop a minus sign when you say "cuts" or "reduces"). Never compute, \
round, convert, add or subtract numbers, and never introduce dates, counts or \
percentages that are not in the JSON. Write numbers as digits.
- Plain English for a non-specialist. Explain "worst 5%" once. No tables.
- Markdown: a "# " title, then "## What we found", "## Who is most at risk" (if \
the facts say), "## Why" (if bottlenecks are given), "## What to do" (if a plan is \
given, including any trade-off it makes worse), "## Limits". Bullet points.
- At most 350 words. End with the disclaimer from the JSON, verbatim."""


def llm_briefing(facts: dict[str, Any], *, model: str, api_key: str | None = None) -> str:
    """Draft a briefing with an Anthropic model (not checked; see :func:`make_briefing`)."""
    import anthropic  # optional dependency: pip install "tailsafe[briefing]"

    client = anthropic.Anthropic(api_key=api_key or os.environ.get(KEY_ENV))
    msg = client.messages.create(
        model=model,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": "FACTS:\n" + json.dumps(facts, ensure_ascii=False, indent=1),
            }
        ],
    )
    return "".join(getattr(block, "text", "") for block in msg.content).strip() + "\n"


@dataclass
class Briefing:
    """A checked briefing and how it was produced."""

    markdown: str
    source: str  # "template" or "llm"
    facts: dict[str, Any]
    unknown_numbers: list[str] = field(default_factory=list)
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return asdict(self)


def make_briefing(facts: dict[str, Any], *, use_llm: bool | None = None) -> Briefing:
    """Briefing from the LLM when configured and its numbers check out, else the template.

    ``use_llm``: ``None`` = when configured, ``False`` = never, ``True`` = try
    (still falls back to the template, with a note, if it fails the check).
    """
    template = template_briefing(facts)
    bad = unknown_numbers(template, facts)
    if bad:  # a bug in the template, not something to hide
        raise AssertionError(f"template briefing uses numbers not in the facts: {bad}")
    model = os.environ.get(MODEL_ENV)
    configured = bool(model and os.environ.get(KEY_ENV))
    if use_llm is False or (use_llm is None and not configured):
        return Briefing(template, "template", facts)
    if not configured:
        return Briefing(
            template,
            "template",
            facts,
            note=f"Set {KEY_ENV} and {MODEL_ENV} to draft the briefing with an LLM.",
        )
    try:
        draft = llm_briefing(facts, model=model or "")
    except Exception as exc:  # network, quota, missing package: fall back, say why
        return Briefing(
            template, "template", facts, note=f"LLM unavailable ({type(exc).__name__}: {exc})."
        )
    bad = unknown_numbers(draft, facts)
    if bad:
        return Briefing(
            template,
            "template",
            facts,
            unknown_numbers=bad,
            note="The LLM draft used numbers that are not in the results "
            f"({', '.join(bad[:8])}); showing the template briefing instead.",
        )
    return Briefing(draft, "llm", facts)


# ----------------------------------------------------------------------------- PDF
# The web UI's chart palette (checked for colour-blind separation) and ink.
_BLUE, _ORANGE = "#7a4a9e", "#d9730d"
_INK, _INK2, _MUTED, _GRID = "#0b0c0c", "#4f5154", "#737373", "#e8e8e8"


def _plain(s: str) -> str:
    return s.replace("**", "").replace("`", "")


def _cvar(x: np.ndarray[Any, Any], alpha: float = 0.95) -> float:
    s = np.sort(x)
    return float(s[int(np.floor(alpha * s.size)) :].mean())


def briefing_pdf(
    markdown: str,
    path: Path,
    distributions: Mapping[str, Sequence[float]] | None = None,
    *,
    outcome: str = "Time until everyone is out (min)",
) -> Path:
    """Render the briefing on one A4 page, with the distribution chart if given.

    ``distributions`` maps a series name ("Baseline", "With plan") to outcome
    samples in seconds.
    """
    from matplotlib.figure import Figure  # object API: no global state, thread-safe

    fig = Figure(figsize=(8.27, 11.69))
    left, right, top = 0.08, 0.92, 0.955
    width_chars = 112
    y = top
    chart_h = 0.17 if distributions else 0.0
    bottom_text = 0.075 + (chart_h + 0.035 if distributions else 0.0)

    def put(text: str, size: float, dy: float, **kw: Any) -> None:
        nonlocal y
        fig.text(left, y, text, fontsize=size, va="top", color=kw.pop("color", _INK), **kw)
        y -= dy

    for raw in markdown.splitlines():
        line = _plain(raw.rstrip())
        if not line:
            y -= 0.004
        elif line.startswith("# "):
            for w in textwrap.wrap(line[2:], 60):
                put(w, 14, 0.024, weight="bold")
            y -= 0.004
        elif line.startswith("## "):
            y -= 0.003
            put(line[3:], 10.5, 0.019, weight="bold")
        elif line.startswith("- "):
            wrapped = textwrap.wrap(line[2:], width_chars - 4)
            for k, w in enumerate(wrapped):
                put(("•  " if k == 0 else "    ") + w, 8.5, 0.0132)
        else:
            for w in textwrap.wrap(line, width_chars):
                put(w, 8.5, 0.0132, color=_INK2)
    if y < bottom_text:
        distributions = None  # no room: text first
    if distributions:
        ax = fig.add_axes((left + 0.04, 0.075, right - left - 0.06, chart_h))
        data = {k: np.asarray(v, dtype=np.float64) / 60.0 for k, v in distributions.items()}
        finite = np.concatenate([d[np.isfinite(d)] for d in data.values()])
        bins = np.linspace(0.0, float(finite.max()) * 1.02, 40)
        colors = [_BLUE, _ORANGE]
        series = [
            (name, d[np.isfinite(d)], c) for (name, d), c in zip(data.items(), colors, strict=False)
        ]
        for name, d, c in series:
            ax.hist(d, bins=bins.tolist(), histtype="step", linewidth=1.6, color=c, label=name)
        top_y = ax.get_ylim()[1]
        for k, (_, d, c) in enumerate(series):
            cv = _cvar(d)
            ax.axvline(cv, color=c, linewidth=1.0, linestyle="--")
            ax.text(cv, top_y * (0.97 - 0.12 * k), " worst 5%", color=_INK2, fontsize=7, va="top")
        ax.set_xlabel(outcome, fontsize=8, color=_INK2)
        ax.set_ylabel("Runs", fontsize=8, color=_INK2)
        ax.tick_params(labelsize=7, colors=_MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(_GRID)
        ax.grid(axis="y", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        if len(data) > 1:
            ax.legend(fontsize=7, frameon=False, labelcolor=_INK)
        ax.set_title(
            "Distribution over simulated runs; dashed lines: average of the worst 5%",
            fontsize=8,
            color=_INK2,
            loc="left",
        )
    fig.text(
        left,
        0.02,
        "Generated by TailSafe from simulated results. All occupants are synthetic.",
        fontsize=7,
        color=_MUTED,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)  # format from the suffix: .pdf, or .png for a preview
    return path
