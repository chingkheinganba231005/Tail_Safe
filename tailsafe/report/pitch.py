"""Headline numbers for the pitch, generated from saved results only.

Every number in the output comes from JSON files written by ``stress run``,
``stress bottlenecks`` and ``optimize``; nothing is typed by hand. Regenerate
with ``tailsafe pitch`` after a new run so the pitch always matches the code.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tailsafe import DISCLAIMER, __version__


def _min(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def _ci(est: dict[str, float]) -> str:
    v, lo, hi = est["value"] / 60, est["lo"] / 60, est["hi"] / 60
    return f"{v:.1f} min (95% CI {lo:.1f} to {hi:.1f})"


def load_json(path: Path | None) -> dict[str, Any] | None:
    """Read a JSON file if it exists."""
    if path is None or not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def pitch_markdown(
    metrics: dict[str, Any],
    bottlenecks: dict[str, Any] | None = None,
    optimization: dict[str, Any] | None = None,
    *,
    sources: dict[str, str] | None = None,
) -> str:
    """Render the pitch metrics page."""
    scen = metrics["scenario"]
    risk = metrics["risk"]
    lines = [
        "# Pitch metrics",
        "",
        f"*Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by `tailsafe pitch` "
        f"(tailsafe {__version__}) from saved results — do not edit by hand.*",
        "",
        f"> {DISCLAIMER}",
        "> Most parameters are still marked `ASSUMPTION — needs citation` in "
        "`config/params.yaml`; these numbers demonstrate the method.",
        "",
        "## Scenario",
        "",
        f"- **Building:** {metrics['building']['name']}",
        f"- **Scenario:** {scen.get('description') or scen['name']}",
        f"- **Monte Carlo:** {metrics['runs']} sampled scenarios (seed {metrics['seed']}), "
        f"≈{metrics['occupants_mean']:.0f} occupants on average",
        "",
        "## Baseline: how bad is the tail?",
        "",
        "| | Mean | P95 | CVaR₉₅ (worst 5%) |",
        "|---|---:|---:|---:|",
    ]
    names = {
        "total_time": "Time until everyone is out (incl. fire-service rescue)",
        "self_evacuation_time": "Time until the last self-evacuee is out",
        "p95_occupant_time": "Time for 95% of occupants to get out",
    }
    for key, label in names.items():
        r = risk[key]
        lines.append(
            f"| {label} | {_min(r['mean']['value'])} | {_min(r['p95']['value'])} "
            f"| {_ci(r['cvar'])} |"
        )
    ten = metrics.get("tenability") or {}
    if ten.get("p_rset_exceeds_aset"):
        p = ten["p_rset_exceeds_aset"]
        inc = ten["p_any_incapacitated"]
        lines += [
            "",
            f"- **P(RSET > ASET):** {p['value']:.2f} (95% CI {p['lo']:.2f}–{p['hi']:.2f}) — "
            "the share of scenarios in which someone is still on a floor after its "
            "corridors become untenable, or breathes a dose above FED 0.3.",
            f"- **P(anyone incapacitated):** {inc['value']:.3f} "
            f"(95% CI {inc['lo']:.3f}–{inc['hi']:.3f}).",
        ]
        worst = ten.get("worst_floors") or []
        if worst:
            floors = ", ".join(f"level {w['level']} ({w['value']:.2f})" for w in worst[:3])
            lines.append(f"- **Floors most often failing:** {floors}.")
    bd = metrics.get("breakdown") or {}
    heads = [b.get("headline") for b in bd.values() if isinstance(b, dict) and b.get("headline")]
    if heads:
        lines += ["", "## Who carries the tail", ""]
        lines += [f"- {h}" for h in heads]
    if bottlenecks:
        lines += [
            "",
            "## Where it comes from (counterfactual bottlenecks)",
            "",
            f"Maximum egress flow {bottlenecks['max_flow_persons_per_s']:.2f} persons/s; "
            f"minimum cut: {', '.join(bottlenecks['min_cut'])}.",
            "",
            f"| Rank | Element | Δ CVaR₉₅ of {bottlenecks['loss'].replace('_', ' ')} |",
            "|---:|---|---:|",
        ]
        for row in bottlenecks["ranking"][:5]:
            d = row["delta_cvar"]
            lines.append(f"| {row['rank']} | {row['label']} | {_ci(d)} |")
        if bottlenecks.get("headline"):
            lines += ["", f"**{bottlenecks['headline']}**"]
    if optimization:
        conf = optimization["confirmation"]
        lines += [
            "",
            "## The fix: optimised operational plan",
            "",
            f"Objective: minimise {optimization['objective']['label']}; "
            f"{optimization['evaluations']} candidate plans evaluated with common random numbers.",
            "",
        ]
        lines += [f"- {x}" for x in optimization["plan_description"]]
        lines += [
            "",
            f"Confirmed on {conf['scenarios']} fresh scenarios (seed {conf['seed']}), paired:",
            "",
            "| | Baseline CVaR₉₅ | With plan | Change (95% CI) |",
            "|---|---:|---:|---:|",
        ]
        for key, label in names.items():
            row = conf["losses"][key]
            d = row["delta_cvar"]
            sig = " ✔" if row["significant"] else ""
            lines.append(
                f"| {label} | {_min(row['before_cvar'])} | {_min(row['after_cvar'])} | "
                f"{d['value'] / 60:+.1f} min ({d['lo'] / 60:+.1f} to {d['hi'] / 60:+.1f}){sig} |"
            )
        pr = conf["p_rset_exceeds_aset"]
        lines.append(
            f"| P(RSET > ASET) | {pr['before']:.2f} | {pr['after']:.2f} | "
            f"{pr['delta']['value']:+.2f} ({pr['delta']['lo']:+.2f} to {pr['delta']['hi']:+.2f}) |"
        )
        lines += ["", "✔ = the 95% confidence interval excludes zero."]
        worse = [
            label for key, label in names.items() if conf["losses"][key]["delta_cvar"]["lo"] > 0
        ]
        if pr["delta"]["lo"] > 0:
            worse.append("P(RSET > ASET)")
        if worse:
            lines += [
                "",
                "**Trade-off:** the plan makes "
                + ", ".join(w[0].lower() + w[1:] if not w.startswith("P(") else w for w in worse)
                + " significantly worse on the confirmation scenarios. Weigh this before "
                "adopting it, or re-run `tailsafe optimize --objective weighted` / "
                "`--objective p_rset`.",
            ]
    if sources:
        lines += ["", "## Sources", ""]
        lines += [f"- {k}: `{v}`" for k, v in sources.items()]
    return "\n".join(lines) + "\n"
