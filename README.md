# TailSafe

**Tail-risk evacuation stress-testing for high-rise Hong Kong.**

TailSafe asks one question about a building: *what does the bad 5% of evacuation
outcomes look like, why, and which cheap operational changes fix it?* It samples
thousands of realistic, uncertain scenarios for a building (who is home at 3 a.m.,
where the fire starts, which stair is smoke-logged, which lift is out of service),
simulates each one with a fast mesoscopic crowd-flow engine, and reports tail-risk
measures such as **CVaR₉₅** (the expected evacuation time in the worst 5% of
scenarios) instead of a single "design" average.

> **Responsible use.** TailSafe is an educational and decision-support prototype.
> It is **not** a substitute for a registered fire engineer, the Fire Services
> Department, or compliance with the Buildings Department's *Code of Practice for
> Fire Safety in Buildings*. All occupants are synthetic; no personal data is used.
> Many parameters are still marked `ASSUMPTION — needs citation` in
> [`config/params.yaml`](config/params.yaml); see [`docs/assumptions.md`](docs/assumptions.md).

## Quick start

Requires Python 3.11+ and `make`.

```bash
make install     # create .venv and install tailsafe + dev tools
make test        # fast test suite
make check       # lint + format check + mypy + tests (what CI runs)
make dev         # start the API on http://localhost:8000 (docs at /docs)
make demo        # generate and render a 40-storey cruciform public-housing block
```

The `tailsafe` command is installed into `.venv/bin`:

```bash
.venv/bin/tailsafe --help
.venv/bin/tailsafe params check              # validate the parameter registry
.venv/bin/tailsafe params list --assumptions # values that still need a citation
```

## What is in the box

| Area | Module | Status |
|---|---|---|
| Parameter registry with sources | `config/params.yaml`, `tailsafe/config.py` | ✅ |
| Building model, JSON schema, HK typologies | `tailsafe/building/` | see [status](#status) |
| Synthetic population and behaviour | `tailsafe/population/` | see [status](#status) |
| Mesoscopic queue-network simulator | `tailsafe/sim/` | see [status](#status) |
| Scenario sampler, Monte Carlo, risk metrics | `tailsafe/scenarios/`, `tailsafe/risk/` | see [status](#status) |
| Hazard (smoke / FED / ASET), bottlenecks, optimiser, surrogate, vision, briefing, web UI | | planned |

## Status

Development follows the milestones in [`Tailsafeidea.md`](Tailsafeidea.md) §9.
The current state of each milestone is tracked in [`CLAUDE.md`](CLAUDE.md).

## Repository layout

```
tailsafe/            Python package (building, population, sim, hazard, scenarios,
                     risk, analysis, optimize, surrogate, vision, report, api)
config/params.yaml   every physical / demographic parameter, each with a source
schemas/             JSON schemas generated from the Pydantic models
data/templates/      generated example buildings
docs/                architecture, validation, assumptions, pitch metrics
tests/               pytest suite
```

## Documentation

- [Architecture](docs/architecture.md) — modules, data flow and key design decisions
- [Validation](docs/validation.md) — analytical checks and what they show
- [Assumptions](docs/assumptions.md) — every simplification, stated plainly
