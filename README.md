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
make stress-demo # 1,000-scenario stress test of the pitch scenario (~1 min on 4 cores)
```

The `tailsafe` command is installed into `.venv/bin`:

```bash
.venv/bin/tailsafe --help
.venv/bin/tailsafe params check              # validate the parameter registry
.venv/bin/tailsafe params list --assumptions # values that still need a citation
.venv/bin/tailsafe validate                  # analytical checks of the simulator
.venv/bin/tailsafe sim run cruciform --slot weekend_night --share-65 0.22 \
    --block-stair A@240 --plot out/run.png   # one scenario: JSON summary + plot
```

![40-storey cruciform public-housing block: typical-floor plan and 3D stack](docs/img/cruciform_40.png)

*A procedurally generated 40-storey cruciform public-housing block (`make demo`):
typical-floor plan with the egress graph, and the 3D stack with the refuge floor.*

### Example: the pitch scenario

*Sunday 3 a.m., 40-storey public housing block, 22% of residents aged 65+,
Stair A smoke-logged at 4 minutes, one lift out of service* — 1,000 sampled
scenarios (`make stress-demo`):

![Distribution of total evacuation time with mean, P95 and CVaR95, and who is still inside in the worst 5%](docs/img/stress_demo.png)

> These numbers come from parameters that are still largely **assumptions**
> (walking speeds of frail residents, pre-movement at night, fire-service
> rescue logistics). Treat them as a demonstration of the method until the
> registry is calibrated.

## What is in the box

| Area | Module | Status |
|---|---|---|
| Parameter registry with sources | `config/params.yaml`, `tailsafe/config.py` | ✅ |
| Building model, JSON schema, HK typologies | `tailsafe/building/` | ✅ |
| Synthetic population and behaviour | `tailsafe/population/` | ✅ |
| Mesoscopic queue-network simulator (validated against hydraulic calculations) | `tailsafe/sim/` | ✅ |
| Scenario sampler, parallel Monte Carlo, CVaR₉₅ with CIs, tail breakdowns | `tailsafe/scenarios/`, `tailsafe/risk/` | ✅ |
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
