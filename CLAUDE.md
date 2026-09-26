# CLAUDE.md — TailSafe project guide

TailSafe stress-tests evacuation of Hong Kong high-rise buildings under uncertain,
worst-case conditions and ranks cheap operational interventions by how much they
shrink the **tail** (CVaR₉₅, P(RSET > ASET)), not the mean. The full product spec
is [`Tailsafeidea.md`](Tailsafeidea.md); this file records conventions, commands
and current status.

## Commands

```bash
make install     # .venv + editable install with dev extras
make test        # fast tests (pytest -m "not slow")
make test-slow   # performance / large Monte Carlo tests
make check       # ruff lint + format check + mypy --strict + tests  (= CI)
make format      # ruff format + safe autofixes
make dev         # FastAPI backend on :8000 with reload
make schema      # regenerate schemas/*.json from the Pydantic models
make demo        # generate + render the 40-storey cruciform block
.venv/bin/tailsafe --help
```

## Conventions

- **Units:** SI everywhere — metres, seconds, persons, persons/s, persons/m².
  Levels are integers, ground floor = 0; display labels follow HK style (`G/F`, `14/F`).
- **Determinism:** every stochastic function takes an explicit `seed` or
  `np.random.Generator`. Never call the global NumPy RNG. Same seed → same result.
  Scenario streams are derived with `np.random.SeedSequence` so different
  interventions see identical draws (common random numbers).
- **Parameters:** every physical / demographic number lives in
  `config/params.yaml` with a `source:`. Unknown → `ASSUMPTION — needs citation`.
  Never invent citations or statistics. Code reads values through
  `tailsafe.config.Params`; no magic numbers in modules (numerical settings such as
  the time step are fine in code).
- **Style:** type hints everywhere, docstrings on public functions/classes,
  `ruff` + `mypy --strict` clean. Prefer clarity over cleverness.
- **Tests:** every module gets unit tests; slow tests are marked
  `@pytest.mark.slow`. Validation checks (analytical, monotonicity, convergence)
  live in `tests/validation/` and are summarised in `docs/validation.md`.
- **Responsible use:** show the disclaimer (`tailsafe.DISCLAIMER`) in the UI,
  API and reports. All occupants are synthetic.
- **Git:** commits are authored by the repository owner; do not add AI
  co-author or session trailers to commit messages.

## Layout

```
tailsafe/config.py     parameter registry loader (inverse-CDF sampling)
tailsafe/building/     egress graph model, schema, validation, HK templates, rendering
tailsafe/population/   profiles, synthetic households, occupancy priors
tailsafe/sim/          mesoscopic queue-network engine (Numba kernel)
tailsafe/scenarios/    scenario sampler, Monte Carlo runner (CRN, LHS, convergence)
tailsafe/risk/         tail metrics with bootstrap CIs, breakdowns
tailsafe/api/          FastAPI app
config/params.yaml     parameter registry
schemas/               generated JSON schemas (do not edit by hand; `make schema`)
docs/                  architecture, validation, assumptions, pitch metrics
```

## Status

| # | Milestone | State |
|---|---|---|
| M0 | Scaffolding: tooling, CI, CLAUDE.md, params.yaml | ✅ done |
| M1 | Building model + JSON schema + procedural HK templates | ✅ done (`make demo`) |
| M2 | Meso simulator + population model | ⏳ next |
| M3 | Scenario sampler, Monte Carlo runner, risk metrics | ⏳ |
| M4–M11 | Hazard, bottlenecks, optimiser, web, micro-sim, vision, surrogate, briefing | not started |

## Decisions taken (open for review)

The spec asks for sign-off on hard-to-reverse choices. These were made to get
started and are easy to revisit while the codebase is small:

1. **Stack:** Python 3.11+, NumPy/SciPy, Numba for the simulator kernel, NetworkX
   for graph analytics, Pydantic v2 models as the source of truth for the JSON
   schema, FastAPI backend, Typer CLI. Frontend (M7): React + TypeScript + Vite +
   Tailwind + react-three-fiber.
2. **Graph schema:** physical connections are stored once as undirected *edges*
   (`flat`, `door`, `stair`, `lift`); the simulator expands them into directed
   *arcs* (`flat`, `stair_down`, `stair_up`). Doors are edges with a `door` block
   (width, fire rating, self-closing), not separate nodes. Lifts are separate
   entities listing the lobby node they serve on each level.
3. **Meso model:** time-stepped link-queue model (MATSim-style): agents walk
   along arcs with density- and profile-dependent speed, queue FIFO at arc ends,
   and enter the next arc subject to its inflow capacity (effective width ×
   max specific flow) and storage capacity. Merges at stair landings share the
   downstream capacity by a configurable floor/stair deference ratio.
4. **Households move as groups** at the pace of their slowest member.

## Open questions for the project owner

- Preferred sources for HK-specific values (stair widths, refuge floor interval,
  household composition, pre-movement times) — currently `ASSUMPTION`.
- Should reaching a refuge floor count as "safe" for RSET, or only final exits?
  (Current default: final exits; refuge arrival is recorded separately.)
- Is JuPedSim acceptable as an optional validation dependency for M8?
- "Twin-core private tower" is implemented as one core with a scissor (twin)
  staircase pair. Should it instead have two separate cores?
- Refuge floors: should occupants be forced to transfer between stairs at a
  refuge floor (stair discontinuity), as some codes require?
