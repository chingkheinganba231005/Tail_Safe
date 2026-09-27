# TailSafe — developer guide

TailSafe stress-tests evacuation of Hong Kong high-rise buildings under uncertain,
worst-case conditions and ranks cheap operational interventions by how much they
shrink the **tail** (CVaR₉₅, P(RSET > ASET)), not the mean. The full product spec
is [`docs/spec.md`](docs/spec.md); this file records commands, conventions
and design decisions.

## Commands

```bash
make install     # .venv + editable install with the dev, surrogate and vision extras
make test        # fast tests (pytest -m "not slow")
make test-slow   # performance / large Monte Carlo tests
make check       # ruff lint + format check + mypy --strict + tests  (= CI)
make format      # ruff format + safe autofixes
make dev         # FastAPI backend on :8000 + Vite UI on :5173 (hot reload)
make web         # build web/dist (served by the API at /)
.venv/bin/tailsafe export-site --out web/public/data  # recorded results for the browser version
cd web && VITE_STATIC=1 npx vite    # the browser version locally (no Python server)
docker build -t tailsafe . && docker run -p 7860:7860 tailsafe   # the full app in a container
make web-check   # web type-check + vitest (CI job "Web UI")
make schema      # regenerate schemas/*.json from the Pydantic models
make example     # generate + render the 40-storey cruciform block
.venv/bin/tailsafe validate --markdown        # analytical checks of the simulator
.venv/bin/tailsafe sim run cruciform --slot weekend_night --share-65 0.22 \
    --block-stair A@240 --plot out/run.png  # one scenario, summary JSON + plot
.venv/bin/tailsafe sim run cruciform --fire L14.unit.N3 --fire-door-open --plot out/fire.png
.venv/bin/tailsafe stress run cruciform --spec reference --runs 1000 --out out/stress
.venv/bin/tailsafe stress report out/stress --loss self_evacuation_time
.venv/bin/tailsafe stress bottlenecks out/stress     # counterfactual ranking (re-runs)
.venv/bin/tailsafe optimize cruciform --out out/plan # plan search + confirmation
.venv/bin/tailsafe brief --stress out/stress --optimization out/plan # briefing .md + .pdf (checked)
.venv/bin/tailsafe micro run cruciform --index 3 --plot out/floor.png --level 14 --time 420
.venv/bin/tailsafe micro compare cruciform --runs 20      # meso–micro agreement table
.venv/bin/tailsafe micro fd                              # micro speed–density vs hydraulic
.venv/bin/tailsafe vision synth slab --out out/plan.png  # rendered plan + truth JSON
.venv/bin/tailsafe vision detect out/plan.png --scale 0,0,200,0,10 --out out/det.json --overlay out/det.png
.venv/bin/tailsafe vision build out/det.json --storeys 20   # stacked, validated building JSON
.venv/bin/tailsafe vision eval                           # precision/recall on rendered plans
.venv/bin/tailsafe surrogate data --cases 800            # simulated training cases (~7 min, 4 cores)
.venv/bin/tailsafe surrogate eval                        # random + leave-one-typology-out (~35 min)
.venv/bin/tailsafe surrogate train                       # ship weights to tailsafe/surrogate/weights/
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

## Layout

```
tailsafe/config.py     parameter registry loader (inverse-CDF sampling)
tailsafe/building/     egress graph model, schema, validation, HK templates, rendering
tailsafe/population/   profiles, synthetic households, occupancy priors
tailsafe/rng.py        named random streams (common random numbers)
tailsafe/sim/          mesoscopic queue-network engine (Numba kernel in _kernel.py);
                       micro replay engine (micro.py, _micro_kernel.py) sharing meso decisions
tailsafe/scenarios/    ScenarioSpec, sampler (16 fixed uniform slots, LHS), Monte Carlo runner
tailsafe/hazard/       zone smoke network, tenability (visibility, FED), ASET, CFD import
tailsafe/analysis/     bottlenecks: queue recurrence, min-cut/load, counterfactual ΔCVaR;
                       meso–micro agreement (agreement.py)
tailsafe/risk/         VaR/CVaR with bootstrap CIs, paired differences, tail breakdowns,
                       RSET vs ASET, plots
tailsafe/optimize/     InterventionPlan levers, SAA search with CRN, CMA-ES, paired confirmation
tailsafe/report/       the briefing (facts → template or LLM, number check, one-page PDF);
                       numbers only from saved results
tailsafe/vision/       floor-plan reader (detect.py), plan → building (graph.py),
                       rendered plans with truth (synth.py), evaluation
tailsafe/building/geometry.py  rectangle helpers shared by vision and micro
tailsafe/surrogate/    graph surrogate: training data, features, JAX GNN, evaluation,
                       predictor + shipped weights (weights/surrogate.npz)
tailsafe/api/          FastAPI app: jobs (progress over SSE, disk cache), views for the UI
                       static_site.py records responses for the browser version
web/                   React + TS + Vite + Tailwind + react-three-fiber UI (screens 1–9)
config/params.yaml     parameter registry
schemas/               generated JSON schemas (do not edit by hand; `make schema`)
docs/                  specification, architecture, validation, assumptions
```

## Publishing

Two ways for anyone to use TailSafe from a browser, on any device:

- **Browser version (GitHub Pages).** `.github/workflows/pages.yml` records the
  results (`tailsafe export-site`), builds the UI with `VITE_STATIC=1` and deploys
  on every push to `main`. One-time setup: Settings → Pages → Build and
  deployment → Source: **GitHub Actions**. Address:
  `https://<owner>.github.io/<repo>/`. Nothing runs on a server; the what-if
  network runs in the visitor's browser.
- **Full app (Hugging Face Space).** The `Dockerfile` runs the simulator, API and
  UI on port 7860. `.github/workflows/space.yml` pushes it to a Space on every
  push to `main` once configured: create a Space with the Docker SDK on
  huggingface.co, create a write token, then add the repository secret
  `HF_TOKEN` and the variable `HF_SPACE` (`<user>/<space>`). The free CPU tier
  has 2 cores, so expect roughly twice the run times of a 4-core laptop (not
  yet measured on a Space). Jobs run one at a time, so simultaneous visitors
  queue.

## Design decisions

Choices that are expensive to reverse, and the reasons for them:

1. **Stack:** Python 3.11+, NumPy/SciPy, Numba for the simulator kernel, NetworkX
   for graph analytics, Pydantic v2 models as the source of truth for the JSON
   schema, FastAPI backend, Typer CLI. Frontend: React + TypeScript + Vite +
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
5. **Walking density** excludes people standing in queues and is capped at the
   flow-maximising density 1/(2a); denser states are represented by queues.
   Without this, slow walkers cause a runaway density–speed collapse.
6. **Surrogate in JAX, not PyTorch Geometric.** The spec named PyG; its
   wheels (download.pytorch.org) were blocked by the development environment's
   network policy, so the message-passing network is written in JAX + optax
   (optional extra `tailsafe[surrogate]`). Same model class; switching back is
   a rewrite of `tailsafe/surrogate/model.py` only (features and data are
   framework-free).
7. **LLM briefing is opt-in.** It runs only when `ANTHROPIC_API_KEY` and
   `TAILSAFE_BRIEFING_MODEL` are set (no model is hard-coded) and the
   `briefing` extra is installed; otherwise the template briefing is used.
   Either way every number is checked against the facts JSON.

## Open questions

- Preferred sources for HK-specific values (stair widths, refuge floor interval,
  household composition, pre-movement times) — currently `ASSUMPTION`.
- Should reaching a refuge floor count as "safe" for RSET, or only final exits?
  (Current default: final exits; refuge arrival is recorded separately.)
- Is JuPedSim acceptable as an optional dependency for validating the
  person-by-person engine?
- "Twin-core private tower" is implemented as one core with a scissor (twin)
  staircase pair. Should it instead have two separate cores?
- Refuge floors: should occupants be forced to transfer between stairs at a
  refuge floor (stair discontinuity), as some codes require?

## Gotchas

- The Numba kernel is compiled on first use (~10 s) and cached on disk
  (`__pycache__`). After editing `tailsafe/sim/_kernel.py` the next run
  recompiles. `NUMBA_DISABLE_JIT=1` runs it as plain Python for debugging (slow).
- Web charts are hand-written SVG in `web/src/components/charts/`; follow the rules in
  `docs/architecture.md` (Web UI) — text never in series colours, a table view per chart.
- The 3D view needs WebGL; headless Chromium renders it with
  `--use-angle=swiftshader --enable-unsafe-swiftshader`.
- The micro kernel (`_micro_kernel.py`) also compiles on first use. It needs room
  polygons; `micro_problems(building)` says why a building can't be replayed.
  Its robustness rules (forced moves below jam density, overlap resolution,
  squeezing past in doorways/stand-offs) exist because pure collision-free
  speed models gridlock in head-on encounters; change them with the
  20/40-storey agreement runs (`tailsafe micro compare`) as the regression check.
- The plan reader assumes axis-aligned plans with walls thicker than other lines.
  Its evaluation is on *rendered* plans (`vision/synth.py`); never present those
  numbers as results on real drawings.
- The surrogate's weights (`tailsafe/surrogate/weights/`) are tied to the feature
  layout in `surrogate/features.py` and to the simulator. Changing either means
  regenerating data, re-running `surrogate eval`, retraining, and updating the
  numbers in `docs/validation.md` §10.
- Monte Carlo worker pools fork (workers inherit the compiled kernel) unless JAX is
  loaded in the process — forking JAX's threads can deadlock — then they use a fork
  server (`scenarios/montecarlo.py:_start_method`). Keep JAX imports lazy.
- Briefing text must only contain numbers present in `briefing_facts(...)`.
  When adding a sentence to `template_briefing`, put any new number in the facts
  first; `make_briefing` raises if the template breaks the rule.
- The browser version answers requests from recorded files keyed by a hash of the
  request (`canon` in `tailsafe/api/static_site.py` and `web/src/static/key.ts`
  must stay identical). If a screen changes the body it sends on the standard
  path, re-record (`tailsafe export-site`) or the browser version will say it has
  no results for those settings.
- `Building` caches lookups (`node_by_id` ...). Treat it as immutable; use
  `model_copy(update=...)`, which drops the caches.
