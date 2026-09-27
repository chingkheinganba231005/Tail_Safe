# TailSafe — Product specification

> **Tail-risk evacuation stress-testing for high-rise Hong Kong.**

---

## 0. What TailSafe is

**TailSafe** is a system that stress-tests how people evacuate Hong Kong high-rise buildings under realistic, uncertain, worst-case conditions, and then recommends interventions that shrink the *worst* outcomes, not just the average.

Engineering principles:
- Decisions that are expensive to reverse (framework choice, data schema, simulation model) are recorded with their reasons in `DEVELOPMENT.md`.
- **Determinism:** every stochastic component takes an explicit seed. Same seed → same result.
- **Never invent citations or statistics.** Put every physical/demographic parameter in `config/params.yaml` with a `source:` field. If the source is not known, write `source: ASSUMPTION — needs citation` so it can be filled in later.
- Prefer clarity over cleverness. Type hints everywhere, docstrings on public functions, `ruff` + `mypy` clean.

---

## 1. Problem and motivation

Hong Kong has some of the densest, tallest residential stock in the world: 30–50+ storey public and private housing towers, long single-stair or scissor-stair cores, refuge floors, and a rapidly ageing population (65+ projected to reach roughly 36% of the population by 2046, excluding foreign domestic helpers — verify and cite the latest Census & Statistics Department projection). Many elderly residents live alone or depend on a domestic helper to move.

Evacuation planning today typically assumes average, able-bodied occupants, a single "design" scenario, and deterministic timings. Real emergencies are driven by the tail: a blocked stair, smoke in the core, residents asleep at 3 a.m., a wheelchair user on the 38th floor. TailSafe asks: **"What does the bad 5% of outcomes look like for this building, why, and what cheap operational changes fix it?"**

---

## 2. Novelty — what makes this different

Existing tools (Pathfinder, MassMotion, buildingEXODUS, open-source JuPedSim/Vadere) are powerful but expert-oriented: models are built by hand, a few scenarios are run, and results are mostly averages. TailSafe's contributions:

1. **Tail-risk objective (finance → fire safety).** Treat evacuation time as a loss distribution and optimize **CVaR₉₅** (expected evacuation time in the worst 5% of scenarios) and **P(RSET > ASET)**, borrowing risk measures from portfolio management. Interventions are ranked by how much they cut the tail, not the mean.
2. **Hybrid meso–micro simulation.** A fast mesoscopic network-flow engine runs thousands of Monte Carlo scenarios in seconds; a microscopic agent (social-force) engine replays only the most informative scenarios (e.g., the CVaR tail) for realistic animation. Speed where you need volume, fidelity where you need insight.
3. **Hong Kong–specific behaviour and typology.** Helper-assisted elderly pairs, family groups, night-time sleeping occupants with long pre-movement times, counter-flow (people going back for relatives/belongings), stair-descent fatigue over 30+ floors, refuge floors, scissor stairs, and procedural templates of common HK tower layouts.
4. **Occupant-location uncertainty.** Occupancy is not fixed; it is sampled from time-of-day × unit-type priors (weekday 3 p.m. vs Sunday 3 a.m. are different buildings).
5. **Coupled hazard model.** A lightweight zone-based smoke model on the building graph (with stack-effect bias in shafts) degrades visibility, walking speed, and accumulates **Fractional Effective Dose (FED)** per agent — so "unsafe" means tenability failure, not just a slow clock.
6. **Counterfactual bottleneck attribution.** Bottlenecks are identified by *causal* contribution to tail outcomes (what happens to CVaR if this door were wider / this stair unblocked), not just by where crowds look dense.
7. **Optimizing operations, not concrete.** Search over cheap interventions — phased evacuation order, warden placement, evacuation-lift allocation for mobility-impaired residents, stair assignment, door hold-open policies — using variance-reduced stochastic optimization.
8. **GNN surrogate for instant what-if.** A graph neural network trained on simulator outputs predicts evacuation-time distributions and edge congestion in milliseconds, enabling live interactive exploration and fast pre-screening during optimization.
9. **Floor plan → egress graph.** Computer-vision ingestion of floor-plan images into a navigable graph, with a human-in-the-loop correction editor.
10. **Grounded plain-language briefings.** An LLM turns the metrics JSON into a building-manager briefing, strictly constrained to the computed numbers.

---

## 3. System architecture

```
 Floor plan (PNG/PDF) ─┐
 Procedural HK template ├─► Building Model (multi-floor graph + geometry)
 Structured JSON ───────┘            │
                                     ▼
            Scenario Sampler (occupants, hazards, blockages, time of day)
                                     │
             ┌───────────────────────┼─────────────────────────┐
             ▼                       ▼                         ▼
   Meso simulator (fast)    Hazard model (smoke/FED)    Micro simulator (replay)
             │                       │                         │
             └──────────► Monte Carlo runner ◄──────────────────┘
                                     │
          ┌──────────────┬───────────┼──────────────┬──────────────┐
          ▼              ▼           ▼              ▼              ▼
    Risk metrics   Bottleneck   Optimizer      GNN surrogate   LLM report
   (CVaR, P95…)    attribution  (interventions)
                                     │
                                     ▼
              FastAPI backend  ◄──►  Web frontend (2D/3D, charts, before/after)
```

---

## 4. Module specifications

### 4.1 Building model (`tailsafe/building/`)
- Multi-floor directed graph. **Node types:** room/unit, corridor segment, lobby (incl. protected lobby), door, stair landing, stair flight, lift lobby, refuge floor area, final exit. **Edge attributes:** length (m), clear width (m), type (flat/stair-down/stair-up/lift), capacity (persons/s), fire-rated flag, can_block flag.
- Each floor also stores 2D polygon geometry (walls, openings) for micro-simulation and rendering.
- Stacking: a "typical floor" template replicated N times, with ground-floor, refuge-floor, and podium variants.
- Serialization to a documented JSON schema (`schemas/building.schema.json`) with validation.
- **Procedural generator** of HK typologies: cruciform public-housing block, slab block, twin-core private tower with scissor stairs, elderly care home (low-rise, high-dependency). Parameters: floors, flats per floor, stair count/width, refuge floor interval, lift count.

### 4.2 Occupant model (`tailsafe/population/`)
- Agent profiles: able adult, child, older adult (65–79), frail older adult (80+), wheelchair user, domestic helper. Each has a walking-speed distribution, stair-descent speed, fatigue curve, pre-movement time distribution, and body size.
- Behaviour: family groups move together (speed = slowest member); **helper-assisted pairs** (helper escorts older adult, reduces helper speed, increases stair occupancy); a configurable fraction perform **counter-flow** (go back to a unit before leaving); a fraction **wait for lift assistance** if mobility-impaired.
- Occupancy priors by time slot (weekday day, weekday night, weekend) × unit type. Sampled per scenario.
- Pre-movement time: lognormal, with a larger scale for night/sleeping scenarios. All parameters in `params.yaml` with sources.

### 4.3 Mesoscopic simulator (`tailsafe/sim/meso.py`)
- Time-stepped queue-network model: agents traverse edges; each edge/door has a flow capacity derived from clear width × specific flow; stair speed depends on density (fundamental diagram); merging at stair landings uses a configurable deference ratio between floor inflow and stair flow.
- Supports: blocked edges (appearing at time t), lifts used for evacuation (capacity, cycle time, priority rules), refuge floor holding, phased release of floors.
- Vectorized with NumPy (+ Numba where hot). **Performance target:** 40-storey tower, ~2,000 occupants, 1,000 Monte Carlo runs in < 2 minutes on a laptop CPU, parallelized with multiprocessing.
- Outputs per run: per-agent exit time, per-agent FED, per-edge queue length time series, total evacuation time.

### 4.4 Microscopic simulator (`tailsafe/sim/micro.py`)
- Social-force (or collision-free speed) model on the 2D floor geometry, with stairs connecting floors.
- Used only to replay selected scenarios (median, P95, worst, and before/after of an intervention) for animation and to cross-check the meso engine.
- Optional: an adapter to JuPedSim for validation if it simplifies things — to be agreed with the project owner.

### 4.5 Hazard model (`tailsafe/hazard/`)
- Zone model on the building graph: fire origin node, smoke produced over time, spreads through open doors and vertical shafts (stair and lift) with an upward stack-effect bias; fire-rated doors and protected lobbies slow spread.
- Per node: smoke concentration → visibility → walking-speed multiplier; toxic dose accumulation → **FED** per agent. FED ≥ threshold = incapacitation.
- Clearly documented as a simplified engineering approximation; provide an interface to import time-series from a real CFD tool (e.g., FDS) later.
- Per-node **ASET** (available safe egress time) derived from tenability limits.

### 4.6 Scenario sampler & Monte Carlo runner (`tailsafe/scenarios/`)
- Random variables: occupancy, profiles, pre-movement times, fire origin, which stairs/doors are blocked and when, lift availability, smoke growth rate.
- Support **common random numbers** so different interventions are compared on identical scenario draws (variance reduction).
- Latin Hypercube sampling option; convergence check (stop when CVaR₉₅ CI half-width < tolerance).

### 4.7 Risk metrics (`tailsafe/risk/`)
- Total evacuation time: mean, median, P95, P99, **CVaR₉₅**, with bootstrap confidence intervals.
- **P(RSET > ASET)** building-wide and per floor.
- Expected number of occupants incapacitated / not yet evacuated when ASET is reached.
- Breakdowns by occupant profile (e.g., "frail older adults above floor 30 carry 70% of tail risk").

### 4.8 Bottleneck attribution (`tailsafe/analysis/bottlenecks.py`)
- Recurrence: edges whose queue exceeds a threshold in ≥ X% of tail scenarios.
- Structural: min-cut and flow-weighted betweenness of the egress graph.
- **Counterfactual criticality:** for top candidate edges, re-run the tail scenarios with the edge relaxed (wider/unblocked) and measure ΔCVaR₉₅. Rank edges by ΔCVaR.
- Output a ranked table with plain-English labels ("Stair B landing at 12/F").

### 4.9 Intervention optimizer (`tailsafe/optimize/`)
- Decision variables: phased floor release schedule (time offsets), warden positions (k nodes), lift allocation rules for mobility-impaired occupants, stair assignment by floor band, door hold-open policy.
- Objective: minimize CVaR₉₅ of evacuation time (default) or P(RSET > ASET), subject to constraints (e.g., max wardens, lift capacity). Allow a weighted mean + CVaR objective.
- Methods: CMA-ES or genetic algorithm for schedules, greedy/submodular heuristic for warden placement; sample-average approximation with common random numbers; GNN surrogate pre-screens candidates, the true simulator confirms the finalists.
- Output: best intervention set, before/after distributions, improvement with confidence intervals.

### 4.10 GNN surrogate (`tailsafe/surrogate/`)
- Graph neural network (specified in PyTorch Geometric; implemented in JAX, see `DEVELOPMENT.md`). Input: building graph with node/edge features + scenario/intervention features. Output: quantiles of total evacuation time and per-edge congestion.
- Train on simulator-generated data across procedurally generated buildings; hold out whole building typologies for testing (report generalization honestly).
- Metrics: quantile loss, calibration of predicted quantiles, speedup vs simulator.

### 4.11 Floor plan ingestion (`tailsafe/vision/`)
- Input: PNG/JPG/PDF floor plan. Pipeline: preprocessing → wall segmentation (classical morphology/Hough baseline; optional learned segmentation model) → door detection (arc/opening detection) → room segmentation → stair symbol detection → graph extraction.
- Scale calibration from a user-drawn reference line of known length.
- **Human-in-the-loop editor** in the frontend: user corrects walls/doors/stairs, tags refuge areas, then confirms. Full automation is not required; a reliable semi-automatic flow is.
- Evaluate on a small hand-labelled set; report precision/recall for doors and stairs.
- The main path never depends on the floor-plan reader: procedural templates are the default.

### 4.12 LLM briefing (`tailsafe/report/`)
- Optional (uses `ANTHROPIC_API_KEY` if present; otherwise a template report).
- Input only the metrics/bottleneck/optimizer JSON. Prompt must forbid numbers not present in the input; run a post-check that every number in the output appears in the JSON.
- Output: 1-page briefing for a building manager + exportable PDF.

---

## 5. Frontend (`web/`)

React + TypeScript (Vite or Next.js), Tailwind, react-three-fiber for 3D, a charting library for distributions.

Screens:
1. **Building setup** — pick a procedural template or upload a plan → correction editor → confirm.
2. **Scenario builder** — time of day, fire origin, blocked exits, population mix sliders (e.g., % aged 80+), lift availability.
3. **Stress test results** — evacuation-time distribution with mean/P95/CVaR markers, P(RSET>ASET) gauge, risk breakdown by profile and floor.
4. **3D stack view** — the tower as stacked translucent floors; congestion heatmap over time; smoke overlay; scrubber timeline.
5. **Replay** — micro-sim animation of the worst-case scenario, top-down per floor.
6. **Bottlenecks** — ranked table; clicking highlights the edge in 2D/3D.
7. **Optimize** — choose objective and constraints, run, then **before/after split-screen** (distributions + animation side by side).
8. **What-if (live)** — sliders hit the GNN surrogate for instant estimates, with a "confirm with full simulation" button.
9. **Report** — generated briefing, export to PDF.

Design: clean, calm, accessible (colour-blind-safe heatmaps), readable on a phone as well as a large screen.

---

## 6. Backend

- FastAPI; long simulations as background jobs with progress via WebSocket/SSE.
- Endpoints for buildings, scenarios, runs, results, optimization, surrogate inference, reports.
- Results cached by (building hash, scenario config hash, seed).

---

## 7. Validation (must be in the repo, with a `docs/validation.md`)

- **Analytical check:** simple single-stair and corridor cases compared against hydraulic flow calculations; meso results within a documented tolerance.
- **Fundamental diagram:** micro engine reproduces a reasonable speed–density relationship.
- **Meso vs micro agreement** on the same scenarios (report correlation and bias).
- **Monotonicity tests:** blocking a stair never decreases evacuation time; adding exit width never increases it (within MC noise).
- **Convergence:** CVaR estimates stabilize as runs increase.
- Unit tests for every module; an end-to-end test that runs a small tower through the full pipeline.

---

## 8. Repo structure

```
tailsafe/
  building/  population/  sim/  hazard/  scenarios/
  risk/  analysis/  optimize/  surrogate/  vision/  report/
  api/
config/params.yaml
schemas/
data/templates/   data/floorplans/   data/labels/
web/
tests/
notebooks/        # experiments, surrogate training
docs/  (spec.md, architecture.md, validation.md, assumptions.md)
DEVELOPMENT.md  README.md  pyproject.toml  Dockerfile
```

---

## 9. Reference scenario

"Sunday, 3 a.m., 40-storey public housing block, 22% of residents aged 65+, fire on 14/F, Stair A smoke-logged at t = 4 min, one lift out of service."
The default scenario in the CLI (`--spec reference`) and the web UI. The standard path through the tool follows it: baseline distribution → the tail → where it comes from (bottleneck + which residents) → optimized operational plan → before/after animation → one-page briefing.

---

## 10. Responsible use

- TailSafe is an educational and decision-support prototype, **not** a substitute for a registered fire engineer, the Fire Services Department, or compliance with the Buildings Department's Code of Practice for Fire Safety in Buildings. Show this in the UI and README.
- No personal data. All occupants are synthetic.
- Document every simplifying assumption in `docs/assumptions.md`.
