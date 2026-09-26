# Architecture

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
                   attribution
                                     │
                                     ▼
              FastAPI backend  ◄──►  Web frontend
```

This document describes each implemented module and the design decisions behind
it. Sections are added as milestones land.

## Parameter registry (`config/params.yaml`, `tailsafe/config.py`)

- A YAML tree whose leaves are either fixed values (`value:`) or distributions
  (`dist:` = `truncnorm`, `lognormal`, `uniform`, `categorical`). Every leaf has a
  `source:`; unknown sources are the literal string `ASSUMPTION — needs citation`.
- `Params` validates the tree on load and gives dotted-path access.
- All sampling goes through the inverse CDF (`Param.ppf(u)`), so Latin Hypercube
  designs and common random numbers work identically for every distribution.
- `Params.with_overrides()` returns a modified copy for sensitivity studies;
  `Params.digest` is a stable hash used in result cache keys.

## Building model (`tailsafe/building/`)

| File | Role |
|---|---|
| `model.py` | Pydantic models: `Building`, `Level`, `Node`, `Edge`, `Stair`, `Lift`, `Feature` |
| `validate.py` | Semantic checks (references, stair direction, reachability of exits) |
| `io.py` | JSON load/save, JSON-schema export (`schemas/building.schema.json`) |
| `graph.py` | Directed arcs and a NetworkX view for analysis |
| `builder.py` | Geometry helpers and the double-loaded-corridor layout engine |
| `templates.py` | Procedural HK typologies: `cruciform`, `slab`, `twin_core`, `care_home` |
| `render.py` | Matplotlib plan + 3D stack rendering (Okabe–Ito palette) |

**Graph semantics.** Nodes are *places*: `unit`, `corridor`, `lobby`,
`lift_lobby`, `protected_lobby`, `stair_landing`, `refuge`, `open_area`, `exit`.
Edges are *physical connections stored once*: `flat` (walking between spaces),
`door` (through a doorway; carries the door's clear width, fire rating and
self-closing flag, plus the opening segment for drawing) and `stair` (the
flights between two landings on adjacent levels, always stored from the upper
landing to the lower). `graph.arcs()` expands each edge into directed arcs —
`flat`, `stair_down`, `stair_up` — which is what the simulators use. The spec's
"door" and "stair flight" node types are represented as edges because they are
capacity-limited connections rather than places; this keeps the queue network
simple (capacity lives on arcs) while every door and flight still has its own id
and plain-English label (e.g. "Stair A, 13/F → 12/F").

**Lifts** are entities listing the lobby node they stop at on each level, the
discharge level and whether they are the firefighting lift. The simulator
models them as a service (capacity, cycle time, priority rules), not as arcs.

**Geometry.** Every generated space carries a plan polygon; door edges carry the
opening as a plan segment; non-walkable features (lift shafts, refuse rooms,
voids) are stored per level. This is enough for rendering and, later, for the
microscopic simulator. Coordinates are rounded to millimetres.

**Procedural typologies.** Floors are described as double-loaded corridors:
a corridor split into *columns*, each with optional *bays* north and south (a
flat, a lift lobby, a stair enclosure with or without protected lobbies, a
refuge area, ...) and optional bays beyond either end. `add_corridor()` turns
that into nodes, edges and polygons, optionally rotated into place — the
cruciform block is four rotated corridors around a hand-laid core. Storey counts
include G/F (a 40-storey block has levels 0–39). Refuge floors replace flats
with refuge areas at the interval in `params.yaml`.

**Validation layers.** (1) JSON schema / Pydantic for shape and ranges;
(2) `validate_building()` for meaning — unique ids, resolvable references,
stairs pointing down between levels, non-stair edges within a level, lift stops
on the right level, and a reverse BFS proving every unit can reach an exit.
Tests additionally check generated plans for overlapping spaces and doors that
do not sit on their rooms.

## Synthetic population (`tailsafe/population/`)

- **Profiles** (`profiles.py`): able adult, child, older adult (65–79), frail
  older adult (80+), wheelchair user, domestic helper — each with horizontal,
  stair-down and stair-up speed distributions, a fatigue curve
  `m(n) = m_min + (1 − m_min)·exp(−n / n_e)` over storeys descended, a
  pre-movement multiplier and a space factor (able-adult equivalents).
- **Households** (`synth.py`): per unit, a household size (by unit type), ages
  (residential or care-home mix, optionally rescaled to a target 65+ share),
  wheelchair use, a live-in domestic helper, and presence by time slot
  (weekday/weekend × day/night). Care homes get staff who escort the most
  dependent rooms.
- **Groups.** Each occupied unit's present members form a group that moves at
  the pace of its slowest member; space is the sum of members' space factors.
  Evacuation *mode*: walk; carry a wheelchair user down (needs an able escort);
  wait for an evacuation lift (only if lifts are in evacuation service); or wait
  for fire-service rescue.
- **Behaviours.** Counter-flow visits to another flat (own floor or up to N
  floors above) with a dwell; rest stops at a refuge floor on the way down.
- **Common random numbers.** For every unit a fixed-shape block of uniforms is
  drawn per named stream (`tailsafe/rng.py`: occupancy, speeds, pre-movement,
  behaviour). Scenario knobs change only the transforms, so two scenarios with
  the same seed are coupled draw-for-draw.

## Mesoscopic simulator (`tailsafe/sim/`)

| File | Role |
|---|---|
| `network.py` | Building → arrays: arcs with effective width, capacity, area, storage |
| `routing.py` | Next-hop tables per blockage state and route class; stair-choice logit |
| `_kernel.py` | Numba time-stepping kernel (queues, merging, lifts, rescue, exposure) |
| `meso.py` | Scenario inputs, preparation, `run_meso()`, `MesoResult` |
| `cases.py`, `validation.py` | Idealised cases and the analytical validation report |
| `plot.py` | Evacuation curve and per-stair congestion heatmaps |

A **link-queue model** (in the family of MATSim's queue simulation) adapted to
pedestrians:

1. **Arcs.** Capacity = max specific flow × effective width (hydraulic model);
   storage = jam density × area; doors also count part of the room in front of
   them, where their queue forms.
2. **Walking.** Groups advance continuously along arcs; speed = profile speed
   (stair-down speed decays with fatigue) × hydraulic density factor × hazard
   multiplier. Density counts walkers in both directions (counter-flow), not the
   people standing in the end queue, and is capped at the flow-maximising
   density 1/(2a) because denser states are represented by queues. Without the
   cap, slow walkers trigger a runaway density–speed collapse.
3. **Queues.** A group reaching the end of an arc joins a FIFO queue and moves
   on when the next arc has inflow capacity this step (capacity is a token
   budget; a group spends its size) and storage space. Entry times carry the
   exact arrival time within the step, so free-flowing groups are not delayed
   by the time step. Groups blocked by storage for over 60 s squeeze in anyway
   (counted), which prevents counter-flow gridlock.
4. **Merging.** A smooth weighted round-robin shares the capacity of a
   contested arc among the queues feeding it; at stair landings the floor stream
   gets the deference ratio and the stair stream the rest, and unused share goes
   to whoever is waiting. This reproduces configured ratios to within 0.005.
5. **Routing.** Next-hop tables by estimated travel time (Dijkstra from the
   exits on the reversed graph), one per blockage epoch and route class
   (fastest / only stair *s*). Each group picks a staircase by a logit on
   extra travel time, or gets one assigned by floor. Blockages are discovered on
   arrival at the blocked arc. Waypoint tables send groups to a relative's flat,
   a refuge area or a lift lobby. A `Router` caches tables across scenarios.
6. **Lifts** in evacuation service start after a delay, pick floors top-down (or
   nearest / bottom-up), board FIFO up to car capacity, and unload at the
   discharge level; outages stop a lift after its current trip.
7. **Rescue.** Households that cannot self-evacuate wait. Fire-service teams
   start at the rescue time and take the lowest waiting floor first: climb,
   handle, carry down. When only rescue is left, the kernel skips ahead
   event by event.
8. **Exposure.** With a hazard field, groups accumulate FED at their location
   and are incapacitated at the threshold (removed from queues).

Performance: a 40-storey, ~1,850-occupant night scenario runs in 0.1–0.3 s on
one core after the one-off JIT compile (cached on disk).

## Scenarios, Monte Carlo and risk (`tailsafe/scenarios/`, `tailsafe/risk/`)

**Scenario specification** (`spec.py`). A `ScenarioSpec` (Pydantic, JSON/YAML)
fixes what is known — time slot, population mix, operational measures (phased
release, stair assignment, evacuation lifts and their priority rule) — and
describes what is uncertain with `Dist` objects: when named stairs become
impassable, whether a random stair is lost and when, how many lifts are out of
service, when fire-service rescue starts. `demo_spec()` is the pitch scenario.

**Sampler** (`sampler.py`). Scenario-level draws use a fixed 16-slot uniform
vector per scenario (rescue start, fire level, random blockage ×3, lifts out ×3,
named blockage times ×4, 4 reserved for the hazard model). Slots never move, so
any two specs evaluated on scenario *i* share draws. With LHS each batch of
indices is one Latin Hypercube design (`scipy.stats.qmc`), otherwise the vector
comes from the scenario's own stream. Occupant-level randomness comes from the
population streams of `(seed, i)`. Together these give **common random numbers**:
a baseline and an intervention see the same households, speeds, pre-movement
times and failures.

**Monte Carlo runner** (`montecarlo.py`). Batches of scenarios are split into
chunks and run in a process pool (fork on Linux, after compiling the kernel in
the parent). Each worker builds the network and a caching `Router` once.
Results are identical for any number of workers. Each run is reduced to a
`RunOutput`: scalar losses, sampled scenario info, per-household outcomes
(optional) and per-arc max queue and queue integral. With `target_halfwidth` the
runner stops once the bootstrap CI of CVaR₉₅ is narrow enough. `MCResult` saves
to `result.json` + `arrays.npz`.

**Losses.** Three per-scenario loss variables are reported:
- `total_time` — last occupant out, including fire-service rescue;
- `self_evacuation_time` — last occupant out on foot or by lift;
- `p95_occupant_time` — 95th percentile of occupants' exit times (robust to one
  straggler).

**Risk metrics** (`risk/metrics.py`). VaR (inverted empirical CDF) and CVaR by
the Rockafellar–Uryasev formula (exactly the mean of the worst 5% when that is
a whole number of runs), mean/median/P95/P99, all with vectorised percentile
bootstrap CIs. `paired_difference()` gives a paired-bootstrap CI for the change
in CVaR between two interventions evaluated with common random numbers.

**Who carries the tail** (`risk/breakdown.py`). In each tail scenario (loss ≥
VaR₉₅), the *stragglers* are households exiting after 90% of that scenario's
loss. Shares of stragglers by the household's most dependent member and by
floor band are compared with shares of all occupants (risk ratio), which gives
a plain-language headline.

Performance: 1,000 scenarios of the 40-storey demo block (~1,830 occupants)
take ~41 s on 4 cores (`make stress-demo`; target < 120 s, checked by the slow
test).

## Hazard model (`tailsafe/hazard/`)

A deliberately simple **multi-zone smoke network**, documented as an
engineering approximation rather than fire engineering:

| File | Role |
|---|---|
| `zones.py` | Zone network: volumes, exchange flows through openings, shafts, vents |
| `_kernel.py` | Numba integrators: zone transport and fused field derivation |
| `tenability.py` | Visibility, walking speed in smoke, CO/CO₂/heat FED (ISO 13571 style) |
| `model.py` | `FireSpec`, `HazardModel.run()` → `HazardResult` (fields + per-node ASET) |
| `external.py` | Import node time series from CFD (FDS) or network models (`.npz`) |
| `plot.py` | Worst visibility per floor over time (corridors, each stair) |

1. **Fire.** t² growth (medium/fast/ultra-fast, sampled) capped at a sampled
   peak HRR, in one flat. Its door is left open with a probability; its
   windows vent.
2. **Transport.** Each node is a well-mixed zone. A fuel-equivalent "products"
   tracer and the convective heat move along exchange flows. Open openings
   exchange `v_h × area`. Shut self-closing doors leak a fraction (less if
   fire-rated), and stair doors are also open part of the time while people
   pass. Stair flights and lift shafts exchange vertically with an upward bias
   (stack effect). Exits are sinks and refuge floors are ventilated. Explicit
   integration at the largest stable step: tracer mass is conserved except
   through sinks and vents (tested).
3. **Tenability.** Soot → extinction → visibility and walking-speed multiplier
   (Frantzich–Nilsson). CO with CO₂ hyperventilation, plus convective heat above
   ambient → FED rate. **ASET per node** = first time visibility < 10 m,
   T > 60 °C, or a person standing there since ignition reaches FED 0.3.
4. **Coupling.** The meso kernel reads the speed multiplier and FED rate on a
   10 s grid. Groups slow down in smoke, accumulate FED, and are incapacitated
   at FED 1. **RSET vs ASET** (`risk/tenability.py`): a floor fails if its last
   household leaves after the earliest ASET of its corridors and lobbies; a
   scenario fails if any floor fails or anyone reaches FED 0.3. Monte Carlo
   reports P(RSET > ASET) with Wilson intervals, per-floor failure
   probabilities and incapacitation statistics.
5. **Door hold-open policy.** Holding stair doors open restores their full
   flow capacity (self-closing doors otherwise lose 10%) but lets smoke into the
   stairs: a real trade-off for the optimiser.

Cost: ~45 ms per fire (1,500 zones, 2 h at 10 s records), so a 1,000-scenario
stress test with smoke takes about a minute on 4 cores.

## Bottleneck attribution (`tailsafe/analysis/`)

`attribute_bottlenecks(result)` (CLI: `tailsafe stress bottlenecks DIR`)
combines three views into one ranked table with plain-English labels:

1. **Recurrence** (`queue_recurrence`). In tail scenarios (loss ≥ VaR₉₅), the
   share where each arc's queue reaches a threshold (default 10 people), and the
   mean person-seconds of queueing there ("Stair B, 18/F → 17/F").
2. **Structure** (`structural.py`). Max-flow / min-cut from all flats to all
   exits with arc capacities in persons/s gives the building's best egress rate
   and the arcs that limit it. The flow-weighted load of each arc (expected
   users along the simulator's routes, split between stairs by the same logit)
   divided by its capacity is a structural clearance time.
3. **Counterfactual criticality.** Candidates: each staircase (all flights),
   each staircase's doors, each final exit, the element each worst queue is
   waiting for, and each staircase the spec blocks. For each, the scenarios are
   re-run with common random numbers with capacity × `factor` (default 1.5,
   via `capacity_multipliers`) or the stair kept usable (blockage moved to
   "never", so random-number slots stay aligned). The paired-bootstrap
   ΔCVaR₉₅ ranks the candidates.

   *Exactness.* Relaxing never slows a scenario (tested), so a scenario not
   re-run is bounded by its baseline loss. The runner starts with the worst 20%
   and keeps re-running any scenario whose baseline could still be in the new
   tail; when none can, the counterfactual CVaR equals that of a full re-run
   (tested). A capacity improvement shifts every scenario, so most get re-run.
   Re-running only the old tail would cap the estimated benefit at the
   80th-percentile baseline, a trap the first version fell into.

## Intervention optimiser (`tailsafe/optimize/`)

| File | Role |
|---|---|
| `plan.py` | `InterventionPlan`: evacuation lifts + dispatch rule, stair-door hold-open, stair assignment by floor band, phased release by band, floor wardens; `apply()` to a `ScenarioSpec`, `describe()` in plain English |
| `search.py` | `Objective` (CVaR, P(RSET > ASET), or weighted mean + CVaR), `optimize()`, paired confirmation |
| `cmaes.py` | Small deterministic CMA-ES for the continuous phasing delays |
| `plot.py` | Before/after distributions on the same scenarios |

**Sample-average approximation with common random numbers.** Every candidate
plan is simulated on the same `n_scenarios` draws (seed fixed), through one
reusable `MonteCarloPool`, so differences between plans are not swamped by
scenario noise. The search:

1. *Screen* single levers: three lift dispatch rules, door hold-open, stair
   assignments (split level × stair order), three phasing presets (upper floors
   first, lower floors first, fire floor and the floor above first), and a
   warden on each of a few candidate floors (the fire floor, floors whose
   residents most often wait for rescue, slowest-clearing floors).
2. *Combine* greedily: start from the best lever, add each other improving
   lever's best setting while the objective improves.
3. *Wardens*: add more, greedily, up to `max_wardens`.
4. *Refine* phasing delays with CMA-ES (delays rounded to 30 s so repeated
   points hit the cache) — only when a phasing preset beat the baseline during
   screening. Otherwise CMA-ES tends to "find" tiny in-sample gains from
   holding floors back that do not survive confirmation and raise
   P(RSET > ASET), because held residents wait in their flats while smoke
   spreads.
5. *Confirm* baseline vs best plan on **fresh** scenarios (a different seed):
   paired-bootstrap CIs for ΔCVaR₉₅ of each loss and for ΔP(RSET > ASET). The
   in-sample optimum is biased low (winner's curse); the confirmation is what
   `optimize` reports as significant or not.

**Wardens** (`scenarios/sampler.apply_wardens`) are a deterministic transform
of the sampled population, so common random numbers are kept. Households on
covered floors react no later than the warden's sweep time, and each warden
escorts one household that would otherwise wait for rescue down the stairs.

CLI: `tailsafe optimize cruciform --spec demo --scenarios 100 --confirm 400 --out out/opt`.
The GNN surrogate (M10) will pre-screen candidates here; the simulator will
still confirm finalists.

## Web API (`tailsafe/api/`)

| File | Role |
|---|---|
| `app.py` | FastAPI app: buildings, stress tests, bottlenecks, optimisation, replays; serves the built web UI from `web/dist` when present |
| `jobs.py` | `JobManager`: one background job at a time (each uses a process pool), progress counters, results cached on disk by request key |
| `views.py` | Compact JSON views for the browser: risk summary, histograms' raw losses, tail breakdowns, stair congestion by level, replay frames |

Long work never blocks a request. `POST /api/stress`, `/api/bottlenecks`,
`/api/optimize` and `/api/replay` return a job; the browser follows
`GET /api/jobs/{id}/events` (server-sent events, one status message per
change) and then fetches `GET /api/jobs/{id}/result`. The cache key combines
the building digest, the request and the parameter-registry digest, so a
repeated request is answered from `runs/cache/` (or `$TAILSAFE_CACHE_DIR`)
without simulating. Non-finite floats are sent as `null` (strict JSON).
Buildings are stored by digest; uploaded JSON is validated before use.
Every result carries the responsible-use disclaimer.
