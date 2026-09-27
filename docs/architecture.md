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

This document describes each module and the design decisions behind it.

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
service, when fire-service rescue starts. `reference_spec()` is the reference scenario (spec §9).

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

Performance: 1,000 scenarios of the 40-storey reference block (~1,830 occupants)
take 65–75 s on 4 cores with the smoke model (`make stress-example`; the target
is under 120 s, checked by a slow test). Scenarios are handed to workers in chunks of at most 10, and smaller for short runs (tail
re-runs, optimiser samples) so every worker stays busy; each scenario is
seeded by its index, so chunking never changes results.

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
| `plan.py` | `InterventionPlan`: evacuation lifts + dispatch rule + eligibility, stair-door hold-open, stair assignment by floor band, phased release by band, floor wardens; `apply()` to a `ScenarioSpec`, `describe()` in plain English |
| `search.py` | `Objective` (CVaR, P(RSET > ASET), or weighted mean + CVaR), `optimize()`, paired confirmation |
| `cmaes.py` | Small deterministic CMA-ES for the continuous phasing delays |
| `plot.py` | Before/after distributions on the same scenarios |

**Sample-average approximation with common random numbers.** Every candidate
plan is simulated on the same `n_scenarios` draws (seed fixed), through one
reusable `MonteCarloPool`, so differences between plans are not swamped by
scenario noise. The search:

1. *Screen* single levers: three lift dispatch rules × two eligibility rules
   (all mobility-impaired residents, or wheelchair users only), door hold-open, stair
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

CLI: `tailsafe optimize cruciform --scenarios 100 --confirm 400 --out out/plan`.
The graph surrogate does not pre-screen candidates yet: every candidate plan is
simulated.

## Web API (`tailsafe/api/`)

| File | Role |
|---|---|
| `app.py` | FastAPI app: buildings, stress tests, bottlenecks, optimisation, replays, micro replays, floor plans, surrogate, briefing; serves the built web UI from `web/dist` when present |
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

## Microscopic replay engine (`tailsafe/sim/micro.py`, `_micro_kernel.py`)

The meso engine is fast enough for thousands of Monte Carlo runs; the micro
engine replays *selected* scenarios (median, worst, before/after) person by
person, for animation and to cross-check the meso engine.

- **Movement:** a collision-free speed model (model form after Tordeux,
  Chraibi & Seyfried, 2016). Each person is a disc; the walking direction is
  the direction to the next doorway plus exponential repulsion from
  neighbours and walls; the speed is `min(v0, max(0, gap / T))`, where `gap`
  is the free distance to the nearest person ahead *in the direction actually
  taken*. `v0` is the household's speed times the smoke factor of the room.
- **Space:** each node's polygon (axis-aligned rectangles in the templates)
  is a room. Rooms connect through *portals*: door openings, or the shared
  boundary of adjacent corridor segments clipped to the walkway width.
  People outside a doorway's span first walk to a point in front of it.
- **Doorways** have lanes (width ÷ `movement.micro.lane_width`, at least
  one); each lane lets the next person through only one headway
  (`T + 2r / v`) after the previous one — the same headway the speed model
  keeps in a queue — so a door's capacity grows with its width. A person
  enters the next room only if there is room at the entry point.
- **Stairs:** a flight is a strip of lanes between side-by-side points at
  the end of the landings (dog-leg geometry): people step onto the flight
  from a line across its width, keep the time gap to whoever is ahead in
  their lane, and step off at the next landing when there is room.
- **Decisions are shared** with the meso engine through `meso.prepare()`:
  reaction times, stair choice and assignments, counter-flow and refuge
  waypoints, blockages and when each person discovers them, smoke-reduced
  speeds. The two engines differ only in movement and queueing, so their
  comparison isolates the flow model.
- **Not walked:** households waiting for an evacuation lift or for
  fire-service rescue take their exit times from the meso run of the same
  scenario; toxic dose is not recomputed.
- **Robustness rules** (documented because they shape results): forced
  moves after 30 s of being blocked, but only into rooms below jam density;
  a small overlap-resolution step; two people facing each other in a doorway,
  or stuck face to face for 5 s, squeeze past each other (swap places).
- **Output:** exit and left-floor times per person, and frames every 2 s
  (floor, x, y, state per person) for the replay screen.

`tailsafe.analysis.agreement` runs both engines on the same scenarios and
reports bias (with a bootstrap CI), relative bias, correlation, RMSE and the
largest difference for the times by which half, 95% and all of the walkers
are out. CLI: `tailsafe micro run | compare | fd`. API: `POST /api/micro`,
then `GET /api/micro/{job}/level/{level}` for one floor's frames.

## Floor-plan ingestion (`tailsafe/vision/`)

| File | Role |
|---|---|
| `raster.py` | Load PNG / JPEG / data URLs, and the first page of a PDF when the optional `pypdfium2` is installed |
| `detect.py` | Walls, doorways, rooms, stairs, room types → `PlanDetection` (plain JSON in image pixels) |
| `graph.py` | `plan_to_building`: stack the corrected floor into N storeys; `assign_doors` for doorways drawn in the editor |
| `synth.py` | Render plans from buildings with ground truth (tests, evaluation, the "sample plan" in the UI) |
| `evaluate.py` | Precision / recall of doorways and stairs on the rendered suite |
| `overlay.py` | Detection drawn over the image (CLI) |

Pipeline (classical image processing; NumPy / SciPy only):

1. **Ink** by Otsu's threshold (inverted if the paper is dark).
2. **Wall thickness** = upper quartile of the lengths of ink runs that cross a
   line continuing on both sides; thin lines (text, treads, furniture,
   door swings) give crossings of one or two pixels.
3. **Walls** = ink after a morphological opening a little smaller than that
   thickness.
4. **Scale** from a reference line drawn by the user; otherwise from the
   wall thickness assuming `vision.default_wall_thickness` (with a warning).
5. **Doorways**: gaps between `door_width_min` and `door_width_max` along a
   column or row whose two ends are the *ends of a wall running that way*
   (a jamb, narrow across) — so the gap between the two walls of a corridor
   is not mistaken for a door; plus **corridor mouths**, the gap between the
   free ends of two parallel walls (an entrance filling the end of a
   corridor).
6. **Rooms** = connected free space with doorways sealed; spaces touching
   the image border are outside. Each room is covered by up to a dozen
   greedy largest rectangles, grown by half a wall where a wall runs along
   the side, so rooms meet on wall centre lines and doorways lie on shared
   boundaries (as in the templates; the micro engine can replay them).
7. **Stairs**: thin lines inside a room that cross most of it and repeat at
   `stair_tread_min`–`stair_tread_max` (autocorrelation of the line profile).
8. **Types**: stair; void (no doorway); corridor (long and narrow); lobby
   (three or more doorways); otherwise a flat, with the household-size prior
   chosen by area (`unit_area_small_max`, `unit_area_medium_max`).
9. **Doorway → rooms**: the room labels a little way out on either side.

`plan_to_building` repeats the floor for every storey, makes one staircase
per stair room (landings stacked and joined by flights), keeps doorways with
their openings (stair doors fire-rated and self-closing), turns doorways to
the outside on G/F into exits, and — when the plan has none (a typical floor)
— assumes an exit at the foot of every staircase and records
`metadata.exits_assumed`. The result is validated like any other building.

**Human in the loop** (Building screen → *Floor plan image*): draw a reference
line and give its length, read the plan, click a room to change its type
(flat, corridor, lobby, staircase, refuge area, not walkable), click a
doorway to delete it or draw new ones, choose the number of storeys, build.
API: `POST /api/vision/detect` (base64 image, optional scale),
`POST /api/vision/build`, `GET /api/vision/sample`. CLI:
`tailsafe vision synth | detect | build | eval`.

## Graph surrogate (`tailsafe/surrogate/`)

| File | Role |
|---|---|
| `data.py` | Random (building, scenario) cases — template and options drawn from `TYPOLOGIES`, scenario settings drawn at random — each simulated with 64 Latin-Hypercube Monte Carlo runs; keeps every loss sample and the mean queueing per edge (JSON lines) |
| `features.py` | The circulation graph: corridors, lobbies, landings, refuges and exits as nodes (flats folded in as counts and expected occupants), walkways, doors and flights as directed edges with widths, lengths and compiled capacities; scenario settings as node / edge flags and a global vector |
| `model.py` | Message-passing graph network in JAX (encoders, 5 rounds with a global context, monotone quantile head, edge head), pinball-loss training with optax, save / load (`.npz` weights + `.json` config, feature statistics and metadata) |
| `evaluate.py` | Random 80/20 split and leave-one-typology-out: quantile errors, CVaR₉₅ error, R², coverage, edge rank correlation, speed |
| `predictor.py` | `Surrogate`: loads the shipped weights (`weights/surrogate.npz`), caches building graphs, predicts in milliseconds |

The network predicts, for each loss, P50 / P75 / P90 / P95 (built as a
positive base plus positive increments, so they never cross) and CVaR₉₅
(P95 plus a positive increment), and the mean queueing on every edge. It is
trained against **all** 64 simulated outcomes of a case with the pinball
(quantile) loss rather than against the case's own noisy quantile estimates,
plus a squared error on the sample CVaR₉₅ and on log edge queueing. A
global context vector (mean and max over nodes) feeds every node update, so
information crosses a 40-storey building in five layers. Outcomes that never
happen within the 4-hour horizon are censored at the horizon, as in the
stress-test report.

The simulator stays the source of truth: the what-if screen (screen 8) shows
the surrogate's estimate instantly and offers *Confirm with full simulation*,
which runs a real stress test and plots both. API:
`POST /api/surrogate/predict` (503 when the optional extra or the weights are
missing). CLI: `tailsafe surrogate data | eval | train`.

The spec named PyTorch Geometric. Its wheels could not be downloaded in the
development environment, so the same model class is written directly in JAX
(`pip install 'tailsafe[surrogate]'` pulls `jax[cpu]` and `optax`). This is
recorded as an open decision in `DEVELOPMENT.md`.

## Briefing (`tailsafe/report/`)

| File | Role |
|---|---|
| `briefing.py` | Facts from saved results, the template briefing, the optional LLM briefing with its number check, the one-page PDF |

`briefing_facts` turns a stress test (and, when available, the bottleneck
table and the optimised plan) into a small JSON document with every number
already rounded the way it may appear. Two writers use only that document:

* **Template** (always available): fixed sentences filled from the facts.
* **LLM** (optional): when `ANTHROPIC_API_KEY` and `TAILSAFE_BRIEFING_MODEL`
  are set and `pip install 'tailsafe[briefing]'` is done, an Anthropic model
  drafts the text. The prompt forbids any number that is not in the facts,
  and `unknown_numbers` checks the draft: every number in it must appear in
  the facts (signs and trailing zeros aside). A draft that fails, or an API
  error, falls back to the template with a note saying why. The template is
  checked the same way; a failure there is a bug and raises.

`briefing_pdf` lays the Markdown out on one A4 page with Matplotlib (no extra
dependency) and adds the distribution of the time until everyone is out —
before and after the plan when there is one. The scenario sentence is built
from the scenario's settings, never from its free-text description.

CLI: `tailsafe brief` (Markdown, the facts as JSON, PDF). API:
`POST /api/briefing` (results in, checked briefing out) and
`POST /api/briefing/pdf`.

## Browser version (`tailsafe/api/static_site.py`, `web/src/static/`)

The same web UI, served as static files (GitHub Pages) with no Python server.

* **Recording.** `tailsafe export-site --out web/public/data` drives the API
  in-process with exactly the requests the UI sends on the standard path for
  each building type (standard building, reference scenario fitted to it, 300-run
  stress test, bottlenecks, optimised plan, 3D and before/after replays, the
  person-by-person replay of the worst run with every floor, briefings and
  PDFs), and writes each response to `r/<key>.json` (or `.pdf`).
* **Keys.** A key is two FNV-1a hashes of `METHOD path canonical-body`, where
  the canonical body has sorted keys and integral numbers written without a
  decimal point (`canon` in Python and TypeScript; the briefing endpoints are
  keyed by building and which results are included). Tests on both sides
  check the same reference hashes. Job ids are replaced by ids derived from
  the request, so follow-up requests (bottlenecks of a stress test, floors of
  a replay) have stable keys.
* **In the browser.** With `VITE_STATIC=1` the API client (`web/src/api.ts`)
  answers from those files; a request that was not recorded gets a plain
  message ("run TailSafe on your computer, or use What-if"). The scenario
  screen notices when the scenario differs from the recorded one and offers
  to reset it. The what-if network runs in the browser: `web/src/static/
  surrogate.ts` ports the feature construction, the message-passing network
  and the coverage notes; the exporter writes float32 weights, feature
  statistics and each building's graph. A unit test compares the port with
  predictions from the Python model.
* **Deploying.** `.github/workflows/pages.yml` records the data (simulation
  results cached by a hash of the code and parameters), builds with
  `VITE_BASE=/<repo>/` and deploys to GitHub Pages on every push to `main`.

## Web UI (`web/`)

React + TypeScript (Vite), Tailwind, react-three-fiber. It talks only to the
job API above; `vite dev` proxies `/api` to the backend, and `make web` builds
`web/dist`, which the FastAPI app serves at `/`.

| Screen | What it shows |
|---|---|
| 1 Building | Template gallery with options, a floor-plan image read by `tailsafe.vision` with a correction editor, or JSON upload; plan of any floor; stair and exit widths editable; confirm |
| 2 Scenario | Time of day, age mix, vacancy, counter-flow, fire floor, smoke on/off, stair blockages (fixed or random time), random stair loss, lifts out, evacuation lifts, rescue teams; runs and seed |
| 3 Stress results | Histogram with mean / P95 / CVaR₉₅ markers, stat tiles with CIs, P(RSET > ASET) meter, who is in the tail (risk ratios), floors that fail, stair queues by floor |
| 4 3D stack | One scenario re-simulated with time series: translucent floors coloured by smoke, stair columns by queue length, blocked stairs, scrubber, people still on each floor, evacuation curve |
| 5 Replay (people) | The micro engine's replay of one scenario, top-down, one floor at a time, with the meso/micro comparison for that scenario and people on each floor over time |
| 6 Bottlenecks | Counterfactual ranking with CIs; clicking a row highlights the element in the plan and in 3D; where queues recur |
| 7 Optimise | Objective, levers and sample sizes; plan in plain English; paired confirmation with verdicts; before/after distributions on identical scenarios and the worst confirmation scenario replayed side by side on one clock |
| 8 What-if (live) | Scenario controls; the surrogate's P50–P95 range and CVaR₉₅ for each outcome as the controls move, where queues are expected, and *Confirm with full simulation* overlaying a 300-run stress test |
| 9 Briefing | The checked briefing (template or LLM draft, with the reason when a draft was rejected) and *Download PDF* |

Charts are small hand-written SVG components (`web/src/components/charts/`)
following one set of rules: one hue per single-series chart, fixed categorical
order when there are two series, a legend for two or more series, text in ink
colours only, thin bars with 4 px rounded data ends, hairline grids, a hover
tooltip on every mark, and a table view for every chart. Sequential ramps are
one hue (blue for queues, orange for smoke) and reverse in dark mode so that
"near zero" recedes into the background. Status colours appear only with an
icon and a label. Light and dark themes are both specified (`styles.css`);
the header toggle overrides the OS setting.

