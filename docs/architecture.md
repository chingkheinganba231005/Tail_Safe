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
