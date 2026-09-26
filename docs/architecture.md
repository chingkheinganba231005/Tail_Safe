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
