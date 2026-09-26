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
