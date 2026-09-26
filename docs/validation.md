# Validation

Validation checks are executable. `tests/validation/` asserts them in CI, and
`tailsafe validate --markdown` regenerates the table below from the current
code and parameters.

## 1. Analytical checks of the mesoscopic engine (M2)

Idealised cases (`tailsafe/sim/cases.py`) with identical occupants moving at
the hydraulic model's unimpeded speeds (0.85 k: 1.19 m/s level, 0.92 m/s on
7/11 stairs), all ready at t = 0, compared with hand calculations:

- **Corridor:** `T = walk distance / speed + N / door flow`, door flow =
  1.3 p/s/m × (clear width − 2 × 0.15 m boundary layer).
- **Single stair:** `T = first arrival from 1/F + P / stair flow`, stair flow =
  1.01 p/s/m × (clear width − 2 × 0.15 m); the stair is saturated from the start.
- **Saturated discharge:** flow out of the bottom of a queued stair.
- **Merging:** share of a stair's flow taken by people entering from the floor
  when both the floor and the stair above are queueing, for three settings of
  `movement.merge.floor_deference_ratio`.
- **Time step:** sensitivity of the result to the integration step.

| Case | Quantity | Reference | Meso | Deviation | Tolerance | |
|---|---|---:|---:|---:|---:|---|
| Corridor 30 m, 100 people, 1.0 m exit door | evacuation time (s) | 135.94 | 135.97 | 0.0% | 3% | ✅ |
| Corridor 30 m, 300 people, 0.9 m exit door | evacuation time (s) | 410.67 | 410.48 | 0.0% | 3% | ✅ |
| Corridor 30 m, 60 people, 1.6 m exit door | evacuation time (s) | 61.55 | 61.94 | 0.6% | 3% | ✅ |
| Single 1.2 m stair, 10 floors × 30 people | evacuation time (s) | 341.00 | 340.20 | 0.2% | 3% | ✅ |
| Single 1.2 m stair, 20 floors × 15 people | evacuation time (s) | 341.00 | 340.20 | 0.2% | 3% | ✅ |
| Single 1.2 m stair, 6 floors × 60 people | evacuation time (s) | 407.00 | 406.20 | 0.2% | 3% | ✅ |
| Saturated stair discharge | flow (persons/s) | 0.91 | 0.91 | 0.1% | 3% | ✅ |
| Stair-landing merge, deference ratio 0.3 | floor share of stair flow | 0.30 | 0.30 | 0.005 | ±0.04 | ✅ |
| Stair-landing merge, deference ratio 0.5 | floor share of stair flow | 0.50 | 0.50 | 0.000 | ±0.04 | ✅ |
| Stair-landing merge, deference ratio 0.7 | floor share of stair flow | 0.70 | 0.70 | 0.002 | ±0.04 | ✅ |
| Time step 1.0 s vs 0.25 s (8-floor stair) | evacuation time (s) | 230.12 | 229.78 | 0.1% | 1% | ✅ |

**What this does and does not show.** The engine reproduces the hydraulic
method's flow capacities, travel times and merge behaviour, and is insensitive
to the time step. It does *not* show that the hydraulic method, or the occupant
parameters in `params.yaml` (many still assumptions), are right for Hong Kong
towers. Comparison with drill data and with the microscopic engine (M8) is
still to come.

## 2. Behavioural and monotonicity checks (`tests/sim/`)

| Check | Result |
|---|---|
| Same inputs give identical outputs | ✅ |
| Blocking a staircase never speeds up evacuation, and nobody uses it | ✅ |
| Widening a staircase never slows evacuation | ✅ |
| A blockage discovered mid-evacuation still lets everyone out | ✅ |
| Phased release: no household leaves before its floor is released | ✅ |
| Stair assignment by floor is respected | ✅ |
| Evacuation lifts carry waiting households; without lifts they are rescued | ✅ |
| Rescue completion time matches the rescue model formula | ✅ |
| A refuge-floor rest adds its dwell time to the journey | ✅ |
| A counter-flow detour uses upward stair arcs and delays the household | ✅ |
| Hazard speed multipliers slow evacuation; FED ≥ 1 incapacitates | ✅ |

## 3. Parameter registry

| Check | Test | Status |
|---|---|---|
| Every value has a source; assumptions are flagged | `tests/test_params.py` | ✅ |
