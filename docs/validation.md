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

## 3. Monte Carlo convergence (M3)

CVaR₉₅ estimates and 95% bootstrap CIs from the first *n* scenarios of the
1,000-run demo (`make stress-demo`, seed 0, Latin Hypercube batches of 100),
in minutes:

| n | CVaR₉₅ total time | 95% CI | half-width | CVaR₉₅ self-evacuation | 95% CI | half-width |
|---:|---:|---|---:|---:|---|---:|
| 100 | 206.1 | [191.2, 216.9] | 12.8 | 113.6 | [102.4, 122.9] | 10.2 |
| 200 | 218.2 | [200.8, 234.0] | 16.6 | 113.6 | [105.7, 119.4] | 6.8 |
| 500 | 222.3 | [212.8, 230.6] | 8.9 | 112.3 | [108.0, 116.3] | 4.2 |
| 1000 | 219.4 | [211.9, 226.4] | 7.2 | 113.5 | [109.9, 117.0] | 3.5 |

Estimates settle within the earlier intervals and the intervals narrow as runs
grow. The total-time tail (driven by the rescue model) converges more slowly
than the self-evacuation tail. `tests/validation/test_convergence.py` checks the
narrowing on a smaller building; `MCConfig(target_halfwidth=...)` stops a run
once the CI is tight enough.

Also tested (`tests/scenarios/`, `tests/risk/`): results are identical for 1 or 2
worker processes; Latin Hypercube columns are stratified; an intervention spec
sees exactly the same households, pre-movement times, lift failures and rescue
start as the baseline on each scenario; CVaR matches its closed form for a
normal distribution; a paired bootstrap detects a uniform 2% improvement.

## 4. Hazard model and monotonicity (M4)

| Check | Test | Status |
|---|---|---|
| Zone integrator conserves tracer mass; a puff mixes to equal concentrations | `tests/hazard/test_hazard.py` | ✅ |
| Stack effect: upward exchange exceeds downward on every flight; the floor above the fire is worse than the floor below | `tests/hazard/test_hazard.py` | ✅ |
| Leaving the fire flat's door open makes the corridor untenable sooner | `tests/hazard/test_hazard.py` | ✅ |
| Holding stair doors open lets more smoke into the stair | `tests/hazard/test_hazard.py` | ✅ |
| FED: 1,000 ppm CO for 35 min gives FED 1 (ISO 13571 simplified form); ambient air gives no heat dose | `tests/hazard/test_hazard.py` | ✅ |
| Imported CFD fields give the expected ASET | `tests/hazard/test_hazard.py` | ✅ |
| **Monotonicity (paired, 40 scenarios):** blocking a stair never shortens evacuation (≥ 95% of scenarios, mean increase) | `tests/validation/test_monotonicity.py` | ✅ |
| Wider stair doors and exits never lengthen evacuation | `tests/validation/test_monotonicity.py` | ✅ |
| Smoke never speeds evacuation; P(RSET > ASET) rises when fire-flat doors are left open | `tests/validation/test_monotonicity.py` | ✅ |

Note on stair width: widening a *stair* in this model also lengthens the
walking line around each dog-leg turn (π·W/2), so for uncongested scenarios a
wider stair can be slightly slower. The monotonicity test therefore widens
doors and exits, which add capacity without changing path lengths.

## 5. Bottleneck attribution (M5)

| Check | Test | Status |
|---|---|---|
| Single-stair tower: max flow equals the stair capacity; the min cut is the bottom flight; every unit's route loads it | `tests/analysis/test_bottlenecks.py` | ✅ |
| With Stair A blocked, tail queues are on Stair B and the top candidates are "keep Stair A usable" or "widen Stair B" with negative ΔCVaR | `tests/analysis/test_bottlenecks.py` | ✅ |
| Adaptive re-running gives exactly the CVaR of re-running every scenario | `tests/analysis/test_bottlenecks.py` | ✅ |
| "Unblocking" keeps random-number slots aligned (common random numbers) | `tests/analysis/test_bottlenecks.py` | ✅ |

## 6. Optimiser (M6)

| Check | Test | Status |
|---|---|---|
| CMA-ES finds the minimum of a box-constrained quadratic, deterministically | `tests/optimize/test_optimize.py` | ✅ |
| Plans map exactly onto scenario fields and leave the baseline spec untouched | `tests/optimize/test_optimize.py` | ✅ |
| Wardens change only the covered floors' pre-movement; every other draw is identical (common random numbers) | `tests/optimize/test_optimize.py` | ✅ |
| An escorted household switches from waiting for rescue to being carried down | `tests/optimize/test_optimize.py` | ✅ |
| Comparing a result with itself gives ΔCVaR = 0 and "not significant" | `tests/optimize/test_optimize.py` | ✅ |
| The optimiser never returns a plan worse than the baseline in-sample, and confirms on fresh scenarios | `tests/optimize/test_optimize.py` | ✅ |

## 7. Web API and UI (M7)

| Check | Test | Status |
|---|---|---|
| A 3D replay re-simulates exactly the stress-test scenario it claims to (same total time) | `tests/test_api.py` | ✅ |
| The worst confirmation scenario replays exactly with and without the plan (side-by-side animation shows the same draws) | `tests/test_api.py` | ✅ |
| Results are strict JSON (no NaN / ∞), repeated requests come from the cache, progress events end in the final state | `tests/test_api.py` | ✅ |
| Bottleneck rows point at real building edges (for highlighting) | `tests/test_api.py` | ✅ |
| Chart helpers: histogram keeps every value on shared bins, quantiles match NumPy, ramps and floor labels | `web/test/lib.test.ts` | ✅ |

## 8. Parameter registry

| Check | Test | Status |
|---|---|---|
| Every value has a source; assumptions are flagged | `tests/test_params.py` | ✅ |
