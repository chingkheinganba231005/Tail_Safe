# Validation

Validation checks are executable. `tests/validation/` asserts them in CI, and
`tailsafe validate --markdown` regenerates the table below from the current
code and parameters.

## 1. Analytical checks of the mesoscopic engine

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
towers. Comparison with drill data is still to come; the comparison with the
person-by-person engine is in section 8.

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

## 3. Monte Carlo convergence

CVaR₉₅ estimates and 95% bootstrap CIs from the first *n* scenarios of the
1,000-run reference stress test (`make stress-example`, seed 0, Latin Hypercube batches of 100),
in minutes:

| n | CVaR₉₅ total time | 95% CI | half-width | CVaR₉₅ self-evacuation | 95% CI | half-width |
|---:|---:|---|---:|---:|---|---:|
| 100 | 206.1 | [191.1, 215.1] | 12.0 | 114.0 | [103.0, 125.8] | 11.4 |
| 200 | 218.2 | [200.3, 232.9] | 16.3 | 113.9 | [106.0, 119.9] | 7.0 |
| 500 | 222.0 | [212.8, 231.7] | 9.4 | 112.5 | [107.9, 116.2] | 4.1 |
| 1000 | 219.2 | [212.3, 226.2] | 7.0 | 113.7 | [110.2, 117.3] | 3.6 |

For the self-evacuation tail, every later estimate lies inside the earlier
intervals and the intervals narrow steadily as runs grow. The total-time tail,
driven by the rescue model, converges more slowly: at 100 runs its CVaR₉₅ is the
average of only five runs, and the bootstrap interval is too narrow (it misses
all the later estimates), so small studies can understate the tail. From 200
runs on, later estimates fall inside the earlier intervals. `tests/validation/test_convergence.py` checks the
narrowing on a smaller building; `MCConfig(target_halfwidth=...)` stops a run
once the CI is tight enough.

Also tested (`tests/scenarios/`, `tests/risk/`): results are identical for 1 or 2
worker processes; Latin Hypercube columns are stratified; an intervention spec
sees exactly the same households, pre-movement times, lift failures and rescue
start as the baseline on each scenario; CVaR matches its closed form for a
normal distribution; a paired bootstrap detects a uniform 2% improvement.

## 4. Hazard model and monotonicity

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

## 5. Bottleneck attribution

| Check | Test | Status |
|---|---|---|
| Single-stair tower: max flow equals the stair capacity; the min cut is the bottom flight; every unit's route loads it | `tests/analysis/test_bottlenecks.py` | ✅ |
| With Stair A blocked, tail queues are on Stair B and the top candidates are "keep Stair A usable" or "widen Stair B" with negative ΔCVaR | `tests/analysis/test_bottlenecks.py` | ✅ |
| Adaptive re-running gives exactly the CVaR of re-running every scenario | `tests/analysis/test_bottlenecks.py` | ✅ |
| "Unblocking" keeps random-number slots aligned (common random numbers) | `tests/analysis/test_bottlenecks.py` | ✅ |

## 6. Optimiser

| Check | Test | Status |
|---|---|---|
| CMA-ES finds the minimum of a box-constrained quadratic, deterministically | `tests/optimize/test_optimize.py` | ✅ |
| Plans map exactly onto scenario fields and leave the baseline spec untouched | `tests/optimize/test_optimize.py` | ✅ |
| Wardens change only the covered floors' pre-movement; every other draw is identical (common random numbers) | `tests/optimize/test_optimize.py` | ✅ |
| An escorted household switches from waiting for rescue to being carried down | `tests/optimize/test_optimize.py` | ✅ |
| Comparing a result with itself gives ΔCVaR = 0 and "not significant" | `tests/optimize/test_optimize.py` | ✅ |
| The optimiser never returns a plan worse than the baseline in-sample, and confirms on fresh scenarios | `tests/optimize/test_optimize.py` | ✅ |

## 7. Web API and UI

| Check | Test | Status |
|---|---|---|
| A 3D replay re-simulates exactly the stress-test scenario it claims to (same total time) | `tests/test_api.py` | ✅ |
| The worst confirmation scenario replays exactly with and without the plan (side-by-side animation shows the same draws) | `tests/test_api.py` | ✅ |
| Results are strict JSON (no NaN / ∞), repeated requests come from the cache, progress events end in the final state | `tests/test_api.py` | ✅ |
| Bottleneck rows point at real building edges (for highlighting) | `tests/test_api.py` | ✅ |
| Chart helpers: histogram keeps every value on shared bins, quantiles match NumPy, ramps and floor labels | `web/test/lib.test.ts` | ✅ |

## 8. Microscopic engine

| Check | Test | Status |
|---|---|---|
| Door openings and shared boundaries become portals on the source room, facing out | `tests/sim/test_micro.py` | ✅ |
| Buildings without room geometry are refused with a reason | `tests/sim/test_micro.py` | ✅ |
| One walker's exit time matches the walking distance at its speed | `tests/sim/test_micro.py` | ✅ |
| A 0.8 m exit door takes over 20% longer to clear than a 1.8 m one; flow 0.3–2 persons/s | `tests/sim/test_micro.py` | ✅ |
| Stair tower: everyone out, total time within 35% of the meso engine | `tests/sim/test_micro.py` | ✅ |
| 8-storey night scenario with a blocked stair: everyone out, deterministic, lift/rescue households keep meso times | `tests/sim/test_micro.py` | ✅ |
| Speed falls with density (fundamental diagram) | `tests/sim/test_micro.py` | ✅ |
| Meso–micro agreement on a 10-storey tower (last walker within 10%, correlation > 0.9) | `tests/sim/test_micro.py` (slow) | ✅ |

### Fundamental diagram

`tailsafe micro fd`: able adults (radius 0.2 m, free speed 1.40 m/s) in a
16 m × 2 m periodic corridor, 30 s warm-up then 30 s measured, against the
hydraulic model `S = k (1 − a D)`:

| Density (persons/m²) | Micro speed (m/s) | Hydraulic speed (m/s) | Micro flow (persons/m/s) | Hydraulic flow (persons/m/s) |
|---:|---:|---:|---:|---:|
| 0.25 | 1.40 | 1.40 | 0.35 | 0.35 |
| 0.50 | 1.05 | 1.40 | 0.53 | 0.70 |
| 1.00 | 0.60 | 1.03 | 0.60 | 1.03 |
| 1.50 | 0.60 | 0.84 | 0.90 | 1.26 |
| 2.00 | 0.35 | 0.66 | 0.70 | 1.31 |
| 2.50 | 0.24 | 0.47 | 0.59 | 1.17 |
| 3.00 | 0.22 | 0.28 | 0.67 | 0.85 |
| 3.50 | 0.17 | 0.10 | 0.60 | 0.34 |

Free walking matches. Between about 0.5 and 2.5 persons/m² the micro model
is slower than the hydraulic model and its flow peaks lower (≈ 0.9 against
1.3 persons/m/s); at 3.5 persons/m² it still creeps where the hydraulic
model is near standstill. The micro parameters are uncalibrated assumptions
(`movement.micro.*`), so this is a known difference, not a validation of
either model against measurements.

### Meso–micro agreement

`tailsafe micro compare` runs both engines on the same sampled scenarios of
the reference scenario (fire on 14/F, Stair A smoke-logged at 4 min, one lift
out, Sunday 3 a.m.) and compares the people the micro engine walks.
Correlation is across scenarios; where a metric hardly varies between
scenarios, differences between the engines dominate and the correlation is
low even when the bias is small.

**20-storey cruciform block** (seed 0):

| Metric | Meso mean | Micro mean | Bias micro − meso (95% CI) | Relative bias | Correlation | RMSE | Largest difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time for half of the walkers to get out | 14.1 min | 15.5 min | +1.4 min (+0.8 to +2.2) | +10% | 0.11 | 2.2 min | 6.7 min |
| Time for 95% of the walkers to get out | 29.5 min | 31.5 min | +1.9 min (+0.9 to +3.1) | +7% | 0.58 | 3.2 min | 7.5 min |
| Time the last walker gets out | 60.3 min | 60.4 min | +0.1 min (+0.0 to +0.2) | +0% | 1.00 | 0.2 min | 0.4 min |

**40-storey cruciform block** (the reference building, seed 0):

| Metric | Meso mean | Micro mean | Bias micro − meso (95% CI) | Relative bias | Correlation | RMSE | Largest difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| Time for half of the walkers to get out | 25.3 min | 29.1 min | +3.8 min (+2.4 to +5.8) | +15% | 0.14 | 5.4 min | 18.4 min |
| Time for 95% of the walkers to get out | 57.7 min | 61.5 min | +3.8 min (+0.6 to +7.3) | +7% | -0.11 | 8.4 min | 20.5 min |
| Time the last walker gets out | 91.2 min | 85.3 min | -5.9 min (-8.5 to -3.4) | -7% | 0.92 | 8.3 min | 21.4 min |

Reading: the engines agree closely on when the last walker gets out in the
20-storey block (driven by the slowest households' reaction and walking
times). The micro engine is a few minutes slower for the bulk of occupants —
merging at landings and doorways costs more when queues are physical — and
in the 40-storey block its stair lanes let the last walkers out somewhat
sooner than the meso engine's hydraulic stair capacity. Tail results
come from the meso engine; the micro engine is for replay and for
this cross-check.

## 9. Floor-plan reader

| Check | Test | Status |
|---|---|---|
| Otsu threshold and wall thickness from line crossings | `tests/vision/test_vision.py` | ✅ |
| Largest-rectangle cover of an L-shaped room | `tests/vision/test_vision.py` | ✅ |
| Typical cruciform floor: every door found, precision ≥ 0.9, both stairs, all room types | `tests/vision/test_vision.py` | ✅ |
| Scale from wall thickness within 5%; entrances at corridor ends found as exits | `tests/vision/test_vision.py` | ✅ |
| Noisy, blurred slab floor: both stairs, ≥ 90% of doors | `tests/vision/test_vision.py` | ✅ |
| Detected floor → validated 8-storey building that the meso engine evacuates and the micro engine can replay | `tests/vision/test_vision.py` | ✅ |
| Doorways drawn in the editor get their rooms | `tests/vision/test_vision.py` | ✅ |
| API and CLI round trips (sample → detect → edit → build) | `tests/test_api.py`, `tests/test_cli.py` | ✅ |

### Precision and recall on rendered plans

`tailsafe vision eval`: the ground floor and an upper floor of each template
(cruciform, slab, twin-core, care home), rendered at three qualities. A
detected doorway is correct if it lies within 0.5 m of a true door or open
passage with the same orientation; recall is over true doors. Stairs match
when their boxes overlap by at least 30% of the smaller one.

**These plans are synthetic** — rendered from the same geometry the reader
was designed around, with furniture and text-like clutter, noise and blur,
but no hatching, dimension lines, columns, curved walls or scanning
artefacts. They are not a hand-labelled set of real drawings; the numbers
are an upper bound for real plans, and the correction editor is part of the
intended workflow.

With a reference line (the editor's normal workflow):

| Plan quality | Plans | Door precision | Door recall | Stair precision | Stair recall | Largest scale error |
|---|---:|---:|---:|---:|---:|---:|
| clean, 20 px/m | 7 | 0.98 | 1.00 | 1.00 | 1.00 | 0% |
| noisy and blurred, 20 px/m | 7 | 0.93 | 0.96 | 1.00 | 1.00 | 0% |
| low resolution, 12 px/m | 7 | 0.84 | 0.99 | 0.93 | 1.00 | 0% |

Scale guessed from wall thickness (no reference line):

| Plan quality | Plans | Door precision | Door recall | Stair precision | Stair recall | Largest scale error |
|---|---:|---:|---:|---:|---:|---:|
| clean, 20 px/m | 7 | 0.98 | 1.00 | 1.00 | 1.00 | 0% |
| noisy and blurred, 20 px/m | 7 | 0.93 | 0.96 | 1.00 | 1.00 | 0% |
| low resolution, 12 px/m | 7 | 0.86 | 0.97 | 0.93 | 1.00 | 40% |

At 12 px/m a 0.2 m wall rasterises to three or four pixels, so the guessed
scale is 40% off; door and stair finding still works, but widths and areas
are wrong until a reference line is drawn.

## 10. Graph surrogate

| Check | Test | Status |
|---|---|---|
| Training cases are deterministic and stay within the typology ranges | `tests/surrogate/test_surrogate.py` | ✅ |
| Graph features: flats folded into their corridor, scenario flags set, building part unchanged | `tests/surrogate/test_surrogate.py` | ✅ |
| Quantiles never cross, CVaR₉₅ ≥ P95; weights reload to identical predictions | `tests/surrogate/test_surrogate.py` | ✅ |
| Cases outside the training draw are flagged (`coverage_notes`) | `tests/surrogate/test_surrogate.py` | ✅ |
| API returns an estimate (or 503 without the optional extra) | `tests/test_api.py` | ✅ |

### Accuracy and calibration

`tailsafe surrogate eval` on 800 cases (197 slab, 199 cruciform, 209 care
home, 195 twin core), each simulated with 64 Latin-Hypercube runs. Two test
sets: a random 20% of cases, and *leave one typology out* (train on three
building types, test on the fourth). Errors are against the held-out cases'
own 64-run estimates, which carry sampling error themselves. *Coverage* is the
share of simulated outcomes below the predicted quantile: a calibrated model
has coverage close to 0.50 / 0.75 / 0.90 / 0.95.

| Test set | Cases | Loss | P95 error (min) | P95 relative error | R² of P95 | CVaR₉₅ error (min) | Coverage P50 / P75 / P90 / P95 |
|---|---:|---|---:|---:|---:|---:|---|
| Random 20% | 160 | everyone out | 6.4 | 9% | 0.92 | 7.2 | 0.49 / 0.72 / 0.87 / 0.93 |
| Random 20% | 160 | last self-evacuee | 4.4 | 10% | 0.93 | 5.7 | 0.50 / 0.73 / 0.89 / 0.95 |
| Random 20% | 160 | 95% out | 11.0 | 19% | 0.65 | 11.8 | 0.49 / 0.71 / 0.87 / 0.91 |
| Unseen: care_home | 209 | everyone out | 21.1 | 33% | -0.29 | 21.1 | 0.06 / 0.17 / 0.31 / 0.42 |
| Unseen: care_home | 209 | last self-evacuee | 6.4 | 44% | 0.67 | 7.9 | 0.53 / 0.73 / 0.87 / 0.95 |
| Unseen: care_home | 209 | 95% out | 78.1 | 59% | -0.82 | 91.8 | 0.03 / 0.07 / 0.17 / 0.29 |
| Unseen: cruciform | 199 | everyone out | 12.6 | 16% | 0.81 | 13.6 | 0.58 / 0.77 / 0.90 / 0.95 |
| Unseen: cruciform | 199 | last self-evacuee | 6.5 | 12% | 0.82 | 7.5 | 0.31 / 0.61 / 0.84 / 0.93 |
| Unseen: cruciform | 199 | 95% out | 13.1 | 44% | -1.01 | 12.6 | 0.82 / 0.92 / 0.96 / 0.97 |
| Unseen: slab | 197 | everyone out | 14.9 | 23% | 0.57 | 12.0 | 0.77 / 0.88 / 0.97 / 0.98 |
| Unseen: slab | 197 | last self-evacuee | 6.6 | 12% | 0.81 | 6.8 | 0.39 / 0.79 / 0.83 / 0.90 |
| Unseen: slab | 197 | 95% out | 8.0 | 28% | -0.15 | 7.8 | 0.69 / 0.90 / 0.97 / 0.98 |
| Unseen: twin_core | 195 | everyone out | 5.5 | 9% | 0.89 | 7.2 | 0.44 / 0.71 / 0.88 / 0.93 |
| Unseen: twin_core | 195 | last self-evacuee | 5.7 | 12% | 0.84 | 7.2 | 0.53 / 0.78 / 0.91 / 0.95 |
| Unseen: twin_core | 195 | 95% out | 2.9 | 10% | 0.84 | 5.3 | 0.44 / 0.64 / 0.84 / 0.91 |

| Test set | Edge queueing rank correlation | Top-5 congested edges recovered |
|---|---:|---:|
| Random 20% | 0.82 | 82% |
| Unseen: care_home | 0.59 | 62% |
| Unseen: cruciform | 0.39 | 82% |
| Unseen: slab | 0.69 | 81% |
| Unseen: twin_core | 0.75 | 83% |

How to read this:

- **Buildings like the training set** (random split): P95 of "everyone out"
  and "last self-evacuee" within about 10% (R² 0.92–0.93), coverage within
  0.04 of nominal — the tails are slightly too narrow at P90 / P95.
- **"95% out" is harder** (19%, R² 0.65). In care homes with smoke more than
  5% of residents can be incapacitated, so the outcome jumps between about
  20 minutes and the 4-hour censoring horizon.
- **A building type it has never seen** degrades: a held-out twin core is
  predicted about as well as the random split (it resembles the others), a
  held-out cruciform or slab is 16–23% off for "everyone out", and held-out
  care homes are badly wrong (33–59%, coverage far from nominal) — their
  non-ambulant residents and staff assistance are unlike anything in the
  towers. The shipped model is trained on all four types; buildings and
  settings outside the training draw are flagged on the what-if screen,
  which always offers the full simulation.
- **Where queues form**: the rank correlation of predicted and simulated
  queueing per edge is 0.82 on the random split, and 82% of the five most
  congested edges are recovered.

### Speed

A prediction takes about 37 ms on one CPU core (including building the graph
features); a 1,000-run stress test of the same buildings takes about 31 s on
one core — about 835× faster, and still about 200× against a perfectly
parallel 4-core run. The design target was ≥ 100×.

The shipped weights (`tailsafe/surrogate/weights/`, 614 KiB) are trained on
all 800 cases (10% held back for validation; 120 epochs with a cosine
learning-rate schedule, keeping the weights with the lowest validation loss);
their metadata stores the numbers above, which the what-if screen quotes.

## 11. Briefing

| Check | Test | Status |
|---|---|---|
| The template briefing uses only numbers present in the facts (with and without bottlenecks and a plan) | `tests/report/test_briefing.py` | ✅ |
| The number check accepts signs and trailing zeros, rejects invented numbers | `tests/report/test_briefing.py` | ✅ |
| An LLM draft with a number not in the facts is rejected (template shown, with the number); API errors fall back | `tests/report/test_briefing.py` (model stubbed) | ✅ |
| The scenario sentence comes from the settings, not from stale free text | `tests/report/test_briefing.py` | ✅ |
| One-page PDF (and PNG preview) with the before/after distribution | `tests/report/test_briefing.py` | ✅ |
| CLI `tailsafe brief` and the API (`/api/briefing`, `/api/briefing/pdf`) | `tests/report/test_briefing.py`, `tests/test_api.py` | ✅ |

The number check reads digits only: a number written as a word ("three
floors") is not checked, which is why the prompt asks for digits. It checks
that each number *exists* in the facts, not that it is attached to the right
quantity; the template is safe by construction, an LLM draft is only as
faithful as its wording.


## 12. Parameter registry

| Check | Test | Status |
|---|---|---|
| Every value has a source; assumptions are flagged | `tests/test_params.py` | ✅ |
