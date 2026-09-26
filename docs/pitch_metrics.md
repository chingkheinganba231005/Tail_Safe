# Pitch metrics

*Generated 2026-09-26 10:26 UTC by `tailsafe pitch` (tailsafe 0.1.0) from saved results — do not edit by hand.*

> TailSafe is an educational and decision-support prototype. It is not a substitute for a registered fire engineer, the Fire Services Department, or compliance with the Buildings Department's Code of Practice for Fire Safety in Buildings. All occupants are synthetic; no personal data is used.
> Most parameters are still marked `ASSUMPTION — needs citation` in `config/params.yaml`; these numbers demonstrate the method.

## Scenario

- **Building:** Cruciform public housing block, 40 storeys
- **Scenario:** Sunday, 3 a.m., 40-storey public housing block, 22% of residents aged 65+, fire on 14/F, Stair A smoke-logged at t = 4 min, one lift out of service.
- **Monte Carlo:** 1000 sampled scenarios (seed 0), ≈1825 occupants on average

## Baseline: how bad is the tail?

| | Mean | P95 | CVaR₉₅ (worst 5%) |
|---|---:|---:|---:|
| Time until everyone is out (incl. fire-service rescue) | 137.9 min | 200.4 min | 219.2 min (95% CI 212.3 to 226.2) |
| Time until the last self-evacuee is out | 85.8 min | 104.3 min | 113.7 min (95% CI 110.2 to 117.3) |
| Time for 95% of occupants to get out | 60.1 min | 65.2 min | 66.5 min (95% CI 66.0 to 67.0) |

- **P(RSET > ASET):** 0.60 (95% CI 0.57–0.63) — the share of scenarios in which someone is still on a floor after its corridors become untenable, or breathes a dose above FED 0.3.
- **P(anyone incapacitated):** 0.065 (95% CI 0.051–0.082).
- **Floors most often failing:** 14/F (0.58), 15/F (0.47), 16/F (0.19).

## Who carries the tail

- Households whose most dependent member is a wheelchair user, on 30/F–39/F, are 1.2% of occupants but 100% of the people still inside late in the worst 5% of scenarios.
- Households whose most dependent member is a frail older adult (80+), on 30/F–39/F, are 4.9% of occupants but 70% of the people still inside late in the worst 5% of scenarios.

## Where it comes from (counterfactual bottlenecks)

Maximum egress flow 1.62 persons/s; minimum cut: Stair A, 1/F → G/F, Stair B, 1/F → G/F.

| Rank | Element | Δ CVaR₉₅ of p95 occupant time |
|---:|---|---:|
| 1 | Keep Stair A usable (no blockage) | -12.8 min (95% CI -13.4 to -12.2) |
| 2 | Stair B (all flights) | -8.6 min (95% CI -9.1 to -8.1) |
| 3 | Stair A doors on every floor | -0.3 min (95% CI -0.5 to -0.2) |
| 4 | Stair B doors on every floor | -0.3 min (95% CI -0.4 to -0.2) |
| 5 | Stair A (all flights) | -0.2 min (95% CI -0.4 to -0.1) |

**Losing Stair A drives the tail: keeping it usable cuts CVaR95 of p95 occupant time by 12.8 min (95% CI 12.2 to 13.4 min).**

## The fix: optimised operational plan

Objective: minimise CVaR95 of total time; 36 candidate plans evaluated with common random numbers.

- Use the non-firefighting lifts to evacuate mobility-impaired residents (nearest waiting floor first).
- Station floor wardens on 14/F, 35/F.

Confirmed on 400 fresh scenarios (seed 10000), paired:

| | Baseline CVaR₉₅ | With plan | Change (95% CI) |
|---|---:|---:|---:|
| Time until everyone is out (incl. fire-service rescue) | 204.9 min | 104.6 min | -100.3 min (-106.9 to -91.7) ✔ |
| Time until the last self-evacuee is out | 109.5 min | 104.6 min | -5.0 min (-8.7 to -0.7) ✔ |
| Time for 95% of occupants to get out | 66.3 min | 75.6 min | +9.3 min (+7.6 to +11.0) ▲ |
| P(RSET > ASET) | 0.58 | 0.55 | -0.03 (-0.05 to -0.01) ✔ |

✔ = significantly better, ▲ = significantly worse (the 95% confidence interval of the change excludes zero).

**Trade-off:** the plan makes time for 95% of occupants to get out significantly worse on the confirmation scenarios. Weigh this before adopting it, or re-run `tailsafe optimize --objective weighted` / `--objective p_rset`.

## Sources

- stress test: `out/demo1000/metrics.json`
- bottlenecks: `out/demo1000/bottlenecks.json`
- optimisation: `out/opt-demo/optimization.json`
