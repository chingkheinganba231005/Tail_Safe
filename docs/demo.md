# Five-minute demo

The pitch scenario (spec §10): *Sunday, 3 a.m., 40-storey public housing
block, 22% of residents aged 65+, fire on 14/F, Stair A smoke-logged at
t = 4 min, one lift out of service.*

The demo never depends on the floor-plan reader: the building is the
procedural cruciform template.

## Before the talk (about 4 minutes on a 4-core laptop)

```bash
make install
.venv/bin/tailsafe demo --out out/pitch   # everything the talk shows, timed
make dev                                   # API + web UI on http://localhost:5173
```

`tailsafe demo` runs the whole story and prints how long each step took:

| Step | Output | Time (4 cores) |
|---|---|---:|
| 40-storey cruciform block | `building.json`, `building.png` | 1 s |
| Stress test, 600 scenarios | `stress/metrics.json`, `stress/distribution.png` | 38 s |
| Counterfactual bottlenecks (tail re-runs) | `stress/bottlenecks.json`, `.png` | 84 s |
| Optimised plan, confirmed on 200 fresh scenarios | `plan/optimization.json`, `plan/before_after.png` | 106 s |
| Worst scenario person by person, 14/F at 7 min | `replay.png` | 9 s |
| Briefing and pitch metrics | `briefing.md`, `briefing.pdf`, `pitch_metrics.md` | 1 s |
| **Total** | | **238 s** |

The headline numbers in [`pitch_metrics.md`](pitch_metrics.md) come from the
larger run (`--runs 1000 --rerun-fraction 0.2 --scenarios 100 --confirm 400`:
494 s, and it reproduces those numbers exactly). The smaller demo run tells
the same story (same plan, same trade-off) with wider confidence intervals.

In the browser, run the same scenario once before the talk (Scenario →
*Run stress test*, then Bottlenecks and Optimise). The API caches every job
on disk, so during the talk each screen opens instantly ("loaded from cache").

## The talk

| Time | Screen | Say |
|---|---|---|
| 0:00 | Header | The responsible-use notice: a decision-support prototype, synthetic residents, not a fire-engineering assessment. |
| 0:20 | 2 Scenario | The pitch scenario in one sentence; every parameter has a source or is flagged as an assumption. |
| 0:40 | 3 Stress results | Not one simulation but hundreds of plausible nights. The average hides the tail: CVaR₉₅ (the average of the worst 5%) against the mean; P(RSET > ASET) — how often someone is caught by smoke; the floors that fail. |
| 1:40 | 3 → *Who is in the tail* | Wheelchair users and frail older residents on the upper floors carry the tail. |
| 2:00 | 6 Bottlenecks | Why: losing Stair A drives the tail; click the row to see it in the plan and in 3D. |
| 2:40 | 7 Optimise | The fix is operational, not structural: lifts for residents who need them, wardens on two floors. Confirmed on fresh scenarios with paired confidence intervals — and the trade-off it makes worse, stated plainly. |
| 3:30 | 7 → replay | The worst confirmation scenario, before and after, on one clock. |
| 4:00 | 8 What-if (live) | Move the sliders: the surrogate answers in milliseconds; *Confirm with full simulation* when it matters. |
| 4:30 | 9 Briefing | One page for the building manager, every number checked against the results; download the PDF. |
| 5:00 | | End. |

If the network or the laptop fails, the files in `out/pitch/` (images,
briefing PDF, pitch metrics) tell the same story without the browser.
