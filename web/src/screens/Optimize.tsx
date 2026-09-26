import { useEffect, useMemo, useState } from "react";
import type { StressRun } from "../App";
import { ChartCard } from "../components/ChartCard";
import { Callout, Icon } from "../components/Icon";
import { JobProgress } from "../components/JobProgress";
import { StackLegend, StackView3D } from "../components/StackView3D";
import { Histogram } from "../components/charts/Histogram";
import { useJob } from "../components/useJob";
import { LOSS_LABELS, LOSS_SHORT, clock, min, signedMin } from "../lib/format";
import { extent } from "../lib/stats";
import { useMode } from "../lib/theme";
import { LOSSES, type BuildingView, type Estimate, type Loss, type OptimizeResult, type Replay } from "../types";
import { replayBody } from "./Stack3D";

interface Props {
  building: BuildingView;
  stress: StressRun;
  result: OptimizeResult | null;
  setResult: (r: OptimizeResult) => void;
}

const LEVERS: { id: string; label: string }[] = [
  { id: "lifts", label: "Evacuation lifts" },
  { id: "hold_open", label: "Hold stair doors open" },
  { id: "stair_assignment", label: "Stair assignment by floor" },
  { id: "phasing", label: "Phased release" },
  { id: "wardens", label: "Floor wardens" },
];

/** Significance of a paired change where lower is better: icon + label, never colour alone. */
function Verdict({ d }: { d: Estimate }) {
  if (d.hi < 0)
    return (
      <span style={{ color: "var(--success-text)" }}>
        <Icon name="check" size={13} className="mr-1 inline -translate-y-px" />
        better
      </span>
    );
  if (d.lo > 0)
    return (
      <span style={{ color: "var(--critical)" }}>
        <Icon name="alert" size={13} className="mr-1 inline -translate-y-px" />
        worse
      </span>
    );
  return (
    <span className="muted">
      <span aria-hidden>– </span>no clear change
    </span>
  );
}

/** Screen 7: search for the operational plan that shrinks the tail, then compare. */
export function Optimize({ building, stress, result, setResult }: Props) {
  const [kind, setKind] = useState<"cvar" | "p_rset" | "weighted">("cvar");
  const [loss, setLoss] = useState<Loss>("total_time");
  const [levers, setLevers] = useState<string[]>(LEVERS.map((l) => l.id));
  const [maxWardens, setMaxWardens] = useState(2);
  const [nScen, setNScen] = useState(60);
  const [nConfirm, setNConfirm] = useState(200);
  const { job, error, running, run } = useJob<OptimizeResult>();

  const start = async () => {
    const out = await run("/api/optimize", {
      building_id: building.id,
      spec: stress.spec,
      objective: { kind, loss },
      n_scenarios: nScen,
      confirm_scenarios: nConfirm,
      levers,
      max_wardens: maxWardens,
      cmaes_iterations: 3,
      seed: 1,
    });
    if (out) setResult(out.result);
  };

  return (
    <div className="space-y-4">
      <section className="card space-y-3 p-4">
        <div>
          <h2 className="page-title">Optimise</h2>
          <p className="secondary max-w-3xl text-sm">
            Candidate plans are simulated on the same sampled scenarios (common random numbers), combined
            greedily, then the best plan is confirmed against the baseline on fresh scenarios. Only the
            confirmation is reported as significant or not.
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-sm">
            <span className="secondary block text-xs">Minimise</span>
            <select value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
              <option value="cvar">CVaR₉₅ (tail)</option>
              <option value="p_rset">P(RSET &gt; ASET)</option>
              <option value="weighted">Mean + CVaR₉₅</option>
            </select>
          </label>
          {kind !== "p_rset" && (
            <label className="text-sm">
              <span className="secondary block text-xs">Of</span>
              <select value={loss} onChange={(e) => setLoss(e.target.value as Loss)}>
                {LOSSES.map((l) => (
                  <option key={l} value={l}>
                    {LOSS_SHORT[l]}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label className="text-sm">
            <span className="secondary block text-xs">Max wardens</span>
            <input type="number" min={0} max={10} className="w-16" value={maxWardens} onChange={(e) => setMaxWardens(Number(e.target.value))} />
          </label>
          <label className="text-sm">
            <span className="secondary block text-xs">Scenarios per plan</span>
            <select value={nScen} onChange={(e) => setNScen(Number(e.target.value))}>
              {[30, 60, 100, 200].map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            <span className="secondary block text-xs">Confirmation scenarios</span>
            <select value={nConfirm} onChange={(e) => setNConfirm(Number(e.target.value))}>
              {[100, 200, 400].map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
        </div>
        <fieldset className="flex flex-wrap gap-3 text-sm">
          <legend className="secondary mb-1 text-xs">Levers the plan may use</legend>
          {LEVERS.map((l) => (
            <label key={l.id} className="flex items-center gap-1.5">
              <input
                type="checkbox"
                checked={levers.includes(l.id)}
                onChange={(e) => setLevers(e.target.checked ? [...levers, l.id] : levers.filter((x) => x !== l.id))}
              />
              {l.label}
            </label>
          ))}
        </fieldset>
        <button className="btn" disabled={running || levers.length === 0} onClick={start}>
          {running ? "Searching…" : "Find a better plan"}
        </button>
      </section>
      <JobProgress job={running || error ? job : null} error={error} label="Optimisation" />
      {result && <OptimizeView building={building} result={result} />}
    </div>
  );
}

function OptimizeView({ building, result }: { building: BuildingView; result: OptimizeResult }) {
  const c = result.confirmation;
  const [loss, setLoss] = useState<Loss>(result.objective.loss);
  const worse = [
    ...LOSSES.filter((l) => c.losses[l].delta_cvar.lo > 0).map((l) => LOSS_SHORT[l].toLowerCase()),
    ...(c.p_rset_exceeds_aset.delta.lo > 0 ? ["P(RSET > ASET)"] : []),
  ];
  const ba = result.before_after[loss];
  const before = useMemo(() => ba.before.map((v) => v / 60), [ba]);
  const after = useMemo(() => ba.after.map((v) => v / 60), [ba]);
  const domain = extent([...before, ...after]);

  return (
    <>
      <section className="card p-4">
        <h3 className="font-semibold">Recommended plan</h3>
        <p className="secondary mb-2 text-sm">
          Objective: minimise {result.objective.label} · {result.evaluations} plans evaluated
        </p>
        <ul className="list-disc space-y-1 pl-5 text-sm">
          {result.plan_description.map((d) => (
            <li key={d}>{d}</li>
          ))}
        </ul>
      </section>
      <section className="card p-4">
        <h3 className="font-semibold">Confirmed on {c.scenarios} fresh scenarios</h3>
        <p className="secondary mb-2 text-sm">Paired comparison on identical scenarios; changes with 95% confidence intervals.</p>
        <div className="overflow-x-auto">
          <table className="tabular w-full text-sm">
            <thead>
              <tr className="secondary text-left">
                <th className="py-1 font-medium">CVaR₉₅ of</th>
                <th className="py-1 text-right font-medium">Baseline</th>
                <th className="py-1 text-right font-medium">With plan</th>
                <th className="py-1 text-right font-medium">Change</th>
                <th className="py-1 text-right font-medium">95% CI</th>
                <th className="py-1 pl-3 font-medium">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {LOSSES.map((l) => {
                const x = c.losses[l];
                return (
                  <tr key={l} style={{ borderTop: "1px solid var(--grid)" }}>
                    <td className="py-1">{LOSS_LABELS[l]}</td>
                    <td className="py-1 text-right">{min(x.before_cvar)} min</td>
                    <td className="py-1 text-right">{min(x.after_cvar)} min</td>
                    <td className="py-1 text-right">{signedMin(x.delta_cvar.value)} min</td>
                    <td className="py-1 text-right">
                      {signedMin(x.delta_cvar.lo)} to {signedMin(x.delta_cvar.hi)}
                    </td>
                    <td className="py-1 pl-3">
                      <Verdict d={x.delta_cvar} />
                    </td>
                  </tr>
                );
              })}
              <tr style={{ borderTop: "1px solid var(--grid)" }}>
                <td className="py-1">P(RSET &gt; ASET)</td>
                <td className="py-1 text-right">{c.p_rset_exceeds_aset.before.toFixed(2)}</td>
                <td className="py-1 text-right">{c.p_rset_exceeds_aset.after.toFixed(2)}</td>
                <td className="py-1 text-right">{c.p_rset_exceeds_aset.delta.value.toFixed(3)}</td>
                <td className="py-1 text-right">
                  {c.p_rset_exceeds_aset.delta.lo.toFixed(3)} to {c.p_rset_exceeds_aset.delta.hi.toFixed(3)}
                </td>
                <td className="py-1 pl-3">
                  <Verdict d={c.p_rset_exceeds_aset.delta} />
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        {worse.length > 0 && (
          <Callout tone="warning" role="note" className="mt-3">
            <strong className="font-medium">Trade-off:</strong> this plan makes {worse.join(" and ")} significantly
            worse. Try another objective or restrict the levers before adopting it.
          </Callout>
        )}
      </section>
      <section className="card p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <h3 className="font-semibold">Before / after</h3>
          <div role="group" aria-label="Outcome" className="flex flex-wrap gap-1">
            {LOSSES.map((l) => (
              <button key={l} className="btn-ghost text-sm" aria-pressed={loss === l} onClick={() => setLoss(l)}>
                {LOSS_SHORT[l]}
              </button>
            ))}
          </div>
        </div>
        <div className="grid gap-4 lg:grid-cols-2">
          {(
            [
              ["Baseline", before, c.losses[loss].before_cvar],
              ["With plan", after, c.losses[loss].after_cvar],
            ] as const
          ).map(([name, values, cvar]) => (
            <ChartCard key={name} title={name} subtitle={`${LOSS_LABELS[loss]} — ${c.scenarios} identical scenarios`}>
              <Histogram
                series={[{ name, values: [...values], color: "var(--series-1)" }]}
                xLabel="minutes"
                domain={domain}
                height={220}
                markers={[{ x: cvar / 60, label: `CVaR₉₅ ${min(cvar)}`, strong: true }]}
              />
            </ChartCard>
          ))}
        </div>
        <SplitReplay building={building} result={result} />
      </section>
    </>
  );
}

/** The worst confirmation scenario replayed without and with the plan, on one clock. */
function SplitReplay({ building, result }: { building: BuildingView; result: OptimizeResult }) {
  const mode = useMode();
  const base = useJob<Replay>();
  const plan = useJob<Replay>();
  const [pair, setPair] = useState<[Replay, Replay] | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const rp = result.replay;

  const load = async () => {
    const a = await base.run("/api/replay", replayBody(building.id, rp.baseline_spec, rp.worst_index, rp.seed, rp.runs, rp.batch_size));
    if (!a) return;
    const b = await plan.run("/api/replay", replayBody(building.id, rp.plan_spec, rp.worst_index, rp.seed, rp.runs, rp.batch_size));
    if (b) {
      setPair([a.result, b.result]);
      setT(0);
    }
  };

  const tEnd = pair ? Math.max(pair[0].times.at(-1) ?? 0, pair[1].times.at(-1) ?? 0) : 0;
  useEffect(() => {
    if (!playing || !pair) return;
    const id = window.setInterval(() => {
      setT((x) => {
        const next = x + tEnd / 300;
        if (next >= tEnd) {
          setPlaying(false);
          return tEnd;
        }
        return next;
      });
    }, 100);
    return () => window.clearInterval(id);
  }, [playing, pair, tEnd]);

  const frameAt = (r: Replay) => {
    let k = 0;
    while (k + 1 < r.times.length && r.times[k + 1] <= t) k++;
    return k;
  };
  const qMax = pair
    ? Math.max(5, ...pair.flatMap((r) => Object.values(r.stair_queues).flatMap((g) => g.flatMap((x) => x))))
    : 5;

  return (
    <div className="mt-4 space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="font-semibold">The worst confirmation scenario, side by side</h4>
          <p className="secondary text-sm">Same fire, same residents, same random draws — only the plan differs.</p>
        </div>
        {!pair && (
          <button className="btn-ghost text-sm" disabled={base.running || plan.running} onClick={load}>
            {base.running || plan.running ? "Simulating…" : "Load animation"}
          </button>
        )}
      </div>
      {(base.error || plan.error) && <JobProgress job={null} error={base.error ?? plan.error} label="Replay" />}
      {pair && (
        <>
          <div className="grid gap-4 lg:grid-cols-2">
            {(["Baseline", "With plan"] as const).map((name, k) => {
              const r = pair[k];
              const f = frameAt(r);
              const inside = r.remaining[f].reduce((a, b) => a + b, 0);
              return (
                <div key={name}>
                  <div className="flex items-baseline justify-between text-sm">
                    <strong>{name}</strong>
                    <span className="secondary tabular">
                      {inside} still inside · everyone out at {clock(r.summary.total_time_s)}
                    </span>
                  </div>
                  <StackView3D building={building.building} replay={r} frame={f} queueMax={qMax} height={420} />
                </div>
              );
            })}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              className="btn text-sm"
              onClick={() => {
                if (t >= tEnd) setT(0);
                setPlaying(!playing);
              }}
            >
              {playing ? "Pause" : "Play"}
            </button>
            <input
              type="range"
              min={0}
              max={tEnd}
              step={Math.max(1, Math.round(tEnd / 500))}
              value={t}
              onChange={(e) => {
                setPlaying(false);
                setT(Number(e.target.value));
              }}
              className="min-w-40 flex-1"
              aria-label="Time"
            />
            <span className="tabular w-24 text-right text-sm">t = {clock(t)}</span>
          </div>
          <StackLegend mode={mode} queueMax={qMax} />
        </>
      )}
    </div>
  );
}
