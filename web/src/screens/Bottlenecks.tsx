import { useMemo, useState } from "react";
import type { StressRun } from "../App";
import { ChartCard } from "../components/ChartCard";
import { RankChart } from "../components/charts/RankChart";
import { JobProgress } from "../components/JobProgress";
import { PlanView } from "../components/PlanView";
import { StackView3D } from "../components/StackView3D";
import { useJob } from "../components/useJob";
import { LOSS_SHORT, levelLabel, pct, signedMin } from "../lib/format";
import { LOSSES, type BottleneckResult, type BuildingView, type Loss } from "../types";

interface Props {
  building: BuildingView;
  stress: StressRun;
  result: BottleneckResult | null;
  setResult: (r: BottleneckResult) => void;
}

/** Screen 6: which element, relaxed, shrinks the tail most — and where it is. */
export function Bottlenecks({ building, stress, result, setResult }: Props) {
  const b = building.building;
  const [loss, setLoss] = useState<Loss>("p95_occupant_time");
  const [factor, setFactor] = useState(1.5);
  const [selected, setSelected] = useState<string | null>(null);
  const [level, setLevel] = useState<number | null>(null);
  const { job, error, running, run } = useJob<BottleneckResult>();
  const nodeLevel = useMemo(() => new Map(b.nodes.map((n) => [n.id, n.level])), [b]);
  const edgeLevel = (id: string) => {
    const e = b.edges.find((x) => x.id === id);
    return e ? Math.max(nodeLevel.get(e.source) ?? 0, nodeLevel.get(e.target) ?? 0) : null;
  };

  const start = async () => {
    const out = await run("/api/bottlenecks", { stress_job_id: stress.jobId, loss, factor });
    if (out) {
      setResult(out.result);
      setSelected(out.result.ranking[0]?.key ?? null);
    }
  };

  const row = result?.ranking.find((r) => r.key === selected) ?? null;
  const queue = result?.queues.find((q) => `queue:${q.arc}` === selected) ?? null;
  const edges = new Set(row ? row.edges : queue ? [queue.edge] : []);
  const stair = row?.kind === "unblock" || row?.kind === "stair" || row?.kind === "stair_doors" ? row.stair : null;
  const fallbackLevel = (() => {
    const first = [...edges][0];
    if (first) return edgeLevel(first) ?? 1;
    if (stair && stress.result.stair_congestion[stair]) {
      const rows = stress.result.stair_congestion[stair];
      return rows.reduce((a, x) => (x.person_seconds > a.person_seconds ? x : a), rows[0]).level;
    }
    return 1;
  })();
  const shownLevel = level ?? fallbackLevel;

  return (
    <div className="space-y-4">
      <section className="card flex flex-wrap items-end justify-between gap-3 p-4">
        <div>
          <h2 className="page-title">Bottlenecks</h2>
          <p className="secondary max-w-2xl text-sm">
            Each candidate element (a whole stair, its doors, an exit, the places where queues form) is given more
            capacity — or a blocked stair is kept usable — and the tail scenarios are re-simulated with the same
            random draws. The ranking is by how much CVaR₉₅ falls.
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-2">
          <label className="text-sm">
            <span className="secondary block text-xs">Outcome</span>
            <select value={loss} onChange={(e) => setLoss(e.target.value as Loss)}>
              {LOSSES.map((l) => (
                <option key={l} value={l}>
                  {LOSS_SHORT[l]}
                </option>
              ))}
            </select>
          </label>
          <label className="text-sm">
            <span className="secondary block text-xs">Extra capacity</span>
            <select value={factor} onChange={(e) => setFactor(Number(e.target.value))}>
              {[1.25, 1.5, 2].map((f) => (
                <option key={f} value={f}>
                  +{Math.round((f - 1) * 100)}%
                </option>
              ))}
            </select>
          </label>
          <button className="btn" disabled={running} onClick={start}>
            {running ? "Ranking…" : "Rank bottlenecks"}
          </button>
        </div>
      </section>
      <JobProgress job={running || error ? job : null} error={error} label="Bottleneck ranking" />
      {result && (
        <>
          {result.headline && (
            <section className="card p-4">
              <p className="text-base font-semibold">{result.headline}</p>
              <p className="secondary text-sm">
                Maximum egress flow {result.max_flow_persons_per_s.toFixed(2)} persons/s · minimum cut:{" "}
                {result.min_cut.join("; ")}
              </p>
            </section>
          )}
          <div className="grid gap-4 xl:grid-cols-2">
            <ChartCard
              title={`Change in CVaR₉₅ of ${LOSS_SHORT[result.loss].toLowerCase()}`}
              subtitle="Minutes, with 95% CI. Click a row to see the element in plan and in 3D."
              table={{
                columns: ["Rank", "Element", "Δ CVaR₉₅ (min)", "95% CI", "Queue recurrence"],
                rows: result.ranking.map((r) => [
                  r.rank,
                  r.label,
                  r.delta_cvar ? signedMin(r.delta_cvar.value) : "–",
                  r.delta_cvar ? `${signedMin(r.delta_cvar.lo)} to ${signedMin(r.delta_cvar.hi)}` : "–",
                  pct(r.recurrence),
                ]),
              }}
            >
              <RankChart
                rows={result.ranking
                  .filter((r) => r.delta_cvar)
                  .map((r) => ({
                    key: r.key,
                    label: `${r.rank}. ${r.label}`,
                    value: r.delta_cvar!.value / 60,
                    lo: r.delta_cvar!.lo / 60,
                    hi: r.delta_cvar!.hi / 60,
                    note: r.in_min_cut ? "In the minimum cut of the egress network" : undefined,
                  }))}
                format={(v) => signedMin(v * 60)}
                valueLabel="Δ CVaR₉₅ (min); negative = better"
                selected={selected}
                onSelect={(k) => {
                  setSelected(k);
                  setLevel(null);
                }}
                labelWidth={260}
              />
            </ChartCard>
            <section className="card p-4">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-semibold">{row?.label ?? queue?.where ?? "Select an element"}</h3>
                <label className="text-sm">
                  <span className="secondary mr-2">Floor</span>
                  <select value={shownLevel} onChange={(e) => setLevel(Number(e.target.value))}>
                    {b.levels.map((l) => (
                      <option key={l.index} value={l.index}>
                        {levelLabel(l.index)}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <PlanView building={b} level={shownLevel} highlightEdges={edges} highlightStair={stair} height={300} />
              <div className="mt-3">
                <StackView3D building={b} highlightEdges={edges} highlightStair={stair} height={320} />
              </div>
              <p className="muted mt-1 text-xs">Highlighted: heavy outline in the plan; outlined stair column or round markers in 3D.</p>
            </section>
          </div>
          <ChartCard
            title="Where queues form in the worst 5%"
            subtitle="Recurrence: share of tail scenarios in which the queue reaches 10 people. Click a row to locate it."
          >
            <div className="max-h-80 overflow-auto">
              <table className="tabular w-full text-sm">
                <thead>
                  <tr className="secondary text-left">
                    <th className="py-1 font-medium">Where</th>
                    <th className="py-1 text-right font-medium">Recurrence</th>
                    <th className="py-1 text-right font-medium">Person-min (tail)</th>
                    <th className="py-1 text-right font-medium">Peak queue</th>
                  </tr>
                </thead>
                <tbody>
                  {result.queues.map((q) => (
                    <tr
                      key={q.arc}
                      onClick={() => {
                        setSelected(`queue:${q.arc}`);
                        setLevel(null);
                      }}
                      style={{
                        borderTop: "1px solid var(--grid)",
                        cursor: "pointer",
                        background:
                          selected === `queue:${q.arc}` ? "color-mix(in srgb, var(--series-1) 10%, transparent)" : undefined,
                      }}
                    >
                      <td className="py-1">{q.where}</td>
                      <td className="py-1 text-right">{pct(q.recurrence)}</td>
                      <td className="py-1 text-right">{(q.tail_person_seconds / 60).toFixed(0)}</td>
                      <td className="py-1 text-right">{q.tail_max_queue.toFixed(0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>
        </>
      )}
    </div>
  );
}
