import { useEffect, useMemo, useRef, useState } from "react";
import type { StressRun } from "../App";
import { get, post } from "../api";
import { ChartCard } from "../components/ChartCard";
import { Callout } from "../components/Icon";
import { JobProgress } from "../components/JobProgress";
import { RangeChart, type RangeRow } from "../components/charts/RangeChart";
import { RankChart } from "../components/charts/RankChart";
import { useJob } from "../components/useJob";
import { LOSS_SHORT, levelLabel } from "../lib/format";
import { quantile } from "../lib/stats";
import { LOSSES, type BuildingView, type ScenarioSpec, type StressResult, type SurrogatePrediction, type TimeSlot } from "../types";

interface Props {
  building: BuildingView;
  spec: ScenarioSpec | null;
  onStress: (run: StressRun) => void;
}

const SLOTS: { id: TimeSlot; label: string }[] = [
  { id: "weekday_day", label: "Weekday day" },
  { id: "weekday_night", label: "Weekday night" },
  { id: "weekend_day", label: "Weekend day" },
  { id: "weekend_night", label: "Weekend night" },
];

/** Screen 8: move the sliders, get estimates in milliseconds, confirm with the simulator. */
export function WhatIf({ building, spec: initial, onStress }: Props) {
  const b = building.building;
  const levels = b.levels.map((l) => l.index).filter((l) => l > 0);
  const [spec, setSpec] = useState<ScenarioSpec | null>(initial);
  const [pred, setPred] = useState<SurrogatePrediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmed, setConfirmed] = useState<{ spec: string; result: StressResult; jobId: string } | null>(null);
  const job = useJob<StressResult>();
  const seq = useRef(0);

  useEffect(() => {
    if (initial) return;
    get<ScenarioSpec>("/api/specs/reference").then((s) =>
      setSpec({ ...s, fire_level: s.fire_level != null ? Math.min(s.fire_level, Math.max(...levels)) : null }),
    );
  }, []);

  useEffect(() => {
    if (!spec) return;
    const my = ++seq.current;
    const id = window.setTimeout(() => {
      post<SurrogatePrediction>("/api/surrogate/predict", { building_id: building.id, spec })
        .then((p) => {
          if (my === seq.current) {
            setPred(p);
            setError(null);
          }
        })
        .catch((e: Error) => my === seq.current && setError(e.message));
    }, 120);
    return () => window.clearTimeout(id);
  }, [spec, building.id]);

  const specKey = useMemo(() => JSON.stringify(spec), [spec]);
  if (!spec) return <p className="muted">Loading…</p>;
  const set = (patch: Partial<ScenarioSpec>) => setSpec({ ...spec, ...patch });
  const blocked = spec.stair_blockages[0];
  const nLifts = b.lifts.length;

  const confirm = async () => {
    const out = await job.run("/api/stress", { building_id: building.id, spec, runs: 300, seed: 0 });
    if (out) setConfirmed({ spec: specKey, result: out.result, jobId: out.job.id });
  };
  const fresh = confirmed && confirmed.spec === specKey ? confirmed : null;

  const rows: RangeRow[] = pred
    ? LOSSES.map((l) => {
        const q = pred.losses[l];
        const series = [
          { name: "Surrogate (instant)", color: "var(--series-1)", p50: q.p50 / 60, p75: q.p75 / 60, p90: q.p90 / 60, p95: q.p95 / 60, cvar: q.cvar95 / 60 },
        ];
        if (fresh) {
          const xs = fresh.result.losses[l].map((v) => v / 60);
          series.push({
            name: "Simulator (300 runs)",
            color: "var(--series-2)",
            p50: quantile(xs, 0.5),
            p75: quantile(xs, 0.75),
            p90: quantile(xs, 0.9),
            p95: quantile(xs, 0.95),
            cvar: fresh.result.risk[l].cvar.value / 60,
          });
        }
        return { key: l, label: LOSS_SHORT[l], series };
      })
    : [];
  const hold = pred?.model.evaluation?.holdout_p95_relative_error;
  const atHorizon = pred ? LOSSES.some((l) => pred.losses[l].cvar95 >= pred.horizon_s - 1) : false;

  return (
    <div className="space-y-4">
      <section className="card p-4">
        <h2 className="page-title">What-if (live)</h2>
        <p className="secondary max-w-3xl text-sm">
          A graph neural network trained on simulator runs estimates the evacuation-time distribution in milliseconds.
          Use it to explore; confirm anything you rely on with the full simulation.
        </p>
      </section>
      <div className="grid gap-4 lg:grid-cols-[300px_1fr]">
        <section className="card space-y-3 self-start p-4 text-sm">
          <div role="group" aria-label="Time of day" className="flex flex-wrap gap-1">
            {SLOTS.map((s) => (
              <button key={s.id} className="btn-ghost text-xs" aria-pressed={spec.time_slot === s.id} onClick={() => set({ time_slot: s.id })}>
                {s.label}
              </button>
            ))}
          </div>
          <label className="block">
            <span className="secondary flex justify-between text-xs">
              Residents aged 65+ <span className="tabular">{Math.round(100 * (spec.share_65_plus ?? 0.24))}%</span>
            </span>
            <input
              type="range"
              min={0.05}
              max={0.45}
              step={0.01}
              className="w-full"
              value={spec.share_65_plus ?? 0.24}
              onChange={(e) => set({ share_65_plus: Number(e.target.value) })}
            />
          </label>
          <label className="block">
            <span className="secondary block text-xs">Fire floor</span>
            <select value={spec.fire_level ?? ""} onChange={(e) => set({ fire_level: e.target.value === "" ? null : Number(e.target.value) })}>
              <option value="">Random floor</option>
              {levels.map((l) => (
                <option key={l} value={l}>
                  {levelLabel(l)}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={spec.hazard?.enabled ?? false}
              onChange={(e) => set({ hazard: e.target.checked ? { enabled: true } : null })}
            />
            Smoke spread and tenability
          </label>
          <label className="block">
            <span className="secondary block text-xs">Staircase smoke-logged</span>
            <select
              value={blocked?.stair ?? ""}
              onChange={(e) =>
                set({
                  stair_blockages: e.target.value
                    ? [{ stair: e.target.value, time: { dist: "constant", value: blocked?.time.value ?? 240 } }]
                    : [],
                })
              }
            >
              <option value="">None</option>
              {b.stairs.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          {blocked && (
            <label className="block">
              <span className="secondary flex justify-between text-xs">
                …from <span className="tabular">{((blocked.time.value ?? 0) / 60).toFixed(0)} min</span>
              </span>
              <input
                type="range"
                min={60}
                max={900}
                step={30}
                className="w-full"
                value={blocked.time.value ?? 240}
                onChange={(e) =>
                  set({ stair_blockages: [{ stair: blocked.stair, time: { dist: "constant", value: Number(e.target.value) } }] })
                }
              />
            </label>
          )}
          <label className="block">
            <span className="secondary block text-xs">Lifts out of service</span>
            <input
              type="number"
              min={0}
              max={nLifts}
              className="w-20"
              value={spec.lifts_out_of_service}
              onChange={(e) => set({ lifts_out_of_service: Number(e.target.value) })}
            />
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={spec.evacuation_lifts} onChange={(e) => set({ evacuation_lifts: e.target.checked })} />
            Evacuation lifts for mobility-impaired residents
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={spec.warden_levels.length > 0}
              disabled={spec.fire_level == null}
              onChange={(e) =>
                set({
                  warden_levels:
                    e.target.checked && spec.fire_level != null
                      ? [spec.fire_level, Math.min(spec.fire_level + 1, Math.max(...levels))]
                      : [],
                })
              }
            />
            Wardens on the fire floor and the floor above
          </label>
        </section>
        <div className="space-y-4">
          {pred && pred.coverage_notes.length > 0 && (
            <Callout tone="warning">
              <p className="font-medium">
                Outside what the surrogate was trained on — treat the estimate as a rough guide and confirm it
              </p>
              <ul className="secondary mt-1 list-disc pl-5">
                {pred.coverage_notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            </Callout>
          )}
          <ChartCard
            title="Evacuation time, instantly"
            subtitle={
              pred
                ? `Estimated in ${pred.elapsed_ms.toFixed(0)} ms. Bar: median to P95; ticks: P75, P90; diamond: CVaR₉₅ (average of the worst 5%).` +
                  (atHorizon ? ` Values at ${(pred.horizon_s / 60).toFixed(0)} min mean "not within the simulated ${(pred.horizon_s / 3600).toFixed(0)} hours".` : "")
                : "…"
            }
            table={
              pred
                ? {
                    columns: ["Outcome", "Source", "Median", "P90", "P95", "CVaR₉₅"],
                    rows: rows.flatMap((r) =>
                      r.series.map((s) => [r.label, s.name, s.p50.toFixed(1), s.p90.toFixed(1), s.p95.toFixed(1), s.cvar.toFixed(1)]),
                    ),
                  }
                : undefined
            }
          >
            {error ? (
              <Callout tone="critical">{error}</Callout>
            ) : pred ? (
              <RangeChart rows={rows} unit="minutes" />
            ) : (
              <p className="muted text-sm">Estimating…</p>
            )}
          </ChartCard>
          <section className="card flex flex-wrap items-center gap-3 p-4 text-sm">
            <button className="btn" disabled={job.running} onClick={confirm}>
              {job.running ? "Simulating…" : "Confirm with full simulation"}
            </button>
            {fresh && (
              <button
                className="btn-ghost"
                onClick={() => onStress({ jobId: fresh.jobId, result: fresh.result, spec, runs: 300, seed: 0 })}
              >
                Open the full results →
              </button>
            )}
            <span className="muted text-xs">
              {confirmed && !fresh ? "Settings changed since the last simulation. " : ""}
              300 Monte Carlo runs with the same settings; the surrogate's estimate stays on the chart for comparison.
            </span>
          </section>
          {job.running && <JobProgress job={job.job} error={null} label="Stress test" />}
          {job.error && <JobProgress job={null} error={job.error} label="Stress test" />}
          {pred && pred.edges.length > 0 && (
            <ChartCard
              title="Where queues are expected"
              subtitle="Estimated average person-minutes queueing, most congested first."
              table={{ columns: ["Where", "Person-minutes"], rows: pred.edges.map((e) => [e.label, e.person_minutes.toFixed(1)]) }}
            >
              <RankChart
                rows={pred.edges.map((e) => ({ key: e.edge, label: e.label, value: e.person_minutes }))}
                format={(v) => v.toFixed(0)}
                valueLabel="person-minutes"
                labelWidth={240}
              />
            </ChartCard>
          )}
          {pred && (
            <p className="muted text-xs">
              Trained on {Object.entries(pred.model.trained_on ?? {}).map(([k, v]) => `${v} ${k.replace("_", " ")}`).join(", ")} cases.
              {hold &&
                ` On a building type it had not seen, its P95 of "everyone out" was off by ${Object.entries(hold)
                  .map(([k, v]) => `${Math.round(100 * v.total_time)}% (${k.replace("_", " ")})`)
                  .join(", ")} on average.`}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
