import type React from "react";
import { useMemo, useState } from "react";
import type { StressRun } from "../App";
import { ChartCard } from "../components/ChartCard";
import { FloorChart } from "../components/charts/FloorChart";
import { Histogram, histogramTable } from "../components/charts/Histogram";
import { Meter } from "../components/charts/Meter";
import { RankChart } from "../components/charts/RankChart";
import { StatTile } from "../components/StatTile";
import { LOSS_LABELS, LOSS_SHORT, levelLabel, min, minutes, pct } from "../lib/format";
import { LOSSES, type BuildingView, type Loss } from "../types";

interface Props {
  stress: StressRun;
  building: BuildingView;
  onReplay: (index: number) => void;
}

/** Screen 3: the distribution, its tail, and who is in it. */
export function StressResults({ stress, building, onReplay }: Props) {
  const r = stress.result;
  const [loss, setLoss] = useState<Loss>("total_time");
  const risk = r.risk[loss];
  const values = useMemo(() => r.losses[loss].map((v) => v / 60), [r, loss]);
  const series = useMemo(() => [{ name: LOSS_SHORT[loss], values, color: "var(--series-1)" }], [loss, values]);
  const bd = r.breakdown[loss === "p95_occupant_time" ? "total_time" : loss];
  const ten = r.tenability;
  const congestion = Object.entries(r.stair_congestion);
  const congMax = Math.max(1, ...congestion.flatMap(([, rows]) => rows.map((x) => x.person_seconds / 60)));

  return (
    <div className="space-y-4">
      <section className="card p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h2 className="page-title">Stress-test results</h2>
            <p className="secondary text-sm">
              {building.summary.name} · {r.runs} scenarios (seed {r.seed}) · ≈{r.occupants_mean.toFixed(0)} occupants ·
              computed in {r.elapsed_s.toFixed(0)} s
            </p>
          </div>
          <div role="group" aria-label="Outcome" className="flex flex-wrap gap-1">
            {LOSSES.map((l) => (
              <button key={l} className="btn-ghost text-sm" aria-pressed={loss === l} onClick={() => setLoss(l)}>
                {LOSS_SHORT[l]}
              </button>
            ))}
          </div>
        </div>
      </section>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label={`CVaR₉₅ — average of the worst 5%`}
          value={minutes(risk.cvar.value)}
          sub={`95% CI ${min(risk.cvar.lo)}–${min(risk.cvar.hi)} min`}
          emphasis
        />
        <StatTile label="P95" value={minutes(risk.p95.value)} sub={`95% CI ${min(risk.p95.lo)}–${min(risk.p95.hi)} min`} />
        <StatTile label="Mean" value={minutes(risk.mean.value)} sub={`95% CI ${min(risk.mean.lo)}–${min(risk.mean.hi)} min`} />
        {ten.p_rset_exceeds_aset ? (
          <Meter
            label="P(RSET > ASET)"
            estimate={ten.p_rset_exceeds_aset}
            description="Share of scenarios where someone is still on a floor after it becomes untenable."
          />
        ) : (
          <StatTile label="P(RSET > ASET)" value="–" sub="Enable smoke in the scenario" />
        )}
      </div>
      <ChartCard
        title={LOSS_LABELS[loss]}
        subtitle="Each bar counts scenarios. The mean hides the tail; CVaR₉₅ is the average of the worst 5%."
        table={histogramTable(series)}
      >
        <Histogram
          series={series}
          xLabel="minutes"
          markers={[
            { x: risk.mean.value / 60, label: `mean ${min(risk.mean.value)}` },
            { x: risk.p95.value / 60, label: `P95 ${min(risk.p95.value)}` },
            { x: risk.cvar.value / 60, label: `CVaR₉₅ ${min(risk.cvar.value)}`, strong: true },
          ]}
        />
      </ChartCard>
      {bd && (
        <div className="grid gap-4 lg:grid-cols-2">
          <ChartCard
            title="Who is still inside in the worst 5%"
            subtitle={bd.headline}
            table={{
              columns: ["Household type", "Share of occupants", "Share of late stragglers", "Risk ratio"],
              rows: bd.profiles.map((p) => [
                p.category,
                pct(p.occupant_share, 1),
                pct(p.straggler_share, 0),
                p.risk_ratio === null ? "–" : p.risk_ratio.toFixed(1),
              ]),
            }}
          >
            <RankChart
              rows={bd.profiles
                .filter((p) => p.risk_ratio !== null)
                .map((p) => ({
                  key: p.category,
                  label: p.category,
                  value: p.risk_ratio ?? 0,
                  note: `${pct(p.occupant_share, 1)} of occupants, ${pct(p.straggler_share)} of late stragglers`,
                }))}
              format={(v) => `×${v.toFixed(1)}`}
              valueLabel="risk ratio (share of stragglers ÷ share of occupants)"
            />
          </ChartCard>
          <ChartCard
            title="Floors that fail"
            subtitle="P(someone is still on the floor after it becomes untenable), with 95% CI."
            table={{
              columns: ["Floor", "P(RSET > ASET)", "95% CI"],
              rows: r.floor_exceedance.map((f) => [levelLabel(f.level), f.p.toFixed(2), `${f.lo.toFixed(2)}–${f.hi.toFixed(2)}`]),
            }}
          >
            {r.floor_exceedance.length ? (
              <FloorChart
                rows={r.floor_exceedance.map((f) => ({ level: f.level, value: f.p, lo: f.lo, hi: f.hi }))}
                format={(v) => v.toFixed(2)}
                valueLabel="P(RSET > ASET)"
                max={Math.max(0.05, ...r.floor_exceedance.map((f) => f.hi))}
              />
            ) : (
              <p className="muted text-sm">Smoke was not simulated.</p>
            )}
          </ChartCard>
        </div>
      )}
      {congestion.length > 0 && (
        <ChartCard
          title="Where stairs queue in the worst 5%"
          subtitle="Average person-minutes spent queueing on each flight, by floor (same scale for every stair)."
          table={{
            columns: ["Stair", "Floor", "Person-minutes"],
            rows: congestion.flatMap(([s, rows]) => rows.map((x) => [s, levelLabel(x.level), (x.person_seconds / 60).toFixed(1)])),
          }}
        >
          <div
            className="stair-grid grid gap-4"
            style={{ "--cols": congestion.length } as React.CSSProperties}
          >
            {congestion.map(([s, rows]) => (
              <div key={s}>
                <div className="text-sm font-semibold">Stair {s}</div>
                <FloorChart
                  rows={rows.map((x) => ({ level: x.level, value: x.person_seconds / 60 }))}
                  format={(v) => v.toFixed(0)}
                  valueLabel="person-minutes"
                  max={congMax}
                />
              </div>
            ))}
          </div>
        </ChartCard>
      )}
      <section className="card p-4">
        <h3 className="font-semibold">Replay a scenario in 3D</h3>
        <p className="secondary mb-2 text-sm">The ten worst scenarios by total time, and the median one.</p>
        <div className="flex flex-wrap gap-2">
          {r.worst_scenarios.map((i, k) => (
            <button key={i} className="btn-ghost text-sm" onClick={() => onReplay(i)}>
              #{k + 1} worst <span className="muted">(run {i})</span>
            </button>
          ))}
          <button className="btn-ghost text-sm" onClick={() => onReplay(r.median_scenario)}>
            Median <span className="muted">(run {r.median_scenario})</span>
          </button>
        </div>
      </section>
    </div>
  );
}
