import { useEffect, useState, type ReactNode } from "react";
import { get } from "../api";
import type { StressRun } from "../App";
import { JobProgress } from "../components/JobProgress";
import { useJob } from "../components/useJob";
import { levelLabel } from "../lib/format";
import type { BuildingView, Dist, ScenarioSpec, StressResult, TimeSlot } from "../types";

interface Props {
  building: BuildingView;
  spec: ScenarioSpec | null;
  setSpec: (s: ScenarioSpec) => void;
  onStress: (run: StressRun) => void;
}

const SLOTS: { id: TimeSlot; label: string }[] = [
  { id: "weekday_day", label: "Weekday day" },
  { id: "weekday_night", label: "Weekday night" },
  { id: "weekend_day", label: "Weekend day" },
  { id: "weekend_night", label: "Weekend night (3 a.m.)" },
];

/** Keep a spec consistent with a building (fire floor, stairs, lifts). */
function fitSpec(spec: ScenarioSpec, view: BuildingView): ScenarioSpec {
  const b = view.building;
  const top = Math.max(...b.levels.map((l) => l.index));
  const stairs = new Set(b.stairs.map((s) => s.id));
  const nonFf = b.lifts.filter((l) => !l.firefighting).length;
  return {
    ...spec,
    fire_level: spec.fire_level != null ? Math.min(spec.fire_level, top) : null,
    stair_blockages: spec.stair_blockages.filter((x) => stairs.has(x.stair)),
    lifts_out_of_service: Math.min(spec.lifts_out_of_service, b.lifts.length),
    evacuation_lifts: spec.evacuation_lifts && nonFf > 0,
    stair_assignment: {},
    phased_release: {},
    warden_levels: spec.warden_levels.filter((l) => l <= top),
    capacity_multipliers: {},
  };
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block text-sm">
      <span className="secondary block text-xs">{label}</span>
      {children}
      {hint && <span className="muted block text-xs">{hint}</span>}
    </label>
  );
}

/** A share in [0, 1] that can fall back to the registry default (null). */
function Share({
  label,
  value,
  onChange,
  max = 1,
}: {
  label: string;
  value: number | null | undefined;
  onChange: (v: number | null) => void;
  max?: number;
}) {
  const isDefault = value === null || value === undefined;
  return (
    <div className="text-sm">
      <div className="flex items-center justify-between">
        <span className="secondary text-xs">{label}</span>
        <label className="muted flex items-center gap-1 text-xs">
          <input type="checkbox" checked={isDefault} onChange={(e) => onChange(e.target.checked ? null : 0)} />
          registry default
        </label>
      </div>
      {!isDefault && (
        <div className="flex items-center gap-2">
          <input
            type="range"
            min={0}
            max={max}
            step={0.01}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
            className="w-full"
            aria-label={label}
          />
          <span className="tabular w-12 text-right">{Math.round(100 * value)}%</span>
        </div>
      )}
    </div>
  );
}

function blockTime(d: Dist): string {
  if (d.dist === "constant") return `${((d.value ?? 0) / 60).toFixed(1)} min`;
  if (d.dist === "uniform") return `${((d.min ?? 0) / 60).toFixed(0)}–${((d.max ?? 0) / 60).toFixed(0)} min`;
  return d.dist;
}

/** Screen 2: what to stress-test (time of day, people, fire, stairs, lifts). */
export function ScenarioBuilder({ building, spec, setSpec, onStress }: Props) {
  const b = building.building;
  const [runs, setRuns] = useState(300);
  const [seed, setSeed] = useState(0);
  const { job, error, running, run } = useJob<StressResult>();

  useEffect(() => {
    if (spec) {
      setSpec(fitSpec(spec, building));
      return;
    }
    get<ScenarioSpec>("/api/specs/demo")
      .then((s) => setSpec(fitSpec(s, building)))
      .catch(() => undefined);
  }, [building.id]);

  if (!spec) return <p className="muted">Loading scenario…</p>;
  const set = (patch: Partial<ScenarioSpec>) => setSpec({ ...spec, ...patch });
  const levels = b.levels.map((l) => l.index).filter((l) => l > 0);
  const hazardOn = spec.hazard?.enabled ?? false;
  const nonFf = b.lifts.filter((l) => !l.firefighting).length;

  const start = async () => {
    const out = await run("/api/stress", { building_id: building.id, spec, runs, seed });
    if (out) onStress({ jobId: out.job.id, result: out.result, spec, runs, seed });
  };

  const loadDemo = async () => setSpec(fitSpec(await get<ScenarioSpec>("/api/specs/demo"), building));

  return (
    <div className="space-y-4">
      <section className="card p-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <h2 className="page-title">Scenario</h2>
            <p className="secondary text-sm">
              Fix what is known; everything else (who is home, reaction times, fire growth, when the fire
              service arrives…) is sampled from the parameter registry in every run.
            </p>
          </div>
          <button className="btn-ghost text-sm" onClick={loadDemo}>
            Load pitch scenario
          </button>
        </div>
      </section>
      <div className="grid gap-4 lg:grid-cols-2">
        <section className="card space-y-3 p-4">
          <h3 className="font-semibold">When and who</h3>
          <div role="group" aria-label="Time of day" className="flex flex-wrap gap-1">
            {SLOTS.map((s) => (
              <button
                key={s.id}
                className="btn-ghost text-sm"
                aria-pressed={spec.time_slot === s.id}
                onClick={() => set({ time_slot: s.id })}
              >
                {s.label}
              </button>
            ))}
          </div>
          <Share label="Residents aged 65+" value={spec.share_65_plus} onChange={(v) => set({ share_65_plus: v })} max={0.6} />
          <Share
            label="Of those, aged 80+"
            value={spec.share_80_plus_of_65_plus}
            onChange={(v) => set({ share_80_plus_of_65_plus: v })}
          />
          <Share label="Vacant flats" value={spec.vacancy_rate} onChange={(v) => set({ vacancy_rate: v })} max={0.5} />
          <Share
            label="Households that first go against the flow (fetch family, check on neighbours)"
            value={spec.counter_flow_probability}
            onChange={(v) => set({ counter_flow_probability: v })}
          />
        </section>
        <section className="card space-y-3 p-4">
          <h3 className="font-semibold">Fire and smoke</h3>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={hazardOn}
              onChange={(e) => set({ hazard: { ...(spec.hazard ?? {}), enabled: e.target.checked } })}
            />
            Simulate smoke spread and tenability (needed for P(RSET &gt; ASET))
          </label>
          <Field label="Fire floor">
            <select
              value={spec.fire_level ?? ""}
              onChange={(e) => set({ fire_level: e.target.value === "" ? null : Number(e.target.value) })}
            >
              <option value="">Random floor</option>
              {levels.map((l) => (
                <option key={l} value={l}>
                  {levelLabel(l)}
                </option>
              ))}
            </select>
          </Field>
          {hazardOn && (
            <Share
              label="Fire flat door left open"
              value={spec.hazard?.door_open_probability}
              onChange={(v) => set({ hazard: { ...spec.hazard!, door_open_probability: v } })}
            />
          )}
        </section>
        <section className="card space-y-3 p-4">
          <h3 className="font-semibold">Staircases</h3>
          {b.stairs.map((s) => {
            const blk = spec.stair_blockages.find((x) => x.stair === s.id);
            const random = blk?.time.dist === "uniform";
            const update = (time: Dist | null) =>
              set({
                stair_blockages: [
                  ...spec.stair_blockages.filter((x) => x.stair !== s.id),
                  ...(time ? [{ stair: s.id, time }] : []),
                ],
              });
            return (
              <div key={s.id} className="flex flex-wrap items-center gap-2 text-sm">
                <label className="flex w-52 items-center gap-2">
                  <input
                    type="checkbox"
                    checked={!!blk}
                    onChange={(e) => update(e.target.checked ? { dist: "constant", value: 240 } : null)}
                  />
                  {s.label} smoke-logged
                </label>
                {blk && (
                  <>
                    <select
                      value={random ? "uniform" : "constant"}
                      onChange={(e) =>
                        update(
                          e.target.value === "uniform"
                            ? { dist: "uniform", min: 120, max: 900 }
                            : { dist: "constant", value: 240 },
                        )
                      }
                      aria-label={`${s.label} blockage timing`}
                    >
                      <option value="constant">at</option>
                      <option value="uniform">between</option>
                    </select>
                    {random ? (
                      <>
                        <input
                          type="number"
                          className="w-16"
                          min={0}
                          value={(blk.time.min ?? 0) / 60}
                          onChange={(e) => update({ ...blk.time, min: 60 * Number(e.target.value) })}
                          aria-label="from (min)"
                        />
                        <span>and</span>
                        <input
                          type="number"
                          className="w-16"
                          min={0}
                          value={(blk.time.max ?? 0) / 60}
                          onChange={(e) => update({ ...blk.time, max: 60 * Number(e.target.value) })}
                          aria-label="to (min)"
                        />
                      </>
                    ) : (
                      <input
                        type="number"
                        className="w-16"
                        min={0}
                        step={0.5}
                        value={(blk.time.value ?? 0) / 60}
                        onChange={(e) => update({ dist: "constant", value: 60 * Number(e.target.value) })}
                        aria-label="time (min)"
                      />
                    )}
                    <span className="muted">min</span>
                  </>
                )}
              </div>
            );
          })}
          <Field label="Chance that one more random staircase is lost">
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                className="w-full"
                value={spec.random_stair_blockage?.probability ?? 0}
                onChange={(e) => {
                  const p = Number(e.target.value);
                  set({
                    random_stair_blockage: p > 0 ? { probability: p, time: { dist: "uniform", min: 0, max: 900 } } : null,
                  });
                }}
              />
              <span className="tabular w-12 text-right">
                {Math.round(100 * (spec.random_stair_blockage?.probability ?? 0))}%
              </span>
            </div>
          </Field>
        </section>
        <section className="card space-y-3 p-4">
          <h3 className="font-semibold">Lifts and rescue</h3>
          <Field label="Lifts out of service (chosen at random each run)">
            <input
              type="number"
              min={0}
              max={b.lifts.length}
              className="w-20"
              value={spec.lifts_out_of_service}
              onChange={(e) => set({ lifts_out_of_service: Number(e.target.value) })}
            />
          </Field>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              disabled={nonFf === 0}
              checked={spec.evacuation_lifts}
              onChange={(e) => set({ evacuation_lifts: e.target.checked })}
            />
            Use non-firefighting lifts to evacuate mobility-impaired residents
          </label>
          {spec.evacuation_lifts && (
            <div className="flex flex-wrap gap-3">
              <Field label="Dispatch">
                <select
                  value={spec.lift_priority}
                  onChange={(e) => set({ lift_priority: e.target.value as ScenarioSpec["lift_priority"] })}
                >
                  <option value="top_down">Highest floors first</option>
                  <option value="nearest">Nearest waiting floor</option>
                  <option value="bottom_up">Lowest floors first</option>
                </select>
              </Field>
              <Field label="Who may wait for a lift">
                <select
                  value={spec.lift_eligibility ?? "mobility_impaired"}
                  onChange={(e) => set({ lift_eligibility: e.target.value as ScenarioSpec["lift_eligibility"] })}
                >
                  <option value="mobility_impaired">Wheelchair users and frail older adults</option>
                  <option value="wheelchair_users">Wheelchair users only</option>
                </select>
              </Field>
            </div>
          )}
          <Field label="Fire-service rescue teams" hint="Empty = registry default">
            <input
              type="number"
              min={0}
              className="w-20"
              value={spec.rescue_teams ?? ""}
              onChange={(e) => set({ rescue_teams: e.target.value === "" ? null : Number(e.target.value) })}
            />
          </Field>
        </section>
      </div>
      <section className="card flex flex-wrap items-end gap-3 p-4">
        <Field label="Monte Carlo runs">
          <select value={runs} onChange={(e) => setRuns(Number(e.target.value))}>
            {[100, 300, 1000, 2000].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Seed">
          <input type="number" min={0} className="w-24" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
        </Field>
        <button className="btn" disabled={running} onClick={start}>
          {running ? "Running…" : "Run stress test"}
        </button>
        <p className="muted text-xs">
          {spec.stair_blockages.length > 0 &&
            `Blocked: ${spec.stair_blockages.map((x) => `Stair ${x.stair} at ${blockTime(x.time)}`).join(", ")}. `}
          Same seed → same scenarios, so runs are reproducible and comparable.
        </p>
      </section>
      <JobProgress job={job} error={error} label="Stress test" />
    </div>
  );
}
