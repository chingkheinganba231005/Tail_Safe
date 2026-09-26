import { useEffect, useMemo, useRef, useState } from "react";
import type { StressRun } from "../App";
import { get } from "../api";
import { ChartCard } from "../components/ChartCard";
import { JobProgress } from "../components/JobProgress";
import { FloorChart } from "../components/charts/FloorChart";
import { LineChart } from "../components/charts/LineChart";
import { useWidth } from "../components/hooks";
import { useJob } from "../components/useJob";
import { clock, levelLabel, minutes } from "../lib/format";
import { dotsAt, frameAt } from "../lib/interp";
import type { BuildingView, MicroLevel, MicroResult } from "../types";
import { replayBody } from "./Stack3D";

interface Props {
  building: BuildingView;
  stress: StressRun;
  index: number | null;
  setIndex: (i: number) => void;
}

const SPEEDS = [1, 5, 15, 30, 60];
const R = 0.22; // drawn radius (m)

/** Screen 5: one scenario replayed person by person, top-down, one floor at a time. */
export function ReplayMicro({ building, stress, index, setIndex }: Props) {
  const idx = index ?? stress.result.worst_scenarios[0] ?? 0;
  const { job, error, running, run } = useJob<MicroResult>();
  const [res, setRes] = useState<{ jobId: string; result: MicroResult } | null>(null);
  const [level, setLevel] = useState<number | null>(null);
  const [floor, setFloor] = useState<MicroLevel | null>(null);
  const [floorError, setFloorError] = useState<string | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(15);

  useEffect(() => {
    let live = true;
    setRes(null);
    setFloor(null);
    setPlaying(false);
    setT(0);
    void run("/api/micro", { ...replayBody(building.id, stress.spec, idx, stress.seed, stress.runs), runs: undefined }).then(
      (out) => {
        if (!live || !out) return;
        setRes({ jobId: out.job.id, result: out.result });
        const r = out.result;
        const fire = r.info.fire_level;
        const busiest = r.levels[r.people_on_level[0].indexOf(Math.max(...r.people_on_level[0]))];
        setLevel(fire !== undefined && r.levels.includes(fire) ? fire : busiest);
      },
    );
    return () => {
      live = false;
    };
  }, [building.id, stress, idx, run]);

  useEffect(() => {
    if (!res || level === null) return;
    let live = true;
    setFloorError(null);
    get<MicroLevel>(`/api/micro/${res.jobId}/level/${level}`)
      .then((f) => live && setFloor(f))
      .catch((e: Error) => live && setFloorError(e.message));
    return () => {
      live = false;
    };
  }, [res, level]);

  const tEnd = floor ? (floor.times.at(-1) ?? 0) : 0;
  const last = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) {
      last.current = null;
      return;
    }
    let raf = 0;
    const tick = (now: number) => {
      const prev = last.current ?? now;
      last.current = now;
      setT((x) => {
        const next = x + ((now - prev) / 1000) * speed;
        if (next >= tEnd) {
          setPlaying(false);
          return tEnd;
        }
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, tEnd]);

  const r = res?.result;
  const k = r ? Math.min(r.frames - 1, Math.floor(t / r.frame_dt)) : 0;
  const levelPos = r && level !== null ? r.levels.indexOf(level) : -1;

  return (
    <div className="space-y-4">
      <section className="card flex flex-wrap items-center justify-between gap-2 p-4">
        <div>
          <h2 className="page-title">Replay, person by person</h2>
          <p className="secondary max-w-3xl text-sm">
            The microscopic engine re-runs scenario {idx} with every person as a disc on the floor plan — same people,
            reaction times and route choices as the fast engine, but physical queues at doors and on stairs.
          </p>
        </div>
        <label className="text-sm">
          <span className="secondary mr-2">Scenario</span>
          <select value={idx} onChange={(e) => setIndex(Number(e.target.value))}>
            {stress.result.worst_scenarios.map((i, n) => (
              <option key={i} value={i}>
                #{n + 1} worst (run {i})
              </option>
            ))}
            <option value={stress.result.median_scenario}>Median (run {stress.result.median_scenario})</option>
          </select>
        </label>
      </section>
      {running && <JobProgress job={job} error={null} label="Micro replay" />}
      {error && <JobProgress job={null} error={error} label="Micro replay" />}
      {r && (
        <>
          <Comparison r={r} />
          <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
            <section className="card space-y-3 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="font-semibold">
                  {level !== null ? levelLabel(level) : ""} at {clock(t)}
                  {r.info.fire_level === level && <span className="secondary text-sm font-normal"> · fire floor</span>}
                </h3>
                <label className="text-sm">
                  <span className="secondary mr-2">Floor</span>
                  <select value={level ?? ""} onChange={(e) => setLevel(Number(e.target.value))}>
                    {r.levels.map((lv) => (
                      <option key={lv} value={lv}>
                        {levelLabel(lv)}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              {floorError && <p className="text-sm">{floorError}</p>}
              {floor && level !== null && <FloorAnimation r={r} floor={floor} level={level} t={t} />}
              <div className="flex flex-wrap items-center gap-3">
                <button
                  className="btn text-sm"
                  disabled={!floor}
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
                  step={1}
                  value={Math.round(t)}
                  onChange={(e) => {
                    setPlaying(false);
                    setT(Number(e.target.value));
                  }}
                  className="min-w-40 flex-1"
                  aria-label="Time"
                />
                <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Playback speed">
                  {SPEEDS.map((s) => (
                    <option key={s} value={s}>
                      {s}× real time
                    </option>
                  ))}
                </select>
              </div>
              <DotLegend />
            </section>
            <section className="card p-4">
              <h3 className="font-semibold">People on each floor at {clock(t)}</h3>
              <p className="secondary mb-1 text-xs">Click a bar to watch that floor.</p>
              <FloorChart
                rows={r.levels.map((lv, i) => ({ level: lv, value: r.people_on_level[k]?.[i] ?? 0 }))}
                format={(v) => v.toFixed(0)}
                valueLabel="people"
                max={Math.max(1, ...r.people_on_level[0])}
                highlight={new Set(level !== null ? [level] : [])}
                onSelect={setLevel}
                height={Math.min(520, 40 + r.levels.length * 12)}
              />
            </section>
          </div>
          {levelPos >= 0 && (
            <ChartCard
              title={`People on ${levelLabel(level!)} over time`}
              table={{
                columns: ["Minute", "People on the floor"],
                rows: r.people_on_level
                  .filter((_, i) => i % 15 === 0)
                  .map((row, i) => [((i * 15 * r.frame_dt) / 60).toFixed(1), row[levelPos]]),
              }}
            >
              <LineChart
                series={[
                  {
                    name: "People on the floor",
                    x: r.people_on_level.map((_, i) => (i * r.frame_dt) / 60),
                    y: r.people_on_level.map((row) => row[levelPos]),
                    color: "var(--series-1)",
                  },
                ]}
                xLabel="minutes since alarm"
                yLabel="people"
                cursor={t / 60}
                height={180}
              />
            </ChartCard>
          )}
        </>
      )}
    </div>
  );
}

function Comparison({ r }: { r: MicroResult }) {
  const c = r.comparison;
  const rows: [string, number, number][] = [
    ["Half of the walkers out", c.meso.p50_s, c.micro.p50_s],
    ["95% of the walkers out", c.meso.p95_s, c.micro.p95_s],
    ["Last walker out", c.meso.last_s, c.micro.last_s],
  ];
  return (
    <section className="card p-4">
      <h3 className="font-semibold">Fast engine vs person-by-person, this scenario</h3>
      <p className="secondary mb-2 text-sm">
        {c.walkers} of {c.occupants} people walk; households waiting for a lift or for rescue keep their times from
        the fast engine. {r.summary.not_out > 0 ? `${r.summary.not_out} people did not get out in the replay.` : ""}
      </p>
      <table className="tabular w-full max-w-xl text-sm">
        <thead>
          <tr className="secondary text-left">
            <th className="py-1 font-medium" />
            <th className="py-1 text-right font-medium">Fast (meso)</th>
            <th className="py-1 text-right font-medium">Person by person</th>
            <th className="py-1 text-right font-medium">Difference</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([label, a, b]) => (
            <tr key={label} style={{ borderTop: "1px solid var(--grid)" }}>
              <td className="py-1">{label}</td>
              <td className="py-1 text-right">{minutes(a)}</td>
              <td className="py-1 text-right">{minutes(b)}</td>
              <td className="py-1 text-right">
                {b - a >= 0 ? "+" : "−"}
                {minutes(Math.abs(b - a))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function FloorAnimation({ r, floor, level, t }: { r: MicroResult; floor: MicroLevel; level: number; t: number }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const rooms = useMemo(() => r.rooms.filter((x) => x.level === level), [r, level]);
  const box = useMemo(() => {
    const pts = rooms.flatMap((x) => x.polygon);
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    return { x0: Math.min(...xs) - 1, x1: Math.max(...xs) + 1, y0: Math.min(...ys) - 1, y1: Math.max(...ys) + 1 };
  }, [rooms]);
  const scale = Math.min(width / (box.x1 - box.x0), 560 / (box.y1 - box.y0));
  const X = (x: number) => (x - box.x0) * scale;
  const Y = (y: number) => (box.y1 - y) * scale;
  const dots = dotsAt(floor.times, floor.frames, t, floor.scale);
  const k = frameAt(floor.times, t);
  const walking = dots.filter((d) => d.state === 1).length;
  return (
    <div ref={ref} className="w-full">
      <svg
        width={(box.x1 - box.x0) * scale}
        height={(box.y1 - box.y0) * scale}
        role="img"
        aria-label={`${dots.length} people on ${levelLabel(level)} at ${clock(floor.times[k] ?? 0)}, ${walking} walking`}
        style={{ display: "block", margin: "0 auto" }}
      >
        {rooms.map((room) => (
          <polygon
            key={room.id}
            points={room.polygon.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ")}
            fill={
              room.type === "unit"
                ? "var(--surface-1)"
                : room.type === "stair_landing"
                  ? "color-mix(in srgb, var(--text-secondary) 22%, var(--surface-1))"
                  : "color-mix(in srgb, var(--text-secondary) 9%, var(--surface-1))"
            }
            stroke="var(--axis)"
            strokeWidth={0.75}
          />
        ))}
        {dots.map((d) => {
          const moving = d.state === 1 || d.state === 2;
          return (
            <circle
              key={d.id}
              cx={X(d.x)}
              cy={Y(d.y)}
              r={Math.max(2.5, R * scale)}
              fill={d.state === 2 ? "var(--surface-1)" : moving ? "var(--series-1)" : "var(--text-muted)"}
              stroke={d.state === 2 ? "var(--series-1)" : "var(--surface-1)"}
              strokeWidth={d.state === 2 ? 1.5 : 0.75}
            />
          );
        })}
      </svg>
    </div>
  );
}

function DotLegend() {
  const item = (fill: string, stroke: string, label: string) => (
    <span className="flex items-center gap-1.5">
      <svg width="12" height="12" aria-hidden>
        <circle cx="6" cy="6" r="4.5" fill={fill} stroke={stroke} strokeWidth="1.5" />
      </svg>
      {label}
    </span>
  );
  return (
    <div className="secondary flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label="Legend">
      {item("var(--series-1)", "var(--series-1)", "walking")}
      {item("var(--surface-1)", "var(--series-1)", "on the stairs (drawn across the stair enclosure)")}
      {item("var(--text-muted)", "var(--text-muted)", "not moving yet / waiting")}
    </div>
  );
}
