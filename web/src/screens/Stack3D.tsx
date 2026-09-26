import { useEffect, useState } from "react";
import type { StressRun } from "../App";
import { ChartCard } from "../components/ChartCard";
import { FloorChart } from "../components/charts/FloorChart";
import { LineChart } from "../components/charts/LineChart";
import { JobProgress } from "../components/JobProgress";
import { StackLegend, StackView3D } from "../components/StackView3D";
import { useJob } from "../components/useJob";
import { clock, levelLabel, minutes } from "../lib/format";
import { useMode } from "../lib/theme";
import type { BuildingView, Replay, ScenarioSpec } from "../types";

interface Props {
  building: BuildingView;
  stress: StressRun;
  index: number | null;
  setIndex: (i: number) => void;
}

/** Play/pause + scrubber over replay frames. */
export function usePlayer(frames: number) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setFrame((f) => {
        if (f + 1 >= frames) {
          setPlaying(false);
          return f;
        }
        return f + 1;
      });
    }, 120 / speed);
    return () => window.clearInterval(id);
  }, [playing, frames, speed]);
  useEffect(() => setFrame(0), [frames]);
  return { frame, setFrame, playing, setPlaying, speed, setSpeed };
}

export function PlayerControls({
  player,
  times,
}: {
  player: ReturnType<typeof usePlayer>;
  times: number[];
}) {
  const { frame, setFrame, playing, setPlaying, speed, setSpeed } = player;
  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        className="btn text-sm"
        onClick={() => {
          if (frame >= times.length - 1) setFrame(0);
          setPlaying(!playing);
        }}
      >
        {playing ? "Pause" : "Play"}
      </button>
      <input
        type="range"
        min={0}
        max={Math.max(0, times.length - 1)}
        value={frame}
        onChange={(e) => {
          setPlaying(false);
          setFrame(Number(e.target.value));
        }}
        className="min-w-40 flex-1"
        aria-label="Time"
      />
      <span className="tabular w-24 text-right text-sm">t = {clock(times[frame] ?? 0)}</span>
      <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Playback speed">
        {[0.5, 1, 2, 4].map((s) => (
          <option key={s} value={s}>
            {s}×
          </option>
        ))}
      </select>
    </div>
  );
}

export function replayBody(buildingId: string, spec: ScenarioSpec, index: number, seed: number, runs: number, batch = 100) {
  return { building_id: buildingId, spec, index, seed, runs, batch_size: batch };
}

/** Screen 4: one scenario over time — smoke, stair queues, people still inside. */
export function Stack3D({ building, stress, index, setIndex }: Props) {
  const mode = useMode();
  const { job, error, running, run } = useJob<Replay>();
  const [replay, setReplay] = useState<Replay | null>(null);
  const player = usePlayer(replay?.times.length ?? 0);
  const idx = index ?? stress.result.worst_scenarios[0] ?? 0;

  useEffect(() => {
    let live = true;
    setReplay(null);
    void run("/api/replay", replayBody(building.id, stress.spec, idx, stress.seed, stress.runs)).then((out) => {
      if (live && out) setReplay(out.result);
    });
    return () => {
      live = false;
    };
  }, [building.id, stress, idx, run]);

  const r = replay;
  const qMax = r ? Math.max(5, ...Object.values(r.stair_queues).flatMap((g) => g.flatMap((x) => x))) : 5;
  const tMin = r?.times.map((t) => t / 60) ?? [];
  const total = r ? r.remaining.map((row) => row.reduce((a, b) => a + b, 0)) : [];

  return (
    <div className="space-y-4">
      <section className="card flex flex-wrap items-center justify-between gap-2 p-4">
        <div>
          <h2 className="text-lg font-semibold">3D stack view</h2>
          <p className="secondary text-sm">
            Scenario {idx} re-simulated with full time series (same random draws as in the stress test).
          </p>
        </div>
        <label className="text-sm">
          <span className="secondary mr-2">Scenario</span>
          <select value={idx} onChange={(e) => setIndex(Number(e.target.value))}>
            {stress.result.worst_scenarios.map((i, k) => (
              <option key={i} value={i}>
                #{k + 1} worst (run {i})
              </option>
            ))}
            <option value={stress.result.median_scenario}>Median (run {stress.result.median_scenario})</option>
          </select>
        </label>
      </section>
      {running && <JobProgress job={job} error={null} label="Replay" />}
      {error && <JobProgress job={null} error={error} label="Replay" />}
      {r && (
        <>
          <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
            <section className="card space-y-3 p-4">
              <StackView3D building={building.building} replay={r} frame={player.frame} queueMax={qMax} />
              <PlayerControls player={player} times={r.times} />
              <StackLegend mode={mode} queueMax={qMax} />
            </section>
            <section className="card space-y-2 p-4 text-sm">
              <h3 className="font-semibold">This scenario</h3>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                <dt className="secondary">Occupants</dt>
                <dd className="tabular">{r.summary.occupants}</dd>
                <dt className="secondary">Everyone out</dt>
                <dd className="tabular">{minutes(r.summary.total_time_s)}</dd>
                <dt className="secondary">95% out</dt>
                <dd className="tabular">{minutes(r.summary.p95_exit_s)}</dd>
                <dt className="secondary">Rescued</dt>
                <dd className="tabular">{r.summary.rescued_occupants}</dd>
                <dt className="secondary">Incapacitated</dt>
                <dd className="tabular">{r.summary.incapacitated_occupants}</dd>
                {r.info.fire && (
                  <>
                    <dt className="secondary">Fire</dt>
                    <dd>
                      {levelLabel(r.info.fire_level ?? 0)}, flat door {r.info.fire.door_open ? "open" : "closed"}, peak{" "}
                      {(r.info.fire.peak_kw / 1000).toFixed(1)} MW
                    </dd>
                  </>
                )}
                {r.info.blocked_stairs && Object.keys(r.info.blocked_stairs).length > 0 && (
                  <>
                    <dt className="secondary">Stairs lost</dt>
                    <dd>
                      {Object.entries(r.info.blocked_stairs)
                        .map(([s, t]) => `${s} at ${clock(t)}`)
                        .join(", ")}
                    </dd>
                  </>
                )}
                {r.info.lifts_out && r.info.lifts_out.length > 0 && (
                  <>
                    <dt className="secondary">Lifts out</dt>
                    <dd>{r.info.lifts_out.join(", ")}</dd>
                  </>
                )}
                {r.info.rescue_start !== undefined && (
                  <>
                    <dt className="secondary">Rescue starts</dt>
                    <dd className="tabular">{clock(r.info.rescue_start)}</dd>
                  </>
                )}
              </dl>
              <h3 className="pt-3 font-semibold">Still on each floor at {clock(r.times[player.frame] ?? 0)}</h3>
              <FloorChart
                rows={r.levels.map((lv, k) => ({ level: lv, value: r.remaining[player.frame]?.[k] ?? 0 }))}
                format={(v) => v.toFixed(0)}
                valueLabel="people"
                max={Math.max(1, ...r.remaining[0])}
                height={Math.min(420, 40 + r.levels.length * 10)}
              />
            </section>
          </div>
          <ChartCard
            title="People evacuated and still inside"
            table={{
              columns: ["Minute", "Evacuated", "Still inside"],
              rows: tMin.filter((_, i) => i % 6 === 0).map((t, k) => [t.toFixed(1), r.evacuated[k * 6], total[k * 6]]),
            }}
          >
            <LineChart
              series={[
                { name: "Evacuated", x: tMin, y: r.evacuated, color: "var(--series-1)" },
                { name: "Still inside", x: tMin, y: total, color: "var(--series-2)" },
              ]}
              xLabel="minutes since alarm"
              yLabel="people"
              cursor={tMin[player.frame]}
            />
          </ChartCard>
        </>
      )}
    </div>
  );
}
