import { useEffect, useState } from "react";
import { get } from "./api";
import { Bottlenecks } from "./screens/Bottlenecks";
import { BuildingSetup } from "./screens/BuildingSetup";
import { Optimize } from "./screens/Optimize";
import { ReplayMicro } from "./screens/ReplayMicro";
import { Report } from "./screens/Report";
import { ScenarioBuilder } from "./screens/ScenarioBuilder";
import { Stack3D } from "./screens/Stack3D";
import { StressResults } from "./screens/StressResults";
import { WhatIf } from "./screens/WhatIf";
import type { BottleneckResult, BuildingView, OptimizeResult, ScenarioSpec, StressResult } from "./types";

export type Step = "building" | "scenario" | "results" | "stack" | "replay" | "bottlenecks" | "optimize" | "whatif" | "report";

const STEPS: { id: Step; label: string; needs: "none" | "building" | "stress" }[] = [
  { id: "building", label: "Building", needs: "none" },
  { id: "scenario", label: "Scenario", needs: "building" },
  { id: "results", label: "Stress results", needs: "stress" },
  { id: "stack", label: "3D stack", needs: "stress" },
  { id: "replay", label: "Replay (people)", needs: "stress" },
  { id: "bottlenecks", label: "Bottlenecks", needs: "stress" },
  { id: "optimize", label: "Optimise", needs: "stress" },
  { id: "whatif", label: "What-if (live)", needs: "building" },
  { id: "report", label: "Briefing", needs: "stress" },
];

export interface StressRun {
  jobId: string;
  result: StressResult;
  spec: ScenarioSpec;
  runs: number;
  seed: number;
}

type Theme = "auto" | "light" | "dark";

export function App() {
  const [step, setStep] = useState<Step>("building");
  const [disclaimer, setDisclaimer] = useState<string>("");
  const [building, setBuilding] = useState<BuildingView | null>(null);
  const [spec, setSpec] = useState<ScenarioSpec | null>(null);
  const [stress, setStress] = useState<StressRun | null>(null);
  const [replayIndex, setReplayIndex] = useState<number | null>(null);
  const [bottlenecks, setBottlenecks] = useState<BottleneckResult | null>(null);
  const [optimization, setOptimization] = useState<OptimizeResult | null>(null);
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      return (localStorage.getItem("tailsafe-theme") as Theme) || "auto";
    } catch {
      return "auto";
    }
  });

  useEffect(() => {
    get<{ disclaimer: string }>("/api/health")
      .then((h) => setDisclaimer(h.disclaimer))
      .catch(() => setDisclaimer(""));
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "auto") delete root.dataset.theme;
    else root.dataset.theme = theme;
    try {
      localStorage.setItem("tailsafe-theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);

  const available = (needs: string) =>
    needs === "none" || (needs === "building" && building !== null) || (needs === "stress" && stress !== null);

  const onBuilding = (b: BuildingView) => {
    setBuilding(b);
    setStress(null);
    setBottlenecks(null);
    setOptimization(null);
  };

  const onStress = (run: StressRun) => {
    setStress(run);
    setBottlenecks(null);
    setOptimization(null);
    setReplayIndex(run.result.worst_scenarios[0] ?? 0);
    setStep("results");
  };

  return (
    <div className="min-h-screen" style={{ background: "var(--page)" }}>
      <header className="border-b" style={{ borderColor: "var(--border)", background: "var(--surface-1)" }}>
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3">
          <div>
            <h1 className="text-xl font-semibold">TailSafe</h1>
            <p className="secondary text-sm">
              Tail-risk evacuation stress-testing for high-rise buildings: the worst 5% of outcomes, why, and
              what fixes them.
            </p>
          </div>
          <div role="group" aria-label="Colour theme" className="flex gap-1">
            {(["auto", "light", "dark"] as const).map((t) => (
              <button key={t} className="btn-ghost text-xs" aria-pressed={theme === t} onClick={() => setTheme(t)}>
                {t === "auto" ? "Auto" : t === "light" ? "Light" : "Dark"}
              </button>
            ))}
          </div>
        </div>
        {disclaimer && (
          <div className="px-4 pb-2">
            <p
              className="secondary mx-auto max-w-7xl rounded-md px-3 py-1.5 text-xs"
              style={{ border: "1px solid var(--border)" }}
              role="note"
            >
              <strong>Responsible use.</strong> {disclaimer}
            </p>
          </div>
        )}
      </header>
      <div className="mx-auto grid max-w-7xl gap-4 px-4 py-4 md:grid-cols-[200px_1fr]">
        <nav aria-label="Steps">
          <ol className="flex gap-1 overflow-x-auto md:flex-col">
            {STEPS.map((s, i) => {
              const ok = available(s.needs);
              return (
                <li key={s.id}>
                  <button
                    className="btn-ghost w-full text-left text-sm whitespace-nowrap"
                    aria-pressed={step === s.id}
                    aria-current={step === s.id ? "step" : undefined}
                    disabled={!ok}
                    style={{ opacity: ok ? 1 : 0.45 }}
                    onClick={() => setStep(s.id)}
                  >
                    <span className="muted tabular mr-2">{i + 1}</span>
                    {s.label}
                  </button>
                </li>
              );
            })}
          </ol>
        </nav>
        <main className="min-w-0">
          {step === "building" && (
            <BuildingSetup building={building} onBuilding={onBuilding} onConfirm={() => setStep("scenario")} />
          )}
          {step === "scenario" && building && (
            <ScenarioBuilder building={building} spec={spec} setSpec={setSpec} onStress={onStress} />
          )}
          {step === "results" && stress && building && (
            <StressResults
              stress={stress}
              building={building}
              onReplay={(i) => {
                setReplayIndex(i);
                setStep("stack");
              }}
            />
          )}
          {step === "stack" && stress && building && (
            <Stack3D building={building} stress={stress} index={replayIndex} setIndex={setReplayIndex} />
          )}
          {step === "replay" && stress && building && (
            <ReplayMicro building={building} stress={stress} index={replayIndex} setIndex={setReplayIndex} />
          )}
          {step === "bottlenecks" && stress && building && (
            <Bottlenecks building={building} stress={stress} result={bottlenecks} setResult={setBottlenecks} />
          )}
          {step === "whatif" && building && <WhatIf building={building} spec={spec} onStress={onStress} />}
          {step === "report" && stress && building && (
            <Report building={building} stress={stress} bottlenecks={bottlenecks} optimization={optimization} />
          )}
          {step === "optimize" && stress && building && (
            <Optimize building={building} stress={stress} result={optimization} setResult={setOptimization} />
          )}
        </main>
      </div>
    </div>
  );
}
