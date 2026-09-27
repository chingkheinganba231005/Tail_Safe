import { useEffect, useState } from "react";
import { get } from "./api";
import { Icon } from "./components/Icon";
import { Logo } from "./components/Logo";
import { Bottlenecks } from "./screens/Bottlenecks";
import { BuildingSetup } from "./screens/BuildingSetup";
import { Optimize } from "./screens/Optimize";
import { ReplayMicro } from "./screens/ReplayMicro";
import { Report } from "./screens/Report";
import { ScenarioBuilder } from "./screens/ScenarioBuilder";
import { Stack3D } from "./screens/Stack3D";
import { StressResults } from "./screens/StressResults";
import { WhatIf } from "./screens/WhatIf";
import { STATIC } from "./static/site";
import type { BottleneckResult, BuildingView, OptimizeResult, ScenarioSpec, StressResult } from "./types";

export type Step = "building" | "scenario" | "results" | "stack" | "replay" | "bottlenecks" | "optimize" | "whatif" | "report";

const STEPS: { id: Step; label: string; action: string; needs: "none" | "building" | "stress" }[] = [
  { id: "building", label: "Building", action: "Choose the building", needs: "none" },
  { id: "scenario", label: "Scenario", action: "Set the scenario", needs: "building" },
  { id: "results", label: "Results", action: "See how bad the tail is", needs: "stress" },
  { id: "stack", label: "3D stack", action: "Watch it floor by floor", needs: "stress" },
  { id: "replay", label: "Replay", action: "Replay people moving", needs: "stress" },
  { id: "bottlenecks", label: "Bottlenecks", action: "Find what causes it", needs: "stress" },
  { id: "optimize", label: "Optimise", action: "Test operational fixes", needs: "stress" },
  { id: "whatif", label: "What-if", action: "Try changes instantly", needs: "building" },
  { id: "report", label: "Briefing", action: "Get the one-page briefing", needs: "stress" },
];

const GROUPS: { title: string; steps: Step[] }[] = [
  { title: "Set up", steps: ["building", "scenario"] },
  { title: "Understand the worst nights", steps: ["results", "stack", "replay", "bottlenecks"] },
  { title: "Make it safer", steps: ["optimize", "whatif", "report"] },
];

const NEEDS_TEXT = { building: "choose a building first", stress: "run a stress test first" } as const;

export interface StressRun {
  jobId: string;
  result: StressResult;
  spec: ScenarioSpec;
  runs: number;
  seed: number;
}

type Theme = "auto" | "light" | "dark";

const REPO = "https://github.com/chingkheinganba231005/Tail_Safe";

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

  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [step]);

  const index = STEPS.findIndex((s) => s.id === step);
  const current = STEPS[index];
  const prev = index > 0 ? STEPS[index - 1] : null;
  const next = STEPS.slice(index + 1).find((s) => available(s.needs)) ?? null;
  // The first two screens advance with their own main button.
  const showContinue = next !== null && step !== "building" && step !== "scenario";

  const stepList = (
    <ol>
      {GROUPS.map((g, gi) => (
        <li key={g.title} className="steps-group" data-active={g.steps.includes(step)}>
          <span className="steps-marker" aria-hidden>
            {gi + 1}
          </span>
          <p className="pt-1 pb-1.5 font-bold leading-tight">{g.title}</p>
          <ul>
            {g.steps.map((id) => {
              const s = STEPS.find((x) => x.id === id)!;
              const ok = available(s.needs);
              return (
                <li key={id}>
                  <button
                    className="steps-link"
                    aria-current={step === id ? "step" : undefined}
                    disabled={!ok}
                    onClick={() => setStep(id)}
                  >
                    {s.action}
                    {!ok && <span className="sr-only"> ({NEEDS_TEXT[s.needs as "building" | "stress"]})</span>}
                  </button>
                </li>
              );
            })}
          </ul>
        </li>
      ))}
    </ol>
  );

  return (
    <div className="flex min-h-screen flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:p-2">
        Skip to main content
      </a>
      <header className="site-header">
        <div className="mx-auto flex max-w-[1180px] items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <a
            href="#"
            className="flex items-center gap-2.5"
            onClick={(e) => {
              e.preventDefault();
              setStep("building");
            }}
          >
            <Logo size={32} tail="#3ccf7f" />
            <span className="text-[1.45rem] leading-none font-bold tracking-tight">TailSafe</span>
          </a>
          <div role="group" aria-label="Colour theme" className="flex">
            {(["auto", "light", "dark"] as const).map((t) => (
              <button
                key={t}
                className="flex h-8 w-9 items-center justify-center"
                style={theme === t ? { background: "#ffffff", color: "#0b0c0c" } : { color: "#d0d0d0" }}
                aria-pressed={theme === t}
                aria-label={t === "auto" ? "Match system theme" : t === "light" ? "Light theme" : "Dark theme"}
                title={t === "auto" ? "Match system" : t === "light" ? "Light" : "Dark"}
                onClick={() => setTheme(t)}
              >
                <Icon name={t === "auto" ? "auto" : t === "light" ? "sun" : "moon"} size={16} />
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="border-b" style={{ borderColor: "var(--border)" }}>
        <details className="mx-auto max-w-[1180px] px-4 py-2.5 text-sm sm:px-6">
          <summary className="flex cursor-pointer list-none flex-wrap items-center gap-x-2.5 gap-y-1">
            <span className="phase-tag">{STATIC ? "Browser version" : "Prototype"}</span>
            <span>
              {STATIC
                ? "Results are pre-computed for four standard buildings; What-if runs live in your browser. "
                : ""}
              Decision support only — not a fire-safety assessment. All residents are synthetic.{" "}
              <span className="underline underline-offset-2">Read the notice</span>
            </span>
          </summary>
          <p className="secondary mt-2 max-w-3xl leading-relaxed">{disclaimer}</p>
          {STATIC && (
            <p className="secondary mt-2 max-w-3xl leading-relaxed">
              This browser version shows results recorded from the simulator for the reference scenario of
              each building type. To simulate your own building and settings,{" "}
              <a href={`${REPO}#full-app`} target="_blank" rel="noreferrer">
                run the full app
              </a>
              .
            </p>
          )}
        </details>
      </div>

      <details className="border-b px-4 py-2.5 md:hidden" style={{ borderColor: "var(--border)" }}>
        <summary className="flex cursor-pointer list-none items-center justify-between gap-3">
          <span>
            <span className="muted text-sm">
              Step {index + 1} of {STEPS.length}
            </span>
            <span className="block font-bold">{current.action}</span>
          </span>
          <span className="text-sm underline underline-offset-2">All steps</span>
        </summary>
        <nav aria-label="Steps" className="pt-4">
          {stepList}
        </nav>
      </details>

      <div className="mx-auto grid w-full max-w-[1180px] flex-1 gap-10 px-4 pt-6 pb-14 sm:px-6 md:grid-cols-[250px_minmax(0,1fr)] md:pt-10">
        <nav aria-label="Steps" className="hidden md:block">
          <div className="sticky top-6">{stepList}</div>
        </nav>
        <main id="main" className="screen min-w-0">
          <p className="caption mb-1 hidden md:block">
            Step {index + 1} of {STEPS.length}
          </p>
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
          <div className="mt-10 flex flex-wrap items-center justify-between gap-4 border-t pt-5" style={{ borderColor: "var(--border)" }}>
            {prev ? (
              <button className="flex items-center gap-1.5 underline underline-offset-4" onClick={() => setStep(prev.id)}>
                <Icon name="arrow" size={15} className="rotate-180" />
                Back: {prev.action.toLowerCase()}
              </button>
            ) : (
              <span />
            )}
            {showContinue && next && (
              <button className="btn" onClick={() => setStep(next.id)}>
                Continue: {next.action.toLowerCase()}
                <Icon name="arrow" size={16} />
              </button>
            )}
          </div>
        </main>
      </div>

      <footer style={{ background: "var(--surface-2)", borderTop: "1px solid var(--border)" }}>
        <div className="mx-auto flex max-w-[1180px] flex-col gap-4 px-4 py-8 text-sm sm:flex-row sm:items-start sm:justify-between sm:px-6">
          <div className="max-w-md space-y-1">
            <p className="flex items-center gap-2 font-bold">
              <Logo size={22} /> TailSafe
            </p>
            <p className="secondary">
              Evacuation stress-testing for high-rise buildings: how bad the worst nights are, why, and what fixes
              them. All residents are synthetic.
            </p>
          </div>
          <ul className="flex flex-wrap gap-x-5 gap-y-2">
            <li>
              <a href={`${REPO}#readme`} target="_blank" rel="noreferrer">
                About
              </a>
            </li>
            <li>
              <a href={`${REPO}/blob/main/docs/TailSafe-User-Guide.pdf`} target="_blank" rel="noreferrer">
                User guide (PDF)
              </a>
            </li>
            <li>
              <a href={`${REPO}/blob/main/docs/assumptions.md`} target="_blank" rel="noreferrer">
                Assumptions
              </a>
            </li>
            <li>
              <a href={`${REPO}/blob/main/docs/validation.md`} target="_blank" rel="noreferrer">
                Validation
              </a>
            </li>
            <li>
              <a href={REPO} target="_blank" rel="noreferrer">
                Source code
              </a>
            </li>
          </ul>
        </div>
      </footer>
    </div>
  );
}
