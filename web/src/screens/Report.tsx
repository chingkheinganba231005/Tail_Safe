import { Fragment, type ReactNode, useState } from "react";
import type { StressRun } from "../App";
import { post, postBlob } from "../api";
import { Callout } from "../components/Icon";
import type { BottleneckResult, BuildingView, Briefing, OptimizeResult } from "../types";

interface Props {
  building: BuildingView;
  stress: StressRun;
  bottlenecks: BottleneckResult | null;
  optimization: OptimizeResult | null;
}

/** Inline **bold** only; everything else is plain text (no HTML is ever injected). */
function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? <strong key={i}>{part.slice(2, -2)}</strong> : <Fragment key={i}>{part}</Fragment>,
  );
}

/** The small Markdown subset the briefings use: #, ##, bullets, paragraphs. */
export function Markdown({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];
  const flush = () => {
    if (bullets.length) {
      blocks.push(
        <ul key={`ul${blocks.length}`} className="mb-3 list-disc space-y-1 pl-5">
          {bullets.map((b, i) => (
            <li key={i}>{inline(b)}</li>
          ))}
        </ul>,
      );
      bullets = [];
    }
  };
  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    if (line.startsWith("- ") || line.startsWith("* ")) {
      bullets.push(line.slice(2));
      continue;
    }
    flush();
    if (!line) continue;
    const k = `b${blocks.length}`;
    if (line.startsWith("# ")) blocks.push(<h2 key={k} className="mb-2 text-xl font-semibold">{inline(line.slice(2))}</h2>);
    else if (line.startsWith("## ")) blocks.push(<h3 key={k} className="mt-4 mb-1 font-semibold">{inline(line.slice(3))}</h3>);
    else blocks.push(<p key={k} className="mb-2">{inline(line)}</p>);
  }
  flush();
  return <div className="max-w-3xl text-sm leading-relaxed">{blocks}</div>;
}

/** Screen 9: a one-page briefing for the building manager, grounded in the results. */
export function Report({ building, stress, bottlenecks, optimization }: Props) {
  const [brief, setBrief] = useState<Briefing | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const write = async (llm: boolean | null) => {
    setBusy(true);
    setError(null);
    try {
      setBrief(
        await post<Briefing>("/api/briefing", {
          building_id: building.id,
          stress: stress.result,
          bottlenecks,
          optimization,
          llm,
        }),
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const pdf = async () => {
    if (!brief) return;
    const distributions: Record<string, number[]> = optimization
      ? { Baseline: optimization.before_after.total_time.before, "With plan": optimization.before_after.total_time.after }
      : { Baseline: stress.result.losses.total_time };
    let blob: Blob;
    try {
      blob = await postBlob("/api/briefing/pdf", { markdown: brief.markdown, distributions });
    } catch (e) {
      setError(`PDF export failed: ${(e as Error).message}`);
      return;
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "tailsafe-briefing.pdf";
    a.click();
    URL.revokeObjectURL(url);
  };

  const missing = [!bottlenecks && "bottlenecks", !optimization && "an optimised plan"].filter(Boolean);
  return (
    <div className="space-y-4">
      <section className="card space-y-3 p-4">
        <h2 className="page-title">Briefing</h2>
        <p className="secondary max-w-3xl text-sm">
          A one-page summary for the building manager, written only from the numbers computed here. When an LLM is
          configured on the server it drafts the text, and every number in its draft is checked against the results;
          a draft with any other number is rejected and the template briefing is shown instead.
        </p>
        {missing.length > 0 && (
          <p className="muted text-sm">
            Run {missing.join(" and ")} first for a fuller briefing (optional).
          </p>
        )}
        <div className="flex flex-wrap gap-2">
          <button className="btn" disabled={busy} onClick={() => write(null)}>
            {busy ? "Writing…" : brief ? "Write again" : "Write briefing"}
          </button>
          <button className="btn-ghost" disabled={busy} onClick={() => write(false)}>
            Template only
          </button>
          {brief && (
            <button className="btn-ghost" onClick={pdf}>
              Download PDF
            </button>
          )}
        </div>
      </section>
      {error && (
        <Callout tone="critical">{error}</Callout>
      )}
      {brief && (
        <>
          <Callout tone="good">
            {brief.source === "llm" ? "Drafted by the language model; " : "Written from the results; "}
            every number checked against them.
          </Callout>
          {brief.note && (
            <Callout tone="warning" role="note">
              <span className="secondary">{brief.note}</span>
            </Callout>
          )}
          <section className="card p-6">
            <Markdown text={brief.markdown} />
          </section>
        </>
      )}
    </div>
  );
}
