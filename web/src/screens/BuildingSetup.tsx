import { useEffect, useMemo, useState } from "react";
import { get, post } from "../api";
import { PlanView } from "../components/PlanView";
import { StatTile } from "../components/StatTile";
import { levelLabel, titleCase } from "../lib/format";
import type { Building, BuildingView, Template } from "../types";

interface Props {
  building: BuildingView | null;
  onBuilding: (b: BuildingView) => void;
  onConfirm: () => void;
}

/** Screen 1: pick a template (or upload JSON), review the plan, correct widths, confirm. */
export function BuildingSetup({ building, onBuilding, onConfirm }: Props) {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [source, setSource] = useState<"template" | "upload">("template");
  const [name, setName] = useState("cruciform");
  const [options, setOptions] = useState<Record<string, number | boolean>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    get<Template[]>("/api/templates")
      .then((ts) => {
        setTemplates(ts);
        const t = ts.find((x) => x.name === "cruciform") ?? ts[0];
        if (t) {
          setName(t.name);
          setOptions(t.options);
        }
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const generate = async () => {
    setBusy(true);
    setError(null);
    try {
      onBuilding(await post<BuildingView>("/api/buildings", { template: name, options }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const upload = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      const json = JSON.parse(await file.text()) as unknown;
      onBuilding(await post<BuildingView>("/api/buildings/upload", json));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <section className="card p-4">
        <h2 className="text-lg font-semibold">Building</h2>
        <p className="secondary mb-3 text-sm">
          Start from a procedural Hong Kong typology, or upload a building JSON that follows{" "}
          <code>schemas/building.schema.json</code>.
        </p>
        <div role="group" aria-label="Source" className="mb-3 flex gap-1">
          <button className="btn-ghost text-sm" aria-pressed={source === "template"} onClick={() => setSource("template")}>
            Template
          </button>
          <button className="btn-ghost text-sm" aria-pressed={source === "upload"} onClick={() => setSource("upload")}>
            Upload JSON
          </button>
        </div>
        {source === "template" ? (
          <div className="space-y-3">
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              {templates.map((t) => (
                <button
                  key={t.name}
                  className="btn-ghost p-3 text-left"
                  aria-pressed={name === t.name}
                  onClick={() => {
                    setName(t.name);
                    setOptions(t.options);
                  }}
                >
                  <div className="font-semibold">{titleCase(t.name)}</div>
                  <div className="secondary text-xs">{t.description}</div>
                </button>
              ))}
            </div>
            <div className="flex flex-wrap items-end gap-3">
              {Object.entries(options).map(([k, v]) => (
                <label key={k} className="text-sm">
                  <span className="secondary block text-xs">{titleCase(k)}</span>
                  {typeof v === "boolean" ? (
                    <input
                      type="checkbox"
                      checked={v}
                      onChange={(e) => setOptions({ ...options, [k]: e.target.checked })}
                    />
                  ) : (
                    <input
                      type="number"
                      className="w-24"
                      value={v}
                      step={Number.isInteger(v) ? 1 : 0.5}
                      onChange={(e) => setOptions({ ...options, [k]: Number(e.target.value) })}
                    />
                  )}
                </label>
              ))}
              <button className="btn" disabled={busy || !name} onClick={generate}>
                {busy ? "Generating…" : "Generate building"}
              </button>
            </div>
          </div>
        ) : (
          <input
            type="file"
            accept="application/json,.json"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void upload(f);
            }}
          />
        )}
        {error && (
          <p className="mt-2 text-sm" role="alert">
            <span aria-hidden style={{ color: "var(--critical)" }}>
              ⛔{" "}
            </span>
            {error}
          </p>
        )}
      </section>
      {building && <Review view={building} onBuilding={onBuilding} onConfirm={onConfirm} />}
    </div>
  );
}

function Review({
  view,
  onBuilding,
  onConfirm,
}: {
  view: BuildingView;
  onBuilding: (b: BuildingView) => void;
  onConfirm: () => void;
}) {
  const b = view.building;
  const s = view.summary;
  const typical = b.levels.find((l) => l.kind === "typical")?.index ?? b.levels[Math.min(1, b.levels.length - 1)].index;
  const [level, setLevel] = useState(typical);
  useEffect(() => setLevel(typical), [typical]);
  return (
    <>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <StatTile label="Storeys" value={s.storeys} />
        <StatTile label="Flats / rooms" value={s.units} />
        <StatTile label="Staircases" value={s.stairs.length} sub={s.stairs.join(", ")} />
        <StatTile label="Lifts" value={s.lifts.length} sub={s.lifts.join(", ")} />
        <StatTile label="Height" value={`${s.height_m.toFixed(0)} m`} />
      </div>
      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        <section className="card p-4">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="font-semibold">{b.name}</h3>
              <p className="secondary text-sm">{b.description}</p>
            </div>
            <label className="text-sm">
              <span className="secondary mr-2">Floor</span>
              <select value={level} onChange={(e) => setLevel(Number(e.target.value))}>
                {b.levels.map((l) => (
                  <option key={l.index} value={l.index}>
                    {levelLabel(l.index)} {l.kind !== "typical" ? `(${l.kind})` : ""}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <PlanView building={b} level={level} />
          <p className="muted mt-2 text-xs">
            Grey: circulation and stair landings (labelled). Dark lines: doors. Green dots: final exits.
          </p>
        </section>
        <Corrections view={view} onBuilding={onBuilding} onConfirm={onConfirm} />
      </div>
    </>
  );
}

/** Correction editor: stair and exit widths (re-validated by the API). */
function Corrections({
  view,
  onBuilding,
  onConfirm,
}: {
  view: BuildingView;
  onBuilding: (b: BuildingView) => void;
  onConfirm: () => void;
}) {
  const b = view.building;
  const exitIds = useMemo(() => new Set(b.nodes.filter((n) => n.type === "exit").map((n) => n.id)), [b]);
  const exitEdges = b.edges.filter((e) => exitIds.has(e.source) || exitIds.has(e.target));
  const [stairW, setStairW] = useState<Record<string, number>>({});
  const [exitW, setExitW] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setStairW(Object.fromEntries(b.stairs.map((s) => [s.id, s.clear_width])));
    setExitW(Object.fromEntries(exitEdges.map((e) => [e.id, e.width])));
  }, [view.id]);

  const changed =
    b.stairs.some((s) => stairW[s.id] !== undefined && stairW[s.id] !== s.clear_width) ||
    exitEdges.some((e) => exitW[e.id] !== undefined && exitW[e.id] !== e.width);

  const apply = async () => {
    setBusy(true);
    setError(null);
    const next: Building = {
      ...b,
      stairs: b.stairs.map((s) => ({ ...s, clear_width: stairW[s.id] ?? s.clear_width })),
      edges: b.edges.map((e) => {
        if (e.kind === "stair" && e.stair && stairW[e.stair] !== undefined) {
          return { ...e, width: stairW[e.stair] };
        }
        return exitW[e.id] !== undefined ? { ...e, width: exitW[e.id] } : e;
      }),
    };
    try {
      onBuilding(await post<BuildingView>("/api/buildings/upload", next));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="card space-y-3 p-4">
      <div>
        <h3 className="font-semibold">Check and correct</h3>
        <p className="secondary text-xs">Clear widths in metres. Changes are validated by the server.</p>
      </div>
      <fieldset className="space-y-1">
        <legend className="secondary text-xs">Staircases</legend>
        {b.stairs.map((s) => (
          <label key={s.id} className="flex items-center justify-between gap-2 text-sm">
            <span>
              {s.label} <span className="muted text-xs">({s.kind})</span>
            </span>
            <input
              type="number"
              step={0.05}
              min={0.6}
              className="w-20"
              value={stairW[s.id] ?? s.clear_width}
              onChange={(e) => setStairW({ ...stairW, [s.id]: Number(e.target.value) })}
            />
          </label>
        ))}
      </fieldset>
      <fieldset className="space-y-1">
        <legend className="secondary text-xs">Final exits</legend>
        {exitEdges.map((e) => (
          <label key={e.id} className="flex items-center justify-between gap-2 text-sm">
            <span className="truncate">{e.label ?? e.id}</span>
            <input
              type="number"
              step={0.05}
              min={0.5}
              className="w-20"
              value={exitW[e.id] ?? e.width}
              onChange={(ev) => setExitW({ ...exitW, [e.id]: Number(ev.target.value) })}
            />
          </label>
        ))}
      </fieldset>
      {error && (
        <p className="text-sm" role="alert">
          <span aria-hidden style={{ color: "var(--critical)" }}>
            ⛔{" "}
          </span>
          {error}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        <button className="btn-ghost text-sm" disabled={!changed || busy} onClick={apply}>
          Apply corrections
        </button>
        <button className="btn" onClick={onConfirm}>
          Confirm building →
        </button>
      </div>
    </section>
  );
}
