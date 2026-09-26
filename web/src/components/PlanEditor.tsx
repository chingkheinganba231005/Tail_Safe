import { useRef, useState, type MouseEvent } from "react";
import { get, post } from "../api";
import type { BuildingView, PlanDetection, PlanDoor, PlanRoomType, PlanScale } from "../types";
import { Callout, Icon } from "./Icon";

interface Props {
  onBuilding: (b: BuildingView) => void;
}

type Tool = "select" | "scale" | "door";
type Selection = { kind: "room" | "door"; id: string } | null;

const TYPES: { id: PlanRoomType; label: string; code: string; fill: string }[] = [
  { id: "unit", label: "Flat / room", code: "FLAT", fill: "transparent" },
  { id: "corridor", label: "Corridor", code: "COR", fill: "color-mix(in srgb, var(--text-secondary) 22%, transparent)" },
  { id: "lobby", label: "Lobby", code: "LOBBY", fill: "color-mix(in srgb, var(--text-secondary) 38%, transparent)" },
  { id: "stair", label: "Staircase", code: "STAIR", fill: "color-mix(in srgb, var(--series-1) 35%, transparent)" },
  { id: "refuge", label: "Refuge area", code: "REFUGE", fill: "color-mix(in srgb, var(--series-3) 35%, transparent)" },
  { id: "void", label: "Not walkable", code: "—", fill: "color-mix(in srgb, var(--text-muted) 15%, transparent)" },
];
const typeInfo = (t: PlanRoomType) => TYPES.find((x) => x.id === t)!;

/**
 * Floor-plan correction editor: set the scale, let the reader find rooms,
 * doorways and stairs, correct them, then stack the floor into a building.
 */
export function PlanEditor({ onBuilding }: Props) {
  const [image, setImage] = useState<{ src: string; w: number; h: number } | null>(null);
  const [tool, setTool] = useState<Tool>("select");
  const [pending, setPending] = useState<number[] | null>(null);
  const [scaleLine, setScaleLine] = useState<number[] | null>(null); // x1, y1, x2, y2
  const [metres, setMetres] = useState(10);
  const [det, setDet] = useState<PlanDetection | null>(null);
  const [sel, setSel] = useState<Selection>(null);
  const [storeys, setStoreys] = useState(20);
  const [name, setName] = useState("Building from floor plan");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const doorSeq = useRef(1);

  const reset = (src: string, w: number, h: number) => {
    setImage({ src, w, h });
    setDet(null);
    setSel(null);
    setScaleLine(null);
    setPending(null);
    setTool("scale");
  };

  const loadFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const src = String(reader.result);
      if (file.type === "application/pdf") {
        // The server renders PDFs; show a placeholder size until detection.
        reset(src, 1000, 1000);
        return;
      }
      const img = new Image();
      img.onload = () => reset(src, img.naturalWidth, img.naturalHeight);
      img.src = src;
    };
    reader.readAsDataURL(file);
  };

  const loadSample = async () => {
    setError(null);
    const s = await get<{ image: string; width: number; height: number; m_per_px: number }>(
      "/api/vision/sample?template=cruciform&level=5",
    );
    reset(s.image, s.width, s.height);
    // The sample's scale is known: a 10 m reference line along the top.
    setScaleLine([20, 20, 20 + 10 / s.m_per_px, 20]);
    setMetres(10);
    setTool("select");
  };

  const toImage = (e: MouseEvent<SVGSVGElement>): number[] => {
    const box = svgRef.current!.getBoundingClientRect();
    return [((e.clientX - box.left) / box.width) * image!.w, ((e.clientY - box.top) / box.height) * image!.h];
  };

  const snap = (a: number[], b: number[]): number[] =>
    Math.abs(b[0] - a[0]) >= Math.abs(b[1] - a[1]) ? [b[0], a[1]] : [a[0], b[1]];

  const onClick = (e: MouseEvent<SVGSVGElement>) => {
    if (!image || (tool !== "scale" && tool !== "door")) return;
    const p = toImage(e);
    if (!pending) {
      setPending(p);
      return;
    }
    const q = snap(pending, p);
    if (tool === "scale") {
      setScaleLine([pending[0], pending[1], q[0], q[1]]);
      setTool("select");
    } else if (det) {
      const width = Math.hypot(q[0] - pending[0], q[1] - pending[1]) * det.m_per_px;
      const door: PlanDoor = { id: `U${doorSeq.current++}`, a: pending, b: q, width_m: Number(width.toFixed(2)), rooms: [] };
      setDet({ ...det, doors: [...det.doors, door] });
      setSel({ kind: "door", id: door.id });
    }
    setPending(null);
  };

  const detect = async () => {
    if (!image) return;
    setBusy(true);
    setError(null);
    try {
      const scale: PlanScale | null = scaleLine
        ? { x1: scaleLine[0], y1: scaleLine[1], x2: scaleLine[2], y2: scaleLine[3], metres }
        : null;
      const out = await post<{ detection: PlanDetection; preview?: string }>("/api/vision/detect", {
        image: image.src,
        scale,
      });
      setDet(out.detection);
      setImage({ src: out.preview ?? image.src, w: out.detection.width, h: out.detection.height });
      setSel(null);
      setTool("select");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const build = async () => {
    if (!det) return;
    setBusy(true);
    setError(null);
    try {
      onBuilding(await post<BuildingView>("/api/vision/build", { detection: det, storeys, name }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const room = sel?.kind === "room" ? det?.rooms.find((r) => r.id === sel.id) : undefined;
  const door = sel?.kind === "door" ? det?.doors.find((d) => d.id === sel.id) : undefined;
  const counts = det
    ? TYPES.map((t) => ({ ...t, n: det.rooms.filter((r) => r.type === t.id).length })).filter((t) => t.n > 0)
    : [];
  const stroke = Math.max(1, (image?.w ?? 1000) / 500);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <input
          type="file"
          accept="image/png,image/jpeg,application/pdf"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) loadFile(f);
          }}
        />
        <button className="btn-ghost text-sm" onClick={loadSample}>
          Try a sample plan
        </button>
        <span className="muted text-xs">PNG, JPEG or PDF (first page). Plans stay on this machine's server.</span>
      </div>
      {image && (
        <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
          <div>
            <div role="group" aria-label="Tool" className="mb-2 flex flex-wrap gap-1">
              {(
                [
                  ["select", "Select"],
                  ["scale", "Reference line"],
                  ["door", "Add doorway"],
                ] as const
              ).map(([t, label]) => (
                <button
                  key={t}
                  className="btn-ghost text-sm"
                  aria-pressed={tool === t}
                  disabled={t === "door" && !det}
                  onClick={() => {
                    setTool(t);
                    setPending(null);
                  }}
                >
                  {label}
                </button>
              ))}
              <span className="muted self-center text-xs">
                {tool === "scale" && (pending ? "Click the other end of a known length." : "Click one end of a known length.")}
                {tool === "door" && (pending ? "Click the other side of the doorway." : "Click one side of the doorway.")}
                {tool === "select" && det && "Click a room to change its type, or a doorway to delete it."}
              </span>
            </div>
            <div className="relative w-full" style={{ aspectRatio: `${image.w} / ${image.h}` }}>
              <img src={image.src} alt="Uploaded floor plan" className="absolute inset-0 h-full w-full" />
              <svg
                ref={svgRef}
                viewBox={`0 0 ${image.w} ${image.h}`}
                className="absolute inset-0 h-full w-full"
                style={{ cursor: tool === "select" ? "default" : "crosshair" }}
                onClick={onClick}
                role="img"
                aria-label="Detected rooms and doorways"
              >
                {det?.rooms.map((r) => {
                  const t = typeInfo(r.type);
                  const selected = sel?.kind === "room" && sel.id === r.id;
                  const [x0, y0, x1, y1] = r.rects[0];
                  return (
                    <g
                      key={r.id}
                      onClick={(e) => {
                        if (tool !== "select") return;
                        e.stopPropagation();
                        setSel({ kind: "room", id: r.id });
                      }}
                      style={{ cursor: tool === "select" ? "pointer" : undefined }}
                    >
                      {r.rects.map(([a, b, c, d], k) => (
                        <rect
                          key={k}
                          x={a}
                          y={b}
                          width={c - a}
                          height={d - b}
                          fill={t.fill}
                          stroke={selected ? "var(--text-primary)" : "var(--series-1)"}
                          strokeWidth={selected ? stroke * 3 : stroke * 0.8}
                          strokeDasharray={r.type === "void" ? `${stroke * 4} ${stroke * 3}` : undefined}
                        />
                      ))}
                      <text
                        x={(x0 + x1) / 2}
                        y={(y0 + y1) / 2}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fontSize={Math.max(9, image.w / 90)}
                        fontWeight={600}
                        fill="var(--text-primary)"
                        stroke="var(--surface-1)"
                        strokeWidth={stroke * 2}
                        paintOrder="stroke"
                        pointerEvents="none"
                      >
                        {t.code}
                      </text>
                    </g>
                  );
                })}
                {det?.doors.map((d) => {
                  const selected = sel?.kind === "door" && sel.id === d.id;
                  const outside = d.rooms.includes("outside");
                  return (
                    <g key={d.id}>
                      <line
                        x1={d.a[0]}
                        y1={d.a[1]}
                        x2={d.b[0]}
                        y2={d.b[1]}
                        stroke={selected ? "var(--text-primary)" : "var(--series-2)"}
                        strokeWidth={stroke * (selected ? 5 : 3.5)}
                        strokeLinecap="round"
                      />
                      <line
                        x1={d.a[0]}
                        y1={d.a[1]}
                        x2={d.b[0]}
                        y2={d.b[1]}
                        stroke="transparent"
                        strokeWidth={stroke * 12}
                        style={{ cursor: tool === "select" ? "pointer" : undefined }}
                        onClick={(e) => {
                          if (tool !== "select") return;
                          e.stopPropagation();
                          setSel({ kind: "door", id: d.id });
                        }}
                      />
                      {outside && (
                        <text
                          x={(d.a[0] + d.b[0]) / 2}
                          y={(d.a[1] + d.b[1]) / 2 - stroke * 6}
                          textAnchor="middle"
                          fontSize={Math.max(9, image.w / 100)}
                          fill="var(--text-primary)"
                          stroke="var(--surface-1)"
                          strokeWidth={stroke * 2}
                          paintOrder="stroke"
                          pointerEvents="none"
                        >
                          EXIT
                        </text>
                      )}
                    </g>
                  );
                })}
                {scaleLine && (
                  <g pointerEvents="none">
                    <line
                      x1={scaleLine[0]}
                      y1={scaleLine[1]}
                      x2={scaleLine[2]}
                      y2={scaleLine[3]}
                      stroke="var(--text-primary)"
                      strokeWidth={stroke * 2}
                      strokeDasharray={`${stroke * 6} ${stroke * 3}`}
                    />
                    <text
                      x={(scaleLine[0] + scaleLine[2]) / 2}
                      y={(scaleLine[1] + scaleLine[3]) / 2 - stroke * 5}
                      textAnchor="middle"
                      fontSize={Math.max(10, image.w / 80)}
                      fill="var(--text-primary)"
                      stroke="var(--surface-1)"
                      strokeWidth={stroke * 2}
                      paintOrder="stroke"
                    >
                      {metres} m
                    </text>
                  </g>
                )}
                {pending && <circle cx={pending[0]} cy={pending[1]} r={stroke * 4} fill="var(--text-primary)" />}
              </svg>
            </div>
            {det && (
              <ul className="secondary mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label="Legend">
                {counts.map((t) => (
                  <li key={t.id} className="flex items-center gap-1.5">
                    <span
                      aria-hidden
                      style={{
                        width: 12,
                        height: 12,
                        display: "inline-block",
                        background: t.fill,
                        border: "1px solid var(--series-1)",
                      }}
                    />
                    {t.code} {t.label.toLowerCase()} ({t.n})
                  </li>
                ))}
                <li className="flex items-center gap-1.5">
                  <span aria-hidden style={{ width: 14, height: 4, display: "inline-block", background: "var(--series-2)" }} />
                  doorway ({det.doors.length})
                </li>
              </ul>
            )}
          </div>
          <aside className="space-y-3 text-sm">
            <section className="card space-y-2 p-3">
              <h4 className="font-semibold">1. Scale</h4>
              {scaleLine ? (
                <label className="flex items-center gap-2">
                  Reference line is
                  <input
                    type="number"
                    min={0.5}
                    step={0.5}
                    className="w-20"
                    value={metres}
                    onChange={(e) => setMetres(Number(e.target.value))}
                  />
                  m long
                </label>
              ) : (
                <p className="secondary text-xs">
                  Draw a line over a dimension you know (e.g. a 10 m grid line). Without one, the scale is guessed
                  from wall thickness.
                </p>
              )}
              <button className="btn w-full" disabled={busy} onClick={detect}>
                {busy && !det ? "Reading plan…" : det ? "2. Read again" : "2. Read plan"}
              </button>
              {det && (
                <p className="muted text-xs tabular">
                  {(1 / det.m_per_px).toFixed(1)} px per metre ({det.scale_source === "reference" ? "from your line" : "guessed from walls"})
                </p>
              )}
            </section>
            {det && (
              <section className="card space-y-2 p-3">
                <h4 className="font-semibold">3. Check and correct</h4>
                {det.warnings.map((w) => (
                  <p key={w} className="flex gap-1.5 text-xs">
                    <Icon name="alert" size={14} className="mt-px flex-none" style={{ color: "var(--warning)" }} />
                    <span className="secondary">{w}</span>
                  </p>
                ))}
                {room && (
                  <div className="space-y-1">
                    <div className="secondary text-xs">
                      Room {room.id} · {room.area_m2.toFixed(0)} m² · {room.doors} doorway(s)
                    </div>
                    <select
                      value={room.type}
                      onChange={(e) =>
                        setDet({
                          ...det,
                          rooms: det.rooms.map((r) =>
                            r.id === room.id ? { ...r, type: e.target.value as PlanRoomType } : r,
                          ),
                        })
                      }
                      aria-label="Room type"
                    >
                      {TYPES.map((t) => (
                        <option key={t.id} value={t.id}>
                          {t.label}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
                {door && (
                  <div className="space-y-1">
                    <div className="secondary text-xs">
                      Doorway {door.id} · {door.width_m.toFixed(2)} m ·{" "}
                      {door.rooms.length ? door.rooms.join(" ↔ ") : "rooms found when building"}
                    </div>
                    <button
                      className="btn-ghost text-sm"
                      onClick={() => {
                        setDet({ ...det, doors: det.doors.filter((d) => d.id !== door.id) });
                        setSel(null);
                      }}
                    >
                      Delete doorway
                    </button>
                  </div>
                )}
                {!room && !door && <p className="muted text-xs">Nothing selected.</p>}
              </section>
            )}
            {det && (
              <section className="card space-y-2 p-3">
                <h4 className="font-semibold">4. Build</h4>
                <label className="block">
                  <span className="secondary block text-xs">Storeys (including G/F)</span>
                  <input type="number" min={1} max={80} className="w-24" value={storeys} onChange={(e) => setStoreys(Number(e.target.value))} />
                </label>
                <label className="block">
                  <span className="secondary block text-xs">Name</span>
                  <input className="w-full" value={name} onChange={(e) => setName(e.target.value)} />
                </label>
                <button className="btn w-full" disabled={busy} onClick={build}>
                  {busy ? "Building…" : "Build building →"}
                </button>
                <p className="muted text-xs">
                  The plan is used for every floor. Doorways to the outside become exits on G/F.
                </p>
              </section>
            )}
            {error && (
              <Callout tone="critical">{error}</Callout>
            )}
          </aside>
        </div>
      )}
    </div>
  );
}
