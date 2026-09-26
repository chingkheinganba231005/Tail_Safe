import { useMemo } from "react";
import { ORANGE } from "../lib/color";
import { levelLabel } from "../lib/format";
import type { Building, EdgeDef, NodeDef, Point } from "../types";
import { useTooltip } from "./Tooltip";
import { useWidth } from "./hooks";

interface Props {
  building: Building;
  level: number;
  highlightEdges?: Set<string>;
  highlightStair?: string | null;
  fireNode?: string | null;
  height?: number;
}

const CIRCULATION = new Set(["corridor", "lobby", "lift_lobby", "protected_lobby"]);

function nodeFill(n: NodeDef): string {
  if (n.type === "unit") return "var(--surface-1)";
  if (n.type === "stair_landing") return "color-mix(in srgb, var(--text-secondary) 22%, var(--surface-1))";
  if (n.type === "refuge") return "color-mix(in srgb, var(--series-3) 18%, var(--surface-1))";
  if (CIRCULATION.has(n.type)) return "color-mix(in srgb, var(--text-secondary) 9%, var(--surface-1))";
  return "var(--surface-1)";
}

/** Typical-floor plan of one level: rooms, circulation, stairs, doors, exits. */
export function PlanView({ building, level, highlightEdges, highlightStair, fireNode, height = 420 }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const tip = useTooltip();
  const nodes = useMemo(() => building.nodes.filter((n) => n.level === level), [building, level]);
  const byId = useMemo(() => new Map(building.nodes.map((n) => [n.id, n])), [building]);
  const lv = building.levels.find((l) => l.index === level);
  const edges = useMemo(
    () =>
      building.edges.filter((e) => {
        const s = byId.get(e.source);
        const t = byId.get(e.target);
        return (s && s.level === level) || (t && t.level === level);
      }),
    [building, byId, level],
  );

  const pts: Point[] = [
    ...nodes.flatMap((n) => n.polygon ?? [[n.x, n.y] as Point]),
    ...(lv?.features ?? []).flatMap((f) => f.polygon),
  ];
  if (pts.length === 0) return <div ref={ref} className="muted text-sm">Nothing on {levelLabel(level)}.</div>;
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const pad = 2;
  const minX = Math.min(...xs) - pad;
  const maxX = Math.max(...xs) + pad;
  const minY = Math.min(...ys) - pad;
  const maxY = Math.max(...ys) + pad;
  const scale = Math.min(width / (maxX - minX), height / (maxY - minY));
  const w = (maxX - minX) * scale;
  const h = (maxY - minY) * scale;
  const X = (x: number) => (x - minX) * scale;
  const Y = (y: number) => (maxY - y) * scale; // plan y points up
  const poly = (p: Point[]) => p.map(([x, y]) => `${X(x)},${Y(y)}`).join(" ");

  const edgeLine = (e: EdgeDef): [Point, Point] | null => {
    if (e.opening && e.opening.length === 2) return [e.opening[0], e.opening[1]];
    const s = byId.get(e.source);
    const t = byId.get(e.target);
    if (!s || !t) return null;
    return [
      [s.x, s.y],
      [t.x, t.y],
    ];
  };

  const hi = (e: EdgeDef) =>
    highlightEdges?.has(e.id) ||
    (highlightStair != null && e.kind === "stair" && e.stair === highlightStair);

  return (
    <div ref={ref} className="w-full">
      <svg
        width={w}
        height={h}
        role="img"
        aria-label={`Plan of ${levelLabel(level)} in ${building.name}`}
        style={{ display: "block", margin: "0 auto" }}
      >
        {(lv?.features ?? []).map((f, i) => (
          <polygon
            key={`f${i}`}
            points={poly(f.polygon)}
            fill={f.kind === "lift_shaft" ? "color-mix(in srgb, var(--text-secondary) 35%, var(--surface-1))" : "var(--grid)"}
            stroke="var(--axis)"
            strokeWidth={0.5}
          />
        ))}
        {nodes
          .filter((n) => n.polygon && n.type !== "exit")
          .map((n) => {
            const stairHi = highlightStair != null && n.stair === highlightStair;
            return (
              <polygon
                key={n.id}
                points={poly(n.polygon!)}
                fill={n.id === fireNode ? ORANGE[4] : nodeFill(n)}
                stroke={stairHi ? "var(--text-primary)" : "var(--axis)"}
                strokeWidth={stairHi ? 2.5 : 0.75}
                onMouseMove={(e) =>
                  tip.show(e, {
                    title: n.label ?? n.id,
                    rows: [
                      { label: "type", value: n.type.replace(/_/g, " ") },
                      ...(n.area ? [{ label: "area", value: `${n.area.toFixed(0)} m²` }] : []),
                    ],
                  })
                }
                onMouseLeave={tip.hide}
              />
            );
          })}
        {edges
          .filter((e) => e.kind === "door" || hi(e))
          .map((e) => {
            const seg = edgeLine(e);
            if (!seg) return null;
            const strong = hi(e);
            return (
              <g key={e.id}>
                {strong && (
                  <line
                    x1={X(seg[0][0])}
                    y1={Y(seg[0][1])}
                    x2={X(seg[1][0])}
                    y2={Y(seg[1][1])}
                    stroke="var(--surface-1)"
                    strokeWidth={9}
                    strokeLinecap="round"
                  />
                )}
                <line
                  x1={X(seg[0][0])}
                  y1={Y(seg[0][1])}
                  x2={X(seg[1][0])}
                  y2={Y(seg[1][1])}
                  stroke={strong ? "var(--text-primary)" : "var(--text-secondary)"}
                  strokeWidth={strong ? 5 : 2}
                  strokeLinecap="round"
                  onMouseMove={(ev) =>
                    tip.show(ev, {
                      title: e.label ?? e.id,
                      rows: [{ label: "clear width", value: `${e.width.toFixed(2)} m` }],
                    })
                  }
                  onMouseLeave={tip.hide}
                />
              </g>
            );
          })}
        {nodes
          .filter((n) => n.type === "exit")
          .map((n) => (
            <g key={n.id} transform={`translate(${X(n.x)},${Y(n.y)})`}>
              <circle r={6} fill="var(--good)" stroke="var(--surface-1)" strokeWidth={2} />
              <text y={-9} textAnchor="middle" fontSize={10} fill="var(--text-secondary)">
                EXIT
              </text>
            </g>
          ))}
        {nodes
          .filter((n) => n.type === "stair_landing" && n.stair)
          .map((n) => (
            <text
              key={`t${n.id}`}
              x={X(n.x)}
              y={Y(n.y)}
              dy="0.32em"
              textAnchor="middle"
              fontSize={11}
              fontWeight={600}
              fill="var(--text-primary)"
              pointerEvents="none"
            >
              {n.stair}
            </text>
          ))}
      </svg>
      {tip.node}
    </div>
  );
}
