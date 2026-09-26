import { useMemo } from "react";
import { Legend } from "../Legend";
import { useTooltip } from "../Tooltip";
import { useWidth } from "../hooks";
import { histogram, ticks } from "../../lib/stats";

export interface HistSeries {
  name: string;
  values: number[]; // already in display units (minutes)
  color: string; // CSS colour, e.g. "var(--series-1)"
}

export interface Marker {
  x: number;
  label: string;
  strong?: boolean; // solid primary-ink line (CVaR); others dashed secondary
}

interface Props {
  series: HistSeries[];
  markers?: Marker[];
  xLabel: string;
  height?: number;
  domain?: [number, number];
  bins?: number;
}

const M = { top: 34, right: 16, bottom: 38, left: 44 };

/**
 * Distribution of scenario outcomes. One series: thin bars. Two series: 2px
 * step outlines on shared bins, with a legend. Markers (mean, P95, CVaR) are
 * labelled in ink, never in the series colour.
 */
export function Histogram({ series, markers = [], xLabel, height = 260, domain, bins = 30 }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const tip = useTooltip();
  const data = useMemo(() => histogram(series.map((s) => s.values), bins), [series, bins]);
  if (data.length === 0) return <div ref={ref} className="muted text-sm">No data.</div>;

  const iw = width - M.left - M.right;
  const ih = height - M.top - M.bottom;
  const lo = domain ? Math.min(domain[0], data[0].x0) : data[0].x0;
  const hi = domain ? Math.max(domain[1], data[data.length - 1].x1) : data[data.length - 1].x1;
  const maxCount = Math.max(1, ...data.flatMap((b) => b.counts));
  const x = (v: number) => ((v - lo) / (hi - lo || 1)) * iw;
  const y = (c: number) => ih - (c / maxCount) * ih;
  const xt = ticks(lo, hi, Math.max(3, Math.floor(iw / 80)));
  const yt = ticks(0, maxCount, 4).filter((t) => Number.isInteger(t));
  const totals = series.map((s) => s.values.filter(Number.isFinite).length || 1);
  const single = series.length === 1;

  // Stagger marker labels that would collide.
  const placed: { x: number; row: number; m: Marker }[] = [];
  [...markers]
    .sort((a, b) => a.x - b.x)
    .forEach((m) => {
      const px = x(m.x);
      let row = 0;
      while (placed.some((p) => p.row === row && Math.abs(p.x - px) < 92)) row += 1;
      placed.push({ x: px, row, m });
    });

  return (
    <div ref={ref} className="w-full">
      {!single && <Legend items={series.map((s) => ({ label: s.name, color: s.color }))} />}
      <svg width={width} height={height} role="img" aria-label={`Histogram of ${xLabel}`}>
        <g transform={`translate(${M.left},${M.top})`}>
          {yt.map((t) => (
            <g key={t} transform={`translate(0,${y(t)})`}>
              <line x2={iw} stroke="var(--grid)" strokeWidth={1} />
              <text x={-8} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--text-muted)">
                {t}
              </text>
            </g>
          ))}
          {single &&
            data.map((b) => {
              const c = b.counts[0];
              if (!c) return null;
              const bx = x(b.x0) + 1;
              const bw = Math.max(1, Math.min(24, x(b.x1) - x(b.x0) - 2));
              const by = y(c);
              const r = Math.min(4, bw / 2, ih - by);
              return (
                <path
                  key={b.x0}
                  d={`M${bx},${ih}V${by + r}Q${bx},${by} ${bx + r},${by}H${bx + bw - r}Q${bx + bw},${by} ${bx + bw},${by + r}V${ih}Z`}
                  fill={series[0].color}
                />
              );
            })}
          {!single &&
            series.map((s, k) => {
              let d = `M${x(data[0].x0)},${ih}`;
              for (const b of data) d += `V${y(b.counts[k])}H${x(b.x1)}`;
              d += `V${ih}`;
              return (
                <path key={s.name} d={d} fill="none" stroke={s.color} strokeWidth={2} />
              );
            })}
          <line y1={ih} y2={ih} x2={iw} stroke="var(--axis)" />
          {xt.map((t) => (
            <text
              key={t}
              x={x(t)}
              y={ih + 16}
              textAnchor="middle"
              fontSize={11}
              fill="var(--text-muted)"
            >
              {t}
            </text>
          ))}
          <text x={iw / 2} y={ih + 32} textAnchor="middle" fontSize={12} fill="var(--text-secondary)">
            {xLabel}
          </text>
          {placed.map(({ x: px, row, m }) => (
            <g key={m.label} pointerEvents="none">
              <line
                x1={px}
                x2={px}
                y1={-4 - row * 14}
                y2={ih}
                stroke={m.strong ? "var(--text-primary)" : "var(--text-secondary)"}
                strokeWidth={m.strong ? 2 : 1.5}
                strokeDasharray={m.strong ? undefined : "4 3"}
              />
              <text
                x={px > iw - 90 ? px - 4 : px + 4}
                textAnchor={px > iw - 90 ? "end" : "start"}
                y={-8 - row * 14}
                fontSize={11}
                fontWeight={m.strong ? 600 : 400}
                fill={m.strong ? "var(--text-primary)" : "var(--text-secondary)"}
              >
                {m.label}
              </text>
            </g>
          ))}
          {data.map((b) => (
            <rect
              key={`hit-${b.x0}`}
              x={x(b.x0)}
              width={Math.max(1, x(b.x1) - x(b.x0))}
              y={0}
              height={ih}
              fill="transparent"
              onMouseMove={(e) =>
                tip.show(e, {
                  title: `${b.x0.toFixed(0)}–${b.x1.toFixed(0)} min`,
                  rows: series.map((s, k) => ({
                    label: single ? "scenarios" : s.name,
                    value: `${b.counts[k]} (${((100 * b.counts[k]) / totals[k]).toFixed(1)}%)`,
                    swatch: single ? undefined : s.color,
                  })),
                })
              }
              onMouseLeave={tip.hide}
            />
          ))}
        </g>
      </svg>
      {tip.node}
    </div>
  );
}

/** Table rows for a histogram's table view. */
export function histogramTable(series: HistSeries[], bins = 30) {
  const data = histogram(series.map((s) => s.values), bins);
  return {
    columns: ["Minutes", ...series.map((s) => `${s.name} (scenarios)`)],
    rows: data.map((b) => [`${b.x0.toFixed(0)}–${b.x1.toFixed(0)}`, ...b.counts]),
  };
}
