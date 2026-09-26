import { Legend } from "../Legend";
import { useTooltip } from "../Tooltip";
import { useWidth } from "../hooks";
import { ticks } from "../../lib/stats";

export interface RangeSeries {
  name: string;
  color: string;
  p50: number;
  p75: number;
  p90: number;
  p95: number;
  cvar: number;
}

export interface RangeRow {
  key: string;
  label: string;
  series: RangeSeries[];
}

interface Props {
  rows: RangeRow[];
  unit: string;
}

const LEFT = 150;
const RIGHT = 70;
const BAR = 8;

/**
 * Distributions as range bars: P50 to P95 with ticks at P75 and P90 and a
 * diamond at CVaR95 (the average of the worst 5%). One axis for all rows.
 */
export function RangeChart({ rows, unit }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const tip = useTooltip();
  const nSeries = Math.max(1, ...rows.map((r) => r.series.length));
  const rowH = 22 + nSeries * 16;
  const h = rows.length * rowH + 44;
  const iw = Math.max(120, width - LEFT - RIGHT);
  const hi = Math.max(1, ...rows.flatMap((r) => r.series.map((s) => s.cvar))) * 1.08;
  const x = (v: number) => (v / hi) * iw;
  const xt = ticks(0, hi, Math.max(3, Math.floor(iw / 80)));
  const names = rows[0]?.series.map((s) => ({ label: s.name, color: s.color })) ?? [];
  return (
    <div ref={ref} className="w-full">
      {names.length > 1 && <Legend items={names} />}
      <svg width={width} height={h} role="img" aria-label="Outcome ranges">
        <g transform={`translate(${LEFT},8)`}>
          {xt.map((t) => (
            <g key={t} transform={`translate(${x(t)},0)`}>
              <line y2={rows.length * rowH} stroke="var(--grid)" />
              <text y={rows.length * rowH + 14} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
                {t}
              </text>
            </g>
          ))}
          <text x={iw / 2} y={rows.length * rowH + 28} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">
            {unit}
          </text>
          {rows.map((r, i) => (
            <g key={r.key} transform={`translate(0,${i * rowH})`}>
              <text x={-10} y={rowH / 2} dy="0.32em" textAnchor="end" fontSize={12} fill="var(--text-primary)">
                {r.label}
              </text>
              {r.series.map((s, k) => {
                const cy = 12 + k * 16 + BAR / 2;
                const x0 = x(s.p50);
                const w = Math.max(2, x(s.p95) - x0);
                return (
                  <g
                    key={s.name}
                    onMouseMove={(e) =>
                      tip.show(e, {
                        title: `${r.label} — ${s.name}`,
                        rows: [
                          { label: "median", value: `${s.p50.toFixed(1)} ${unit}` },
                          { label: "P90", value: `${s.p90.toFixed(1)} ${unit}` },
                          { label: "P95", value: `${s.p95.toFixed(1)} ${unit}` },
                          { label: "CVaR₉₅ (worst 5%)", value: `${s.cvar.toFixed(1)} ${unit}` },
                        ],
                      })
                    }
                    onMouseLeave={tip.hide}
                  >
                    <rect x={x0} y={cy - BAR / 2} width={w} height={BAR} rx={4} fill={s.color} />
                    {[s.p75, s.p90].map((v) => (
                      <line
                        key={v}
                        x1={x(v)}
                        x2={x(v)}
                        y1={cy - BAR / 2}
                        y2={cy + BAR / 2}
                        stroke="var(--surface-1)"
                        strokeWidth={2}
                      />
                    ))}
                    <path
                      d={`M${x(s.cvar)},${cy - 6} l6,6 l-6,6 l-6,-6 Z`}
                      fill="var(--text-primary)"
                      stroke="var(--surface-1)"
                      strokeWidth={1.5}
                    />
                    <text x={x(s.cvar) + 10} y={cy} dy="0.32em" fontSize={11} fill="var(--text-secondary)" className="tabular">
                      {s.cvar.toFixed(0)}
                    </text>
                    <rect x={x0 - 4} y={cy - 8} width={x(s.cvar) - x0 + 12} height={16} fill="transparent" />
                  </g>
                );
              })}
            </g>
          ))}
        </g>
      </svg>
      {tip.node}
    </div>
  );
}
