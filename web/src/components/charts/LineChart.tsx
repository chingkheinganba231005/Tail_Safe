import { useState } from "react";
import { extent, ticks } from "../../lib/stats";
import { useWidth } from "../hooks";
import { Legend } from "../Legend";
import { useTooltip } from "../Tooltip";

export interface LineSeries {
  name: string;
  x: number[];
  y: number[];
  color: string;
}

interface Props {
  series: LineSeries[];
  xLabel: string;
  yLabel: string;
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  cursor?: number | null; // external cursor (e.g. replay time), in x units
  height?: number;
}

const M = { top: 12, right: 16, bottom: 38, left: 52 };

/** Lines over time with a crosshair and a tooltip listing every series. */
export function LineChart({
  series,
  xLabel,
  yLabel,
  formatX = (v) => v.toFixed(0),
  formatY = (v) => v.toFixed(0),
  cursor,
  height = 220,
}: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const tip = useTooltip();
  const [hover, setHover] = useState<number | null>(null);
  const iw = width - M.left - M.right;
  const ih = height - M.top - M.bottom;
  const [x0, x1] = extent(series.flatMap((s) => s.x));
  const [, y1] = extent(series.flatMap((s) => s.y));
  const ymax = Math.max(1, y1);
  const x = (v: number) => ((v - x0) / (x1 - x0 || 1)) * iw;
  const y = (v: number) => ih - (v / ymax) * ih;
  const xt = ticks(x0, x1, Math.max(3, Math.floor(iw / 80)));
  const yt = ticks(0, ymax, 4);
  const at = (s: LineSeries, xv: number) => {
    let k = 0;
    while (k + 1 < s.x.length && s.x[k + 1] <= xv) k++;
    return s.y[k];
  };
  const shown = hover ?? cursor ?? null;

  return (
    <div ref={ref} className="w-full">
      {series.length > 1 && (
        <Legend items={series.map((s) => ({ label: s.name, color: s.color }))} />
      )}
      <svg width={width} height={height} role="img" aria-label={`${yLabel} over ${xLabel}`}>
        <g transform={`translate(${M.left},${M.top})`}>
          {yt.map((t) => (
            <g key={t} transform={`translate(0,${y(t)})`}>
              <line x2={iw} stroke="var(--grid)" />
              <text x={-8} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--text-muted)">
                {formatY(t)}
              </text>
            </g>
          ))}
          {xt.map((t) => (
            <text key={t} x={x(t)} y={ih + 16} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
              {formatX(t)}
            </text>
          ))}
          <text x={iw / 2} y={ih + 32} textAnchor="middle" fontSize={12} fill="var(--text-secondary)">
            {xLabel}
          </text>
          <text
            transform={`translate(${-40},${ih / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={12}
            fill="var(--text-secondary)"
          >
            {yLabel}
          </text>
          <line y1={ih} y2={ih} x2={iw} stroke="var(--axis)" />
          {series.map((s) => (
            <path
              key={s.name}
              d={s.x.map((xv, i) => `${i ? "L" : "M"}${x(xv)},${y(s.y[i])}`).join("")}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
              strokeLinejoin="round"
            />
          ))}
          {shown !== null && shown >= x0 && shown <= x1 && (
            <g pointerEvents="none">
              <line x1={x(shown)} x2={x(shown)} y2={ih} stroke="var(--text-secondary)" strokeWidth={1} />
              {series.map((s) => (
                <circle
                  key={s.name}
                  cx={x(shown)}
                  cy={y(at(s, shown))}
                  r={4}
                  fill={s.color}
                  stroke="var(--surface-1)"
                  strokeWidth={2}
                />
              ))}
            </g>
          )}
          <rect
            width={iw}
            height={ih}
            fill="transparent"
            onMouseMove={(e) => {
              const box = (e.currentTarget as SVGRectElement).getBoundingClientRect();
              const xv = x0 + ((e.clientX - box.left) / iw) * (x1 - x0);
              setHover(xv);
              tip.show(e, {
                title: formatX(xv),
                rows: series.map((s) => ({
                  label: s.name,
                  value: formatY(at(s, xv)),
                  swatch: series.length > 1 ? s.color : undefined,
                })),
              });
            }}
            onMouseLeave={() => {
              setHover(null);
              tip.hide();
            }}
          />
        </g>
      </svg>
      {tip.node}
    </div>
  );
}
