import { levelLabel } from "../../lib/format";
import { ticks } from "../../lib/stats";
import { useWidth } from "../hooks";
import { useTooltip } from "../Tooltip";

export interface FloorRow {
  level: number;
  value: number;
  lo?: number;
  hi?: number;
}

interface Props {
  rows: FloorRow[];
  format: (v: number) => string;
  valueLabel: string;
  color?: string;
  max?: number;
  height?: number;
  highlight?: Set<number>;
  onSelect?: (level: number) => void;
}

const M = { top: 8, right: 16, bottom: 30, left: 44 };

/**
 * One horizontal bar per floor, highest floor on top (reads like the building).
 * Optional confidence whiskers. Single hue.
 */
export function FloorChart({
  rows,
  format,
  valueLabel,
  color = "var(--series-1)",
  max,
  height,
  highlight,
  onSelect,
}: Props) {
  const [ref, width] = useWidth<HTMLDivElement>(320);
  const tip = useTooltip();
  const sorted = [...rows].sort((a, b) => b.level - a.level);
  const h = height ?? Math.min(640, Math.max(160, sorted.length * 14 + M.top + M.bottom));
  const iw = width - M.left - M.right;
  const ih = h - M.top - M.bottom;
  const band = ih / Math.max(1, sorted.length);
  const bar = Math.max(2, Math.min(24, band - 2));
  const vmax = max ?? Math.max(1e-9, ...sorted.map((r) => r.hi ?? r.value));
  const x = (v: number) => (Math.max(0, v) / vmax) * iw;
  const xt = ticks(0, vmax, Math.max(2, Math.floor(iw / 70)));
  const every = Math.max(1, Math.ceil(12 / band));

  return (
    <div ref={ref} className="w-full">
      <svg width={width} height={h} role="img" aria-label={`${valueLabel} by floor`}>
        <g transform={`translate(${M.left},${M.top})`}>
          {xt.map((t) => (
            <g key={t} transform={`translate(${x(t)},0)`}>
              <line y2={ih} stroke="var(--grid)" />
              <text y={ih + 14} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
                {format(t)}
              </text>
            </g>
          ))}
          <text x={iw / 2} y={ih + 27} textAnchor="middle" fontSize={11} fill="var(--text-secondary)">
            {valueLabel}
          </text>
          {sorted.map((r, i) => {
            const cy = i * band + band / 2;
            const w = x(r.value);
            const sel = highlight?.has(r.level);
            return (
              <g key={r.level} transform={`translate(0,${cy})`}>
                {i % every === 0 && (
                  <text x={-6} dy="0.32em" textAnchor="end" fontSize={10} fill="var(--text-muted)">
                    {levelLabel(r.level)}
                  </text>
                )}
                {w > 0 && (
                  <path
                    d={`M0,${-bar / 2}H${Math.max(0, w - Math.min(4, w))}Q${w},${-bar / 2} ${w},${-bar / 2 + Math.min(4, bar / 2)}V${bar / 2 - Math.min(4, bar / 2)}Q${w},${bar / 2} ${Math.max(0, w - Math.min(4, w))},${bar / 2}H0Z`}
                    fill={color}
                    stroke={sel ? "var(--text-primary)" : undefined}
                    strokeWidth={sel ? 2 : undefined}
                  />
                )}
                {r.lo !== undefined && r.hi !== undefined && r.hi > r.lo && (
                  <g stroke="var(--text-secondary)" strokeWidth={1}>
                    <line x1={x(r.lo)} x2={x(r.hi)} />
                    <line x1={x(r.lo)} x2={x(r.lo)} y1={-3} y2={3} />
                    <line x1={x(r.hi)} x2={x(r.hi)} y1={-3} y2={3} />
                  </g>
                )}
                <rect
                  x={-M.left}
                  y={-band / 2}
                  width={iw + M.left}
                  height={band}
                  fill="transparent"
                  style={{ cursor: onSelect ? "pointer" : undefined }}
                  onClick={() => onSelect?.(r.level)}
                  onMouseMove={(e) =>
                    tip.show(e, {
                      title: levelLabel(r.level),
                      rows: [
                        { label: valueLabel, value: format(r.value) },
                        ...(r.lo !== undefined && r.hi !== undefined
                          ? [{ label: "95% CI", value: `${format(r.lo)} – ${format(r.hi)}` }]
                          : []),
                      ],
                    })
                  }
                  onMouseLeave={tip.hide}
                />
              </g>
            );
          })}
          <line y2={ih} stroke="var(--axis)" />
        </g>
      </svg>
      {tip.node}
    </div>
  );
}
