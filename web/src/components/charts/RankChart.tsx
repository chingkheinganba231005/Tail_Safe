import { useTooltip } from "../Tooltip";
import { useWidth } from "../hooks";
import { ticks } from "../../lib/stats";

export interface RankRow {
  key: string;
  label: string;
  value: number;
  lo?: number;
  hi?: number;
  note?: string;
}

interface Props {
  rows: RankRow[];
  format: (v: number) => string;
  valueLabel: string;
  selected?: string | null;
  onSelect?: (key: string) => void;
  color?: string;
  labelWidth?: number;
}

const ROW = 30;
const M = { top: 6, right: 64, bottom: 30 };

/**
 * Ranked horizontal bars around zero (negative = improvement), one hue, with
 * confidence whiskers and the value at the bar end in ink. Rows are clickable.
 */
export function RankChart({
  rows,
  format,
  valueLabel,
  selected,
  onSelect,
  color = "var(--series-1)",
  labelWidth = 220,
}: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const tip = useTooltip();
  const left = Math.min(labelWidth, width * 0.45);
  const iw = Math.max(80, width - left - M.right);
  const h = rows.length * ROW + M.top + M.bottom;
  const vals = rows.flatMap((r) => [r.value, r.lo ?? r.value, r.hi ?? r.value, 0]);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const span = hi - lo || 1;
  const x = (v: number) => ((v - lo) / span) * iw;
  const x0 = x(0);
  const xt = ticks(lo, hi, Math.max(2, Math.floor(iw / 80)));
  const bar = 16;
  const maxChars = Math.max(8, Math.floor((left - 12) / 7));

  return (
    <div ref={ref} className="w-full">
      <svg width={width} height={h} role="img" aria-label={valueLabel}>
        <g transform={`translate(${left},${M.top})`}>
          {xt.map((t) => (
            <g key={t} transform={`translate(${x(t)},0)`}>
              <line y2={rows.length * ROW} stroke="var(--grid)" />
              <text y={rows.length * ROW + 14} textAnchor="middle" fontSize={11} fill="var(--text-muted)">
                {format(t)}
              </text>
            </g>
          ))}
          <text
            x={iw / 2}
            y={rows.length * ROW + 27}
            textAnchor="middle"
            fontSize={11}
            fill="var(--text-secondary)"
          >
            {valueLabel}
          </text>
          {rows.map((r, i) => {
            const cy = i * ROW + ROW / 2;
            const a = Math.min(x0, x(r.value));
            const w = Math.abs(x(r.value) - x0);
            const sel = selected === r.key;
            const rad = Math.min(4, w / 2);
            // Rounded data end only (the end away from zero).
            const d =
              r.value < 0
                ? `M${a + w},${cy - bar / 2}H${a + rad}Q${a},${cy - bar / 2} ${a},${cy - bar / 2 + rad}V${cy + bar / 2 - rad}Q${a},${cy + bar / 2} ${a + rad},${cy + bar / 2}H${a + w}Z`
                : `M${a},${cy - bar / 2}H${a + w - rad}Q${a + w},${cy - bar / 2} ${a + w},${cy - bar / 2 + rad}V${cy + bar / 2 - rad}Q${a + w},${cy + bar / 2} ${a + w - rad},${cy + bar / 2}H${a}Z`;
            return (
              <g key={r.key}>
                {sel && (
                  <rect
                    x={-left}
                    y={i * ROW}
                    width={left + iw + M.right}
                    height={ROW}
                    fill="color-mix(in srgb, var(--series-1) 10%, transparent)"
                  />
                )}
                <text
                  x={-8}
                  y={cy}
                  dy="0.32em"
                  textAnchor="end"
                  fontSize={12}
                  fontWeight={sel ? 600 : 400}
                  fill="var(--text-primary)"
                >
                  {r.label.length > maxChars ? `${r.label.slice(0, maxChars - 1)}…` : r.label}
                </text>
                {w > 0.5 && <path d={d} fill={color} />}
                {r.lo !== undefined && r.hi !== undefined && r.hi > r.lo && (
                  <g stroke="var(--text-secondary)">
                    <line x1={x(r.lo)} x2={x(r.hi)} y1={cy} y2={cy} />
                    <line x1={x(r.lo)} x2={x(r.lo)} y1={cy - 4} y2={cy + 4} />
                    <line x1={x(r.hi)} x2={x(r.hi)} y1={cy - 4} y2={cy + 4} />
                  </g>
                )}
                <text
                  x={Math.max(x(r.value), x(r.hi ?? r.value), x0) + 6}
                  y={cy}
                  dy="0.32em"
                  fontSize={11}
                  fill="var(--text-secondary)"
                  className="tabular"
                >
                  {format(r.value)}
                </text>
                <rect
                  x={-left}
                  y={i * ROW}
                  width={left + iw + M.right}
                  height={ROW}
                  fill="transparent"
                  style={{ cursor: onSelect ? "pointer" : undefined }}
                  onClick={() => onSelect?.(r.key)}
                  onMouseMove={(e) =>
                    tip.show(e, {
                      title: r.label,
                      rows: [
                        { label: valueLabel, value: format(r.value) },
                        ...(r.lo !== undefined && r.hi !== undefined
                          ? [{ label: "95% CI", value: `${format(r.lo)} to ${format(r.hi)}` }]
                          : []),
                      ],
                      note: r.note,
                    })
                  }
                  onMouseLeave={tip.hide}
                />
              </g>
            );
          })}
          <line x1={x0} x2={x0} y2={rows.length * ROW} stroke="var(--axis)" />
        </g>
      </svg>
      {tip.node}
    </div>
  );
}
