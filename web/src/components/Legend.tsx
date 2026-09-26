export interface LegendItem {
  label: string;
  color: string;
  dashed?: boolean;
}

/** Legend for charts with two or more series (text in ink; the swatch carries colour). */
export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <ul className="secondary mb-1 flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-label="Legend">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-1.5">
          <svg width="16" height="8" aria-hidden>
            <line
              x1="0"
              x2="16"
              y1="4"
              y2="4"
              stroke={it.color}
              strokeWidth="3"
              strokeDasharray={it.dashed ? "3 2" : undefined}
            />
          </svg>
          {it.label}
        </li>
      ))}
    </ul>
  );
}
