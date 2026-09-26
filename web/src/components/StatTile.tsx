import type { ReactNode } from "react";

interface Props {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  emphasis?: boolean;
}

/** A single headline number (a stat tile, not a chart). */
export function StatTile({ label, value, sub, emphasis }: Props) {
  return (
    <div className="card p-4">
      <div className="secondary text-sm">{label}</div>
      <div
        className={`tabular font-semibold ${emphasis ? "text-4xl" : "text-2xl"}`}
        style={{ letterSpacing: "-0.01em" }}
      >
        {value}
      </div>
      {sub && <div className="muted tabular text-xs">{sub}</div>}
    </div>
  );
}
