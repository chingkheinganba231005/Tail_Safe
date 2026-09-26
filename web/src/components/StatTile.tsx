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
    <div className="card p-4 sm:p-5" style={emphasis ? { borderTop: "2px solid var(--accent)" } : undefined}>
      <div className="secondary text-[0.8rem] leading-snug">{label}</div>
      <div
        className={`tabular mt-1 font-bold whitespace-nowrap ${emphasis ? "text-[clamp(1.6rem,1.2rem+1vw,2.2rem)] leading-none" : "text-[1.6rem] leading-tight"}`}
        style={{ letterSpacing: "-0.02em" }}
      >
        {value}
      </div>
      {sub && <div className="muted tabular mt-1.5 text-xs">{sub}</div>}
    </div>
  );
}
