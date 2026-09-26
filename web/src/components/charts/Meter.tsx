import type { Estimate } from "../../types";

interface Props {
  label: string;
  estimate: Estimate;
  description?: string;
}

/**
 * A probability as a meter on [0, 1] with its confidence interval. The fill
 * uses the series colour; the number is in ink.
 */
export function Meter({ label, estimate, description }: Props) {
  const { value, lo, hi } = estimate;
  return (
    <div className="card p-4">
      <div className="secondary text-sm">{label}</div>
      <div className="tabular text-4xl font-semibold">{value.toFixed(2)}</div>
      <div className="muted tabular text-xs">
        95% CI {lo.toFixed(2)} – {hi.toFixed(2)}
      </div>
      <svg
        width="100%"
        height="22"
        viewBox="0 0 100 22"
        preserveAspectRatio="none"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={value}
        aria-label={label}
        className="mt-2"
      >
        <rect x="0" y="6" width="100" height="10" rx="4" fill="var(--grid)" />
        <rect x="0" y="6" width={100 * value} height="10" rx="4" fill="var(--series-1)" />
        <rect
          x={100 * lo}
          y="3"
          width={Math.max(0.4, 100 * (hi - lo))}
          height="16"
          fill="none"
          stroke="var(--text-primary)"
          strokeWidth="0.6"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
      <div className="muted flex justify-between text-xs">
        <span>0</span>
        <span>1</span>
      </div>
      {description && <p className="secondary mt-2 text-xs">{description}</p>}
    </div>
  );
}
