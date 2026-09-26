// Number and label formatting. Times arrive in seconds and are shown in minutes.

export const LOSS_LABELS: Record<string, string> = {
  total_time: "Time until everyone is out",
  self_evacuation_time: "Time until the last self-evacuee is out",
  p95_occupant_time: "Time for 95% of occupants to get out",
};

export const LOSS_SHORT: Record<string, string> = {
  total_time: "Everyone out",
  self_evacuation_time: "Last self-evacuee",
  p95_occupant_time: "95% of occupants",
};

/** Hong Kong storey label: 0 → "G/F", 14 → "14/F", -1 → "B1/F". */
export function levelLabel(level: number): string {
  if (level === 0) return "G/F";
  if (level < 0) return `B${-level}/F`;
  return `${level}/F`;
}

export function minutes(seconds: number | null | undefined, digits = 1): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "–";
  return `${(seconds / 60).toFixed(digits)} min`;
}

/** Minutes without the unit (for tables whose header carries it). */
export function min(seconds: number | null | undefined, digits = 1): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "–";
  return (seconds / 60).toFixed(digits);
}

export function signedMin(seconds: number, digits = 1): string {
  const v = seconds / 60;
  const s = v.toFixed(digits);
  return v > 0 ? `+${s}` : v < 0 ? `−${s.slice(1)}` : s;
}

export function pct(p: number | null | undefined, digits = 0): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "–";
  return `${(100 * p).toFixed(digits)}%`;
}

export function prob(p: number | null | undefined, digits = 2): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "–";
  return p.toFixed(digits);
}

export function num(x: number | null | undefined, digits = 0): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  return x.toLocaleString("en-GB", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** "12:30" style clock for a replay scrubber. */
export function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}
