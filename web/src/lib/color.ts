// Sequential colour ramps (one hue, light → dark on light surfaces; reversed on
// dark surfaces so "near zero" recedes into the background).

export type Mode = "light" | "dark";

// Reference blue ramp, steps 100 → 700.
export const BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
// Orange ramp for the second sequential context (smoke), one hue, monotone lightness.
export const ORANGE = ["#fbe1d6", "#f5b393", "#f08a5d", "#eb6834", "#b8481d", "#7a2a0b"];

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16)) as [number, number, number];
}

function rgbToHex([r, g, b]: [number, number, number]): string {
  return `#${[r, g, b].map((v) => Math.round(v).toString(16).padStart(2, "0")).join("")}`;
}

export function mix(a: string, b: string, t: number): string {
  const x = hexToRgb(a);
  const y = hexToRgb(b);
  return rgbToHex([0, 1, 2].map((i) => x[i] + (y[i] - x[i]) * t) as [number, number, number]);
}

/** Colour for t in [0, 1] on a ramp; in dark mode low values are dark. */
export function ramp(stops: string[], t: number, mode: Mode = "light"): string {
  const s = mode === "dark" ? [...stops].reverse() : stops;
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (s.length - 1);
  const i = Math.min(s.length - 2, Math.floor(x));
  return mix(s[i], s[i + 1], x - i);
}

/** Visibility (m) → smoke intensity in [0, 1]: 20 m or more is clear, 2 m is dense. */
export function smokeLevel(visibility: number | null | undefined): number {
  if (visibility === null || visibility === undefined || !Number.isFinite(visibility)) return 0;
  return Math.min(1, Math.max(0, (20 - visibility) / 18));
}
