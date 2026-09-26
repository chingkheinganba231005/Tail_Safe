// Small numeric helpers for charts: extents, quantiles, histograms, ticks.

export function extent(values: number[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return lo <= hi ? [lo, hi] : [0, 1];
}

/** Empirical quantile with linear interpolation (like numpy's default). */
export function quantile(values: number[], q: number): number {
  const xs = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (xs.length === 0) return NaN;
  const pos = (xs.length - 1) * q;
  const i = Math.floor(pos);
  const f = pos - i;
  return i + 1 < xs.length ? xs[i] * (1 - f) + xs[i + 1] * f : xs[i];
}

export function mean(values: number[]): number {
  const xs = values.filter(Number.isFinite);
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
}

/** Nice round step for about `count` intervals over [lo, hi]. */
export function niceStep(lo: number, hi: number, count = 5): number {
  const span = Math.max(hi - lo, 1e-9);
  const raw = span / Math.max(count, 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return nice * mag;
}

export function ticks(lo: number, hi: number, count = 5): number[] {
  const step = niceStep(lo, hi, count);
  const start = Math.ceil(lo / step - 1e-9) * step;
  const out: number[] = [];
  for (let v = start; v <= hi + step * 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

export interface Bin {
  x0: number;
  x1: number;
  counts: number[]; // one count per series
}

/**
 * Histogram of one or more series on shared bins (so they can be compared).
 * Bin edges are nice round numbers covering all series.
 */
export function histogram(series: number[][], targetBins = 30): Bin[] {
  const all = series.flat().filter(Number.isFinite);
  if (all.length === 0) return [];
  let [lo, hi] = extent(all);
  if (hi === lo) {
    lo -= 0.5;
    hi += 0.5;
  }
  const step = niceStep(lo, hi, targetBins);
  const start = Math.floor(lo / step) * step;
  const n = Math.max(1, Math.ceil((hi - start) / step + 1e-9));
  const bins: Bin[] = Array.from({ length: n }, (_, i) => ({
    x0: start + i * step,
    x1: start + (i + 1) * step,
    counts: series.map(() => 0),
  }));
  series.forEach((xs, s) => {
    for (const x of xs) {
      if (!Number.isFinite(x)) continue;
      const k = Math.min(n - 1, Math.floor((x - start) / step));
      bins[k].counts[s] += 1;
    }
  });
  return bins;
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}
