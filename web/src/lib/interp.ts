// Positions between two recorded frames of a micro replay.

export interface Dot {
  id: number;
  x: number;
  y: number;
  state: number;
}

/** Frame index k with times[k] <= t < times[k + 1] (clamped). */
export function frameAt(times: number[], t: number): number {
  let lo = 0;
  let hi = times.length - 1;
  if (hi < 0 || t <= times[0]) return 0;
  if (t >= times[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid;
    else hi = mid;
  }
  return lo;
}

/**
 * People on the floor at time t: linear interpolation between the frames
 * either side for people present in both, else the earlier frame's position.
 */
export function dotsAt(
  times: number[],
  frames: [number, number, number, number][][],
  t: number,
  scale: number,
): Dot[] {
  if (frames.length === 0) return [];
  const k = frameAt(times, t);
  const a = frames[k];
  const b = frames[Math.min(k + 1, frames.length - 1)];
  const span = (times[Math.min(k + 1, times.length - 1)] ?? 0) - times[k];
  const f = span > 0 ? Math.min(1, Math.max(0, (t - times[k]) / span)) : 0;
  const next = new Map<number, [number, number, number, number]>();
  for (const row of b) next.set(row[0], row);
  return a.map(([id, x, y, state]) => {
    const n = next.get(id);
    if (!n || f === 0) return { id, x: x * scale, y: y * scale, state };
    return { id, x: (x + (n[1] - x) * f) * scale, y: (y + (n[2] - y) * f) * scale, state };
  });
}
