import { describe, expect, it } from "vitest";
import { BLUE, ramp, smokeLevel } from "../src/lib/color";
import { clock, levelLabel, minutes, signedMin } from "../src/lib/format";
import { dotsAt, frameAt } from "../src/lib/interp";
import { extent, histogram, quantile, ticks } from "../src/lib/stats";

describe("format", () => {
  it("labels floors the Hong Kong way", () => {
    expect(levelLabel(0)).toBe("G/F");
    expect(levelLabel(14)).toBe("14/F");
    expect(levelLabel(-1)).toBe("B1/F");
  });
  it("formats minutes and signs", () => {
    expect(minutes(90)).toBe("1.5 min");
    expect(minutes(null)).toBe("–");
    expect(signedMin(-120)).toBe("−2.0");
    expect(signedMin(60)).toBe("+1.0");
    expect(clock(125)).toBe("2:05");
  });
});

describe("stats", () => {
  it("computes quantiles like numpy (linear)", () => {
    expect(quantile([1, 2, 3, 4], 0.5)).toBeCloseTo(2.5);
    expect(quantile([5], 0.95)).toBe(5);
    expect(Number.isNaN(quantile([], 0.5))).toBe(true);
  });
  it("ignores non-finite values in extents", () => {
    expect(extent([3, NaN, -1, Infinity])).toEqual([-1, 3]);
  });
  it("makes nice ticks", () => {
    expect(ticks(0, 100, 5)).toEqual([0, 20, 40, 60, 80, 100]);
  });
  it("bins several series on shared edges and keeps every value", () => {
    const a = [1, 2, 3, 10, 20];
    const b = [5, 6, 7];
    const bins = histogram([a, b], 10);
    expect(bins.length).toBeGreaterThan(1);
    const total = (k: number) => bins.reduce((s, x) => s + x.counts[k], 0);
    expect(total(0)).toBe(a.length);
    expect(total(1)).toBe(b.length);
    for (let i = 1; i < bins.length; i++) expect(bins[i].x0).toBeCloseTo(bins[i - 1].x1);
    expect(bins[0].x0).toBeLessThanOrEqual(1);
    expect(bins.at(-1)!.x1).toBeGreaterThan(20);
  });
  it("handles a constant series", () => {
    const bins = histogram([[4, 4, 4]]);
    expect(bins.reduce((s, x) => s + x.counts[0], 0)).toBe(3);
  });
});

describe("colour", () => {
  it("sequential ramp runs light to dark, reversed in dark mode", () => {
    expect(ramp(BLUE, 0)).toBe(BLUE[0]);
    expect(ramp(BLUE, 1)).toBe(BLUE.at(-1));
    expect(ramp(BLUE, 0, "dark")).toBe(BLUE.at(-1));
    expect(ramp(BLUE, 2)).toBe(BLUE.at(-1)); // clamped
  });
  it("maps visibility to smoke intensity", () => {
    expect(smokeLevel(30)).toBe(0);
    expect(smokeLevel(2)).toBe(1);
    expect(smokeLevel(null)).toBe(0);
    expect(smokeLevel(11)).toBeCloseTo(0.5);
  });
});


describe("replay interpolation", () => {
  const times = [0, 2, 4];
  const frames: [number, number, number, number][][] = [
    [
      [1, 0, 0, 1],
      [2, 10, 10, 0],
    ],
    [[1, 20, 0, 1]],
    [],
  ];
  it("finds the frame before t", () => {
    expect(frameAt(times, -1)).toBe(0);
    expect(frameAt(times, 3)).toBe(1);
    expect(frameAt(times, 9)).toBe(2);
  });
  it("interpolates people present in both frames and holds the others", () => {
    const dots = dotsAt(times, frames, 1, 0.1);
    const one = dots.find((d) => d.id === 1)!;
    expect(one.x).toBeCloseTo(1.0);
    const two = dots.find((d) => d.id === 2)!;
    expect(two.x).toBeCloseTo(1.0);
    expect(dotsAt(times, frames, 4, 0.1)).toEqual([]);
  });
});
