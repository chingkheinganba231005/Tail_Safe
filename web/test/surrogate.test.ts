import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { type GraphFile, type ModelFile, loadNetwork, predictRaw, predict } from "../src/static/surrogate";
import type { ScenarioSpec } from "../src/types";

// Fixture written by the Python model (tiny random network, small slab block):
// the browser port must reproduce its predictions.
const fx = (name: string) => new URL(`./fixtures/${name}`, import.meta.url);
const model = JSON.parse(readFileSync(fx("surrogate-model.json"), "utf8")) as ModelFile;
const buf = readFileSync(fx("surrogate-weights.bin"));
const weights = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const graph = JSON.parse(readFileSync(fx("surrogate-graph.json"), "utf8")) as GraphFile;
const expected = JSON.parse(readFileSync(fx("surrogate-expected.json"), "utf8")) as {
  specs: ScenarioSpec[];
  quantiles: number[][][];
  edges: number[][];
  notes: string[][];
  odd_spec: ScenarioSpec;
  odd_notes: string[];
};

const close = (a: number, b: number) => Math.abs(a - b) <= 1e-3 * Math.max(1, Math.abs(b));

describe("in-browser surrogate", () => {
  const net = loadNetwork(model, weights);
  it("reproduces the Python model's outcome quantiles and edge queueing", () => {
    expected.specs.forEach((spec, k) => {
      const { quantiles, edges } = predictRaw(net, graph, spec);
      quantiles.forEach((row, li) => row.forEach((v, j) => expect(close(v, expected.quantiles[k][li][j])).toBe(true)));
      edges.forEach((v, i) => expect(close(v, expected.edges[k][i])).toBe(true));
    });
  });
  it("answers like the API", () => {
    const out = predict(net, graph, expected.specs[1], "disclaimer", 3);
    const t = out.losses.total_time;
    expect(t.p50 <= t.p75 && t.p75 <= t.p90 && t.p90 <= t.p95 && t.p95 <= t.cvar95).toBe(true);
    expect(out.edges.length).toBe(3);
    expect(out.edges[0].label).not.toBe(out.edges[0].edge);
    expected.specs.forEach((spec, k) => expect(predict(net, graph, spec, "").coverage_notes).toEqual(expected.notes[k]));
    expect(predict(net, graph, expected.odd_spec, "").coverage_notes).toEqual(expected.odd_notes);
  });
});
