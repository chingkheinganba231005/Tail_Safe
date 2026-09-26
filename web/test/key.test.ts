import { describe, expect, it } from "vitest";
import { canon, requestKey } from "../src/static/key";

// Reference hashes computed by tailsafe.api.static_site.request_key (Python).
const CASES: [string, string, unknown, string][] = [
  ["GET", "/api/health", undefined, "6946f38b29590789"],
  [
    "POST",
    "/api/stress",
    {
      building_id: "abc",
      spec: {
        name: "reference",
        share_65_plus: 0.22,
        fire_level: 14,
        stair_blockages: [{ stair: "A", time: { dist: "constant", value: 240.0, min: null } }],
        hazard: {},
        note: "Stair A → G/F ₉₅",
      },
      runs: 300,
      seed: 0,
    },
    "37311fbae03474f0",
  ],
  [
    "POST",
    "/api/briefing",
    { building_id: "abc", stress: { x: 1 }, bottlenecks: null, optimization: { a: 1 }, llm: false },
    "967d0cd1bf6aec77",
  ],
  [
    "POST",
    "/api/briefing/pdf",
    { markdown: "# Title\n- 219.2 min", distributions: { Baseline: [1.5, 2.0] } },
    "e4da5465ff86ff67",
  ],
];

describe("browser-version request keys", () => {
  it("match the Python recorder", () => {
    for (const [method, path, body, hash] of CASES) expect(requestKey(method, path, body)).toBe(hash);
  });
  it("drop undefined fields and sort keys", () => {
    expect(canon({ b: 1, a: undefined, c: [1.5, null] })).toBe('{"b":1,"c":[1.5,null]}');
    expect(canon({ z: 2, a: 1 })).toBe('{"a":1,"z":2}');
    expect(canon(-0)).toBe("0");
  });
});
