// The what-if network, run in the browser for the browser version.
//
// A direct port of tailsafe/surrogate/features.py (graph_features),
// tailsafe/surrogate/model.py (apply, for one graph) and
// tailsafe/surrogate/predictor.py (predict, coverage_notes). The weights
// and each building's graph features are exported by
// tailsafe/api/static_site.py; test/surrogate.test.ts checks this port
// against predictions made by the Python code.
import type { ScenarioSpec, SurrogatePrediction } from "../types";

const LOSS_NAMES = ["total_time", "self_evacuation_time", "p95_occupant_time"] as const;
const QUANTILE_KEYS = ["p50", "p75", "p90", "p95"] as const;
const N_OUT = 5;
const TIME_SCALE = 3600;
const N_TYPES = 8; // NODE_TYPES in features.py
const SLOTS = ["weekday_day", "weekday_night", "weekend_day", "weekend_night"];
const PRIORITY = ["top_down", "nearest", "bottom_up"];
const N_GLOBAL = 24;
const HAZARD_DEFAULTS: Record<string, unknown> = {
  fire_unit: null,
  growth: null,
  peak_hrr: null,
  door_open_probability: null,
  hold_open_stair_doors: false,
  held_open_doors: [],
  horizon: 7200,
  record_dt: 10,
};

export interface ModelFile {
  config: { hidden: number; layers: number };
  stats: Record<"node_mu" | "node_sd" | "edge_mu" | "edge_sd" | "glob_mu" | "glob_sd", number[]>;
  meta: { trained_on?: Record<string, number>; evaluation?: SurrogatePrediction["model"]["evaluation"] };
  weights: Record<string, { offset: number; shape: number[] }>;
  horizon_s: number;
  defaults: { share_65_plus: number; share_80_plus_of_65_plus: number; vacancy_rate: number };
  typologies: Record<string, Record<string, [number, number]>>;
  template_defaults: Record<string, Record<string, unknown>>;
  share_65_range: [number, number];
}

export interface GraphFile {
  node_level: number[];
  node_base: number[][];
  edge_src: number[];
  edge_dst: number[];
  edge_base: number[][];
  edge_ids: string[];
  edge_forward: boolean[];
  edge_stair: (string | null)[];
  landing_stair: (string | null)[];
  lift_lobby: boolean[];
  n_levels: number;
  n_units: number;
  occupants: number;
  care_share: number;
  n_lifts: number;
  n_ff_lifts: number;
  n_stairs: number;
  labels: Record<string, string>;
  metadata: { generator: string | null; args: Record<string, unknown> };
}

interface Layer {
  W: Float32Array;
  b: Float32Array;
  inDim: number;
  outDim: number;
}
type Mlp = Layer[];

export interface Network {
  file: ModelFile;
  mlp: (name: string) => Mlp;
}

export function loadNetwork(file: ModelFile, weights: ArrayBuffer): Network {
  const all = new Float32Array(weights);
  const tensor = (key: string) => {
    const w = file.weights[key];
    if (!w) throw new Error(`missing weight ${key}`);
    const size = w.shape.reduce((a, b) => a * b, 1);
    return { data: all.subarray(w.offset, w.offset + size), shape: w.shape };
  };
  const cache = new Map<string, Mlp>();
  return {
    file,
    mlp(name) {
      const hit = cache.get(name);
      if (hit) return hit;
      const layers: Mlp = [];
      for (let i = 0; file.weights[`${name}/${i}/W`]; i++) {
        const W = tensor(`${name}/${i}/W`);
        layers.push({ W: W.data, b: tensor(`${name}/${i}/b`).data, inDim: W.shape[0], outDim: W.shape[1] });
      }
      cache.set(name, layers);
      return layers;
    },
  };
}

const silu = (x: number) => x / (1 + Math.exp(-x));
const softplus = (x: number) => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));

/** Rows of X [rows × inDim] through an MLP (SiLU between layers). */
function runMlp(layers: Mlp, X: Float32Array, rows: number): Float32Array {
  let cur = X;
  layers.forEach((L, li) => {
    const out = new Float32Array(rows * L.outDim);
    const last = li === layers.length - 1;
    for (let r = 0; r < rows; r++) {
      const xo = r * L.inDim;
      const oo = r * L.outDim;
      for (let j = 0; j < L.outDim; j++) out[oo + j] = L.b[j];
      for (let i = 0; i < L.inDim; i++) {
        const xi = cur[xo + i];
        if (xi === 0) continue;
        const wo = i * L.outDim;
        for (let j = 0; j < L.outDim; j++) out[oo + j] += xi * L.W[wo + j];
      }
      if (!last) for (let j = 0; j < L.outDim; j++) out[oo + j] = silu(out[oo + j]);
    }
    cur = out;
  });
  return cur;
}

function medianTime(t: ScenarioSpec["stair_blockages"][number]["time"]): number {
  if (t.value != null) return t.value;
  if (t.dist === "uniform") return ((t.min ?? 0) + (t.max ?? 0)) / 2;
  if (t.dist === "lognormal") return t.median ?? 0;
  const m = t.mean ?? 0;
  return Math.min(Math.max(m, t.min ?? -Infinity), t.max ?? Infinity);
}

/** Node, edge and global features of a (building, scenario) pair. */
export function graphFeatures(g: GraphFile, spec: ScenarioSpec, defaults: ModelFile["defaults"]) {
  const N = g.node_level.length;
  const E = g.edge_src.length;
  const nodeDim = g.node_base[0]?.length ?? 21;
  const edgeDim = g.edge_base[0]?.length ?? 11;
  const nx = new Float32Array(N * nodeDim);
  g.node_base.forEach((row, i) => nx.set(row, i * nodeDim));
  const o = N_TYPES;
  const lv = g.node_level;
  const set = (i: number, col: number, v: number) => {
    nx[i * nodeDim + o + col] = v;
  };
  for (let i = 0; i < N; i++) {
    if (spec.fire_level != null) set(i, 7, lv[i] === spec.fire_level ? 1 : 0);
    if (spec.warden_levels.length) set(i, 8, spec.warden_levels.includes(lv[i]) ? 1 : 0);
  }
  for (const [level, delay] of Object.entries(spec.phased_release ?? {}))
    for (let i = 0; i < N; i++) if (lv[i] === Number(level)) set(i, 9, Number(delay) / 600);
  const blocked = new Map<string, number>();
  for (const bk of spec.stair_blockages) if (!blocked.has(bk.stair)) blocked.set(bk.stair, medianTime(bk.time));
  g.landing_stair.forEach((sid, i) => {
    if (sid != null && blocked.has(sid)) {
      set(i, 10, 1);
      set(i, 11, blocked.get(sid)! / 900);
    }
  });
  if (spec.evacuation_lifts) g.lift_lobby.forEach((on, i) => set(i, 12, on ? 1 : 0));
  const ex = new Float32Array(E * edgeDim);
  g.edge_base.forEach((row, k) => ex.set(row, k * edgeDim));
  g.edge_stair.forEach((sid, k) => {
    if (sid != null && blocked.has(sid)) {
      ex[k * edgeDim + 9] = 1;
      ex[k * edgeDim + 10] = blocked.get(sid)! / 900;
    }
  });
  const gx = new Float32Array(N_GLOBAL);
  gx[SLOTS.indexOf(spec.time_slot)] = 1;
  gx[4] = spec.share_65_plus ?? defaults.share_65_plus;
  gx[5] = spec.share_80_plus_of_65_plus ?? defaults.share_80_plus_of_65_plus;
  gx[6] = spec.vacancy_rate ?? defaults.vacancy_rate;
  gx[7] = g.n_levels / 50;
  gx[8] = g.n_units / 500;
  gx[9] = g.occupants / 2000;
  gx[10] = g.care_share;
  gx[11] = g.n_lifts / 3;
  gx[12] = spec.lifts_out_of_service / Math.max(g.n_lifts + g.n_ff_lifts, 1);
  const evac = spec.evacuation_lifts ? 1 : 0;
  gx[13] = evac;
  gx[14 + Math.max(0, PRIORITY.indexOf(spec.lift_priority ?? "top_down"))] = evac;
  gx[17] = spec.evacuation_lifts && spec.lift_eligibility === "wheelchair_users" ? 1 : 0;
  gx[18] = spec.hazard != null && (spec.hazard.enabled ?? true) ? 1 : 0;
  gx[19] = (spec.fire_level ?? 0) / 50;
  gx[20] = spec.warden_levels.length / 5;
  gx[21] = g.n_stairs / 2;
  gx[22] = blocked.size / Math.max(g.n_stairs, 1);
  gx[23] = blocked.size ? Math.min(...blocked.values()) / 900 : 0;
  return { nx, ex, gx, N, E, nodeDim, edgeDim };
}

function standardise(x: Float32Array, dim: number, mu: number[], sd: number[]) {
  for (let k = 0; k < x.length; k++) {
    const c = k % dim;
    x[k] = (x[k] - mu[c]) / sd[c];
  }
}

/** Outcome quantiles (seconds) [loss][p50, p75, p90, p95, cvar95] and edge queueing (person-s). */
export function predictRaw(net: Network, g: GraphFile, spec: ScenarioSpec) {
  const f = graphFeatures(g, spec, net.file.defaults);
  const s = net.file.stats;
  standardise(f.nx, f.nodeDim, s.node_mu, s.node_sd);
  standardise(f.ex, f.edgeDim, s.edge_mu, s.edge_sd);
  for (let c = 0; c < N_GLOBAL; c++) f.gx[c] = (f.gx[c] - s.glob_mu[c]) / s.glob_sd[c];
  const H = net.file.config.hidden;
  const { N, E } = f;
  const src = g.edge_src;
  const dst = g.edge_dst;

  const h = runMlp(net.mlp("enc_n"), f.nx, N);
  const e = runMlp(net.mlp("enc_e"), f.ex, E);
  const gv = runMlp(net.mlp("enc_g"), f.gx, 1);
  const deg = new Float32Array(N);
  for (let k = 0; k < E; k++) deg[dst[k]] += 1;

  const pool = () => {
    const mean = new Float32Array(H);
    const mx = new Float32Array(H).fill(-Infinity);
    for (let i = 0; i < N; i++)
      for (let j = 0; j < H; j++) {
        const v = h[i * H + j];
        mean[j] += v;
        if (v > mx[j]) mx[j] = v;
      }
    for (let j = 0; j < H; j++) {
      mean[j] /= Math.max(N, 1);
      if (N === 0) mx[j] = 0;
    }
    return { mean, mx };
  };
  const edgeInput = () => {
    const x = new Float32Array(E * 3 * H);
    for (let k = 0; k < E; k++) {
      const off = k * 3 * H;
      x.set(h.subarray(src[k] * H, src[k] * H + H), off);
      x.set(h.subarray(dst[k] * H, dst[k] * H + H), off + H);
      x.set(e.subarray(k * H, k * H + H), off + 2 * H);
    }
    return x;
  };

  for (let l = 0; l < net.file.config.layers; l++) {
    const m = runMlp(net.mlp(`layers/${l}/msg`), edgeInput(), E);
    const agg = new Float32Array(N * H);
    for (let k = 0; k < E; k++) {
      const d = dst[k] * H;
      for (let j = 0; j < H; j++) agg[d + j] += m[k * H + j];
    }
    for (let i = 0; i < N; i++) {
      const inv = 1 / Math.max(deg[i], 1);
      for (let j = 0; j < H; j++) agg[i * H + j] *= inv;
    }
    const updIn = new Float32Array(N * 3 * H);
    for (let i = 0; i < N; i++) {
      updIn.set(h.subarray(i * H, i * H + H), i * 3 * H);
      updIn.set(agg.subarray(i * H, i * H + H), i * 3 * H + H);
      updIn.set(gv, i * 3 * H + 2 * H);
    }
    const du = runMlp(net.mlp(`layers/${l}/upd`), updIn, N);
    for (let k = 0; k < h.length; k++) h[k] += du[k];
    for (let k = 0; k < e.length; k++) e[k] += m[k];
    const { mean, mx } = pool();
    const gIn = new Float32Array(3 * H);
    gIn.set(gv, 0);
    gIn.set(mean, H);
    gIn.set(mx, 2 * H);
    const dg = runMlp(net.mlp(`layers/${l}/glob`), gIn, 1);
    for (let j = 0; j < H; j++) gv[j] += dg[j];
  }
  const { mean, mx } = pool();
  const headIn = new Float32Array(3 * H);
  headIn.set(gv, 0);
  headIn.set(mean, H);
  headIn.set(mx, 2 * H);
  const raw = runMlp(net.mlp("head_g"), headIn, 1);
  const quantiles = LOSS_NAMES.map((_, li) => {
    const r = raw.subarray(li * N_OUT, li * N_OUT + N_OUT);
    const out = [softplus(r[0])];
    for (let k = 1; k < N_OUT; k++) out.push(out[k - 1] + softplus(r[k]));
    return out.map((v) => v * TIME_SCALE);
  });
  const edgeOut = runMlp(net.mlp("head_e"), edgeInput(), E);
  const edges = Array.from(edgeOut, (v) => Math.max(Math.expm1(v), 0) * 60);
  return { quantiles, edges };
}

/** Where a case lies outside the training draw (mirror of predictor.coverage_notes). */
export function coverageNotes(g: GraphFile, spec: ScenarioSpec, file: ModelFile): string[] {
  const notes: string[] = [];
  const gen = g.metadata.generator;
  const args = g.metadata.args ?? {};
  const pretty = (k: string) => k.replace(/_/g, " ");
  if (gen == null || !(gen in file.typologies)) {
    notes.push(
      "The building is not one of the four generated templates (for example read from a floor plan or uploaded).",
    );
  } else {
    for (const [key, [lo, hi]] of Object.entries(file.typologies[gen])) {
      const v = args[key];
      if (typeof v === "number" && !(lo <= v && v <= hi)) notes.push(`${pretty(key)} = ${v} is outside the trained ${lo}–${hi}.`);
    }
    for (const [key, def] of Object.entries(file.template_defaults[gen] ?? {})) {
      if (key in file.typologies[gen]) continue;
      if (key in args && JSON.stringify(args[key]) !== JSON.stringify(def)) notes.push(`${pretty(key)} was not varied in training.`);
    }
  }
  const [slo, shi] = file.share_65_range;
  if (spec.share_65_plus != null && !(slo <= spec.share_65_plus && spec.share_65_plus <= shi))
    notes.push(`Share aged 65+ outside the trained ${Math.round(slo * 100)}%–${Math.round(shi * 100)}%.`);
  if (spec.stair_blockages.length > 1) notes.push("More than one staircase lost (training lost at most one).");
  if (spec.stair_blockages.some((b) => b.time.value == null))
    notes.push("A staircase lost at a random time (training used fixed times).");
  if (spec.lifts_out_of_service > 1) notes.push("More than one lift out of service (training had at most one).");
  const hz = (spec.hazard ?? {}) as Record<string, unknown>;
  const hazardChanged =
    spec.hazard != null &&
    Object.entries(hz).some(([k, v]) => k !== "enabled" && k in HAZARD_DEFAULTS && JSON.stringify(v ?? null) !== JSON.stringify(HAZARD_DEFAULTS[k]));
  const unseen: [string, boolean][] = [
    ["a random staircase loss", spec.random_stair_blockage != null],
    ["phased release", Object.keys(spec.phased_release ?? {}).length > 0],
    ["stair assignment", Object.keys(spec.stair_assignment ?? {}).length > 0],
    ["fire-service rescue settings", spec.rescue_start != null || spec.rescue_teams != null],
    ["counter-flow", spec.counter_flow_probability != null],
    ["vacancy", spec.vacancy_rate != null],
    ["the share aged 80+", spec.share_80_plus_of_65_plus != null],
    ["capacity changes", Object.keys(spec.capacity_multipliers ?? {}).length > 0],
    ["fire and door settings", hazardChanged],
  ];
  const missing = unseen.filter(([, on]) => on).map(([k]) => k);
  if (missing.length) notes.push(`Not varied in training: ${missing.join(", ")}.`);
  return notes;
}

/** The same response as POST /api/surrogate/predict. */
export function predict(net: Network, g: GraphFile, spec: ScenarioSpec, disclaimer: string, topEdges = 8): SurrogatePrediction {
  const t0 = performance.now();
  const { quantiles, edges } = predictRaw(net, g, spec);
  const horizon = net.file.horizon_s;
  const losses = Object.fromEntries(
    LOSS_NAMES.map((name, li) => {
      const q = quantiles[li].map((v) => Math.min(v, horizon));
      return [name, { ...Object.fromEntries(QUANTILE_KEYS.map((k, i) => [k, q[i]])), cvar95: q[4] }];
    }),
  ) as SurrogatePrediction["losses"];
  const ranked = g.edge_forward
    .map((f, i) => (f ? i : -1))
    .filter((i) => i >= 0)
    .sort((a, b) => edges[b] - edges[a])
    .slice(0, topEdges);
  return {
    losses,
    edges: ranked.map((i) => ({ edge: g.edge_ids[i], label: g.labels[g.edge_ids[i]] ?? g.edge_ids[i], person_minutes: edges[i] / 60 })),
    elapsed_ms: performance.now() - t0,
    horizon_s: horizon,
    coverage_notes: coverageNotes(g, spec, net.file),
    model: { trained_on: net.file.meta.trained_on ?? null, evaluation: net.file.meta.evaluation ?? null },
    disclaimer,
  };
}
