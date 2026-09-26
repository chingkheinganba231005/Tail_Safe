// Shapes of the TailSafe API responses (see tailsafe/api/views.py and app.py).
// Times are in seconds; the UI converts to minutes for display.

export type Point = [number, number];

export interface Feature {
  kind: string;
  polygon: Point[];
  label?: string;
}

export interface Level {
  index: number;
  label: string;
  elevation: number;
  height: number;
  kind: string;
  features?: Feature[];
}

export interface StairDef {
  id: string;
  label: string;
  kind: string;
  clear_width: number;
  riser: number;
  going: number;
}

export interface NodeDef {
  id: string;
  type: string;
  level: number;
  x: number;
  y: number;
  label?: string;
  area?: number;
  polygon?: Point[];
  stair?: string;
  tags?: string[];
}

export interface EdgeDef {
  id: string;
  source: string;
  target: string;
  kind: "flat" | "door" | "stair" | "lift";
  length: number;
  width: number;
  directed?: boolean;
  fire_rated?: boolean;
  self_closing?: boolean;
  can_block?: boolean;
  opening?: Point[];
  label?: string;
  stair?: string;
}

export interface LiftDef {
  id: string;
  label: string;
  stops: { level: number; node: string }[];
  discharge_level: number;
  firefighting: boolean;
}

export interface Building {
  schema_version: number;
  id: string;
  name: string;
  typology: string;
  description?: string;
  levels: Level[];
  stairs: StairDef[];
  nodes: NodeDef[];
  edges: EdgeDef[];
  lifts: LiftDef[];
  metadata?: Record<string, unknown>;
}

export interface BuildingSummary {
  id: string;
  name: string;
  typology: string;
  storeys: number;
  units: number;
  stairs: string[];
  lifts: string[];
  exits: string[];
  height_m: number;
  digest: string;
}

export interface BuildingView {
  id: string;
  summary: BuildingSummary;
  building: Building;
}

export interface Template {
  name: string;
  description: string;
  options: Record<string, number | boolean>;
}

export interface Dist {
  dist: "constant" | "uniform" | "truncnorm" | "lognormal";
  value?: number | null;
  min?: number | null;
  max?: number | null;
  mean?: number | null;
  sd?: number | null;
  median?: number | null;
  sigma?: number | null;
}

export interface HazardSpec {
  enabled: boolean;
  fire_unit?: string | null;
  growth?: Dist | null;
  peak_hrr?: Dist | null;
  door_open_probability?: number | null;
  hold_open_stair_doors?: boolean;
  held_open_doors?: string[];
  horizon?: number;
  record_dt?: number;
}

export type TimeSlot = "weekday_day" | "weekday_night" | "weekend_day" | "weekend_night";
export type LiftPriority = "top_down" | "nearest" | "bottom_up";
export type LiftEligibility = "mobility_impaired" | "wheelchair_users";

export interface ScenarioSpec {
  name: string;
  description?: string | null;
  time_slot: TimeSlot;
  share_65_plus?: number | null;
  share_80_plus_of_65_plus?: number | null;
  counter_flow_probability?: number | null;
  vacancy_rate?: number | null;
  fire_level?: number | null;
  stair_blockages: { stair: string; time: Dist }[];
  random_stair_blockage?: { probability: number; time: Dist } | null;
  evacuation_lifts: boolean;
  lifts_out_of_service: number;
  lift_priority: LiftPriority;
  lift_eligibility?: LiftEligibility;
  phased_release: Record<string, number>;
  stair_assignment: Record<string, string>;
  rescue_start?: Dist | null;
  rescue_teams?: number | null;
  hazard?: HazardSpec | null;
  warden_levels: number[];
  capacity_multipliers: Record<string, number>;
}

export interface Estimate {
  value: number;
  lo: number;
  hi: number;
}

export interface RiskSummary {
  n: number;
  alpha: number;
  confidence: number;
  mean: Estimate;
  median: Estimate;
  p95: Estimate;
  p99: Estimate;
  var: Estimate;
  cvar: Estimate;
  max: number;
  censored: number;
}

export const LOSSES = ["total_time", "self_evacuation_time", "p95_occupant_time"] as const;
export type Loss = (typeof LOSSES)[number];

export interface BreakdownRow {
  category: string;
  occupant_share: number;
  straggler_share: number;
  risk_ratio: number | null;
  last_out_share: number;
}

export interface Breakdown {
  loss: string;
  alpha: number;
  var: number;
  tail_scenarios: number;
  straggler_fraction: number;
  profiles: BreakdownRow[];
  floor_bands: BreakdownRow[];
  headline: string;
}

export interface Tenability {
  p_rset_exceeds_aset?: Estimate;
  p_any_incapacitated?: Estimate;
  mean_incapacitated?: number;
  mean_over_fed_limit?: number;
  cvar_incapacitated?: number;
  worst_floors?: ({ level: number } & Estimate)[];
}

export interface StressResult {
  runs: number;
  seed: number;
  elapsed_s: number;
  spec: ScenarioSpec;
  risk: Record<Loss, RiskSummary>;
  losses: Record<Loss, number[]>;
  occupants_mean: number;
  rescued_mean: number;
  tenability: Tenability;
  floor_exceedance: { level: number; p: number; lo: number; hi: number }[];
  breakdown: Partial<Record<"total_time" | "self_evacuation_time", Breakdown>>;
  stair_congestion: Record<string, { level: number; person_seconds: number }[]>;
  worst_scenarios: number[];
  median_scenario: number;
  disclaimer: string;
}

export interface Replay {
  times: number[];
  levels: number[];
  stair_queues: Record<string, number[][]>;
  remaining: number[][];
  evacuated: number[];
  visibility: number[][] | null;
  info: {
    rescue_start?: number;
    fire_level?: number;
    blocked_stairs?: Record<string, number>;
    lifts_out?: string[];
    evacuation_lifts?: string[];
    warden_escorts?: number;
    fire?: { unit: string; growth_kw_s2: number; peak_kw: number; door_open: boolean };
  };
  summary: {
    occupants: number;
    groups: number;
    total_time_s: number;
    self_evacuation_time_s: number;
    p50_exit_s: number;
    p95_exit_s: number;
    rescued_occupants: number;
    incapacitated_occupants: number;
    max_fed: number;
    not_evacuated: number;
    lift_trips: number;
  };
  disclaimer: string;
}

export interface BottleneckRow {
  rank: number;
  key: string;
  label: string;
  kind: "stair" | "stair_doors" | "exit" | "edge" | "unblock";
  delta_cvar: Estimate | null;
  relative_change: number | null;
  recurrence: number;
  tail_person_seconds: number;
  structural_clearance_s: number;
  in_min_cut: boolean;
  edges: string[];
  stair: string | null;
}

export interface QueueSpot {
  arc: number;
  where: string;
  edge: string;
  recurrence: number;
  tail_person_seconds: number;
  all_person_seconds: number;
  tail_max_queue: number;
}

export interface BottleneckResult {
  loss: Loss;
  alpha: number;
  baseline_cvar: number;
  factor: number;
  rerun_scenarios: Record<string, number>;
  max_flow_persons_per_s: number;
  min_cut: string[];
  queues: QueueSpot[];
  ranking: BottleneckRow[];
  headline: string;
}

export interface Plan {
  evacuation_lifts: boolean;
  lift_priority: LiftPriority;
  lift_eligibility?: LiftEligibility;
  hold_open_stair_doors: boolean;
  stair_split_level: number | null;
  upper_stair: string | null;
  lower_stair: string | null;
  band_edges: number[];
  band_delays: number[];
  warden_levels: number[];
}

export interface Evaluation {
  stage: string;
  objective: number;
  cvar: number;
  mean: number;
  p_rset_exceeds_aset: number;
  plan: Plan;
  levers: string[];
}

export interface ConfirmLoss {
  before_cvar: number;
  after_cvar: number;
  delta_cvar: Estimate;
  relative: number | null;
  before_mean: number;
  after_mean: number;
  significant: boolean;
}

export interface OptimizeResult {
  objective: { kind: string; loss: Loss; alpha: number; mean_weight: number; label: string };
  baseline: Evaluation;
  best: Evaluation;
  plan_description: string[];
  evaluations: number;
  history: Evaluation[];
  confirmation: {
    scenarios: number;
    seed: number;
    losses: Record<Loss, ConfirmLoss>;
    p_rset_exceeds_aset: { before: number; after: number; delta: Estimate };
    significant: boolean;
  };
  before_after: Record<Loss, { before: number[]; after: number[] }>;
  replay: {
    baseline_spec: ScenarioSpec;
    plan_spec: ScenarioSpec;
    seed: number;
    runs: number;
    batch_size: number;
    worst_index: number;
  };
  disclaimer: string;
}

export type JobStatus = "queued" | "running" | "done" | "error";

export interface Job {
  id: string;
  kind: string;
  status: JobStatus;
  done: number;
  total: number;
  message: string;
  error: string | null;
  cached: boolean;
  elapsed_s: number;
}

export interface MicroTimes {
  p50_s: number;
  p95_s: number;
  last_s: number;
}

export interface MicroResult {
  levels: number[];
  frame_dt: number;
  frames: number;
  people_on_level: number[][]; // [frame][level position]
  rooms: { id: string; type: string; level: number; label?: string; polygon: Point[] }[];
  comparison: { walkers: number; occupants: number; meso: MicroTimes; micro: MicroTimes };
  summary: { occupants: number; walked: number; total_time_s: number; not_out: number; forced_moves: number };
  meso_summary: { total_time_s: number };
  info: Replay["info"];
  disclaimer: string;
}

/** One floor of a micro replay: per frame, [person, x (dm), y (dm), state] of those on it. */
export interface MicroLevel {
  level: number;
  times: number[];
  frames: [number, number, number, number][][];
  scale: number;
  states: Record<string, number>;
}

export type PlanRoomType = "unit" | "corridor" | "lobby" | "stair" | "refuge" | "void";

export interface PlanRoom {
  id: string;
  type: PlanRoomType;
  rects: number[][]; // [x0, y0, x1, y1] image pixels, y down
  area_m2: number;
  doors: number;
  stair_score: number;
  unit_type?: string | null;
}

export interface PlanDoor {
  id: string;
  a: number[];
  b: number[];
  width_m: number;
  rooms: string[];
}

export interface PlanDetection {
  width: number;
  height: number;
  m_per_px: number;
  scale_source: "reference" | "walls";
  wall_px: number;
  rooms: PlanRoom[];
  doors: PlanDoor[];
  warnings: string[];
}

export interface PlanScale {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  metres: number;
}
