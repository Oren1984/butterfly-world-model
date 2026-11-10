import type { Pt } from "./types";

export type Mode = "prediction" | "planning" | "visual";

export interface Rect {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface LayoutSpec {
  name: string;
  description: string;
  start: Pt;
  target: { x: number; y: number; radius: number };
  obstacles: Rect[];
  reachable: boolean;
  width: number;
  height: number;
}

export interface LatencyStats {
  mean_ms: number;
  p50_ms: number;
  p90_ms: number;
  p99_ms: number;
  max_ms: number;
}

export interface PlanningPlan {
  decision: number;
  sim_step: number;
  position: Pt;
  route: Pt[];
  waypoint: Pt | null;
  high_route: Pt[] | null;
}

export interface VisualPlan {
  decision: number;
  frame: string; // base64 uint8 frame (rows x cols)
  imagined_route?: Pt[];
  subgoal_decoded?: Pt;
}

export interface EpisodeBase {
  layout: string;
  seed: number;
  outcome: "landed" | "collided" | "timeout" | "flying";
  landed: boolean;
  collided: boolean;
  decisions: number;
  seconds_to_land: number | null;
  replans: number;
  evaluations_per_replan: number;
  evaluations_total: number;
  latency: LatencyStats | null;
  path: Pt[];
  layout_spec: LayoutSpec;
  source: string;
  model_version: string;
}

export interface PlanningEpisode extends EpisodeBase {
  planner: string;
  evaluations_high_total: number;
  plans: PlanningPlan[];
}

export interface VisualEpisode extends EpisodeBase {
  planner: "one_level" | "two_level";
  distractor: boolean;
  frame_size: [number, number];
  goal_frame: string;
  distractor_path: Pt[];
  plans: VisualPlan[];
  model_seed: number;
  model_flags: string[];
}

export interface ModesInfo {
  default: Mode;
  planning: {
    layouts: Record<string, LayoutSpec>;
    planners: Record<string, { label: string; model: "exact" | "learned" }>;
    learned_layouts: string[];
    budget_per_replan: number;
    eval_seeds: [number, number];
    learned_model: { available: boolean; model_version?: string; message?: string };
    recorded: { generated_utc: string } | null;
  };
  visual: {
    layouts: string[];
    budget_per_replan: number;
    eval_seeds: [number, number];
    models: { available: boolean; models: { variant: string; model_seed: number; model_version: string; flags: string[] }[]; message?: string };
    recorded: { generated_utc: string } | null;
  };
}

export interface PlannerSummary {
  episodes: number;
  landing_rate: number;
  landing_rate_ci95: [number, number];
  collision_rate: number;
  mean_seconds_to_land: number | null;
  evaluations_per_replan: number;
  high_level_share_of_evaluations: number;
  planning_latency_ms_per_replan: { p50: number; mean: number };
}

export interface PlanningResults {
  generated_utc: string;
  seeds: { first: number; last: number; count: number };
  config: { budget_evaluations_per_replan: number; replan_every_decisions: number };
  exact_dynamics: Record<string, Record<string, PlannerSummary>>;
  learned_model: Record<string, Record<string, PlannerSummary>> | null;
  learned_model_version: string | null;
}

export interface VisualCell {
  mean_landing_rate: number;
  sd_landing_rate_across_model_seeds: number;
  collision_rate: number;
  per_model_seed: Record<string, number>;
}

export interface VisualResults {
  generated_utc: string;
  seeds: { first: number; last: number; count: number };
  model_seeds: number[];
  config: { budget_evaluations_per_replan: number };
  by_variant: Record<string, Record<string, { clean: VisualCell; distractor: VisualCell; change_under_distraction: number }>>;
  probe_r2_mean_over_model_seeds: Record<string, Record<string, { butterfly_position_r2: number; distractor_position_r2: number }>>;
  failures: Record<string, string[]>;
}
