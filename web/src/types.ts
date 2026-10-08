export const NO_LANDING = "no_landing_within_horizon";

export type Pt = [number, number];

export interface Flower {
  id: string;
  x: number;
  y: number;
  radius: number;
}

export interface Observation {
  step: number;
  butterfly: { x: number; y: number; vx: number; vy: number };
  wind: { strength: number; direction_deg: number };
  randomness: number;
  flowers: Flower[];
  airborne_steps: number;
  landed_on: string | null;
  width: number;
  height: number;
  dt: number;
}

export interface Future {
  outcome: string;
  landing_step: number | null;
  landing_position: Pt | null;
  path: Pt[];
}

export interface Prediction {
  observation_step: number;
  n_futures: number;
  horizon_steps: number;
  path_stride: number;
  probabilities: Record<string, number>;
  top_outcome: string;
  confidence: number;
  predicted_landing_position: Pt | null;
  expected_landing_step: number | null;
  uncertainty: {
    entropy: number;
    landing_spread: number | null;
    spread_over_time: number[];
    intent_belief: Record<string, number>;
  };
  futures: Future[];
}

export interface PredictResponse {
  config_version: number | null;
  prediction: Prediction;
}

export interface Committed {
  observation_step: number;
  top_outcome: string;
  confidence: number;
  probabilities: Record<string, number>;
  predicted_landing_position: Pt | null;
}

export interface CycleResult {
  predicted: string;
  actual: string;
  confidence: number;
  probability_of_actual: number;
  correct: boolean;
  position_error: number | null;
  predicted_position: Pt | null;
  actual_position: Pt | null;
  flight_steps: number;
}

export type Status = "observing" | "flying" | "landed" | "no_landing";

export interface WorldState {
  config_version: number;
  seed: number;
  status: Status;
  observation: Observation;
  horizon_steps: number;
  steps_remaining: number;
  committed: Committed | null;
  result: CycleResult | null;
  session: { cycles: number; correct: number };
  path: Pt[];
}

export interface PredictorMetrics {
  top1_accuracy: number;
  top1_accuracy_ci95: [number, number];
  brier_score: number;
  landing_position_error: { coverage: number; n_defined: number; mean: number | null; median: number | null };
  calibration: { expected_calibration_error: number | null };
  latency: { mean_ms: number; p50_ms: number; p90_ms: number; p99_ms: number; max_ms: number };
}

export interface Metrics {
  n_scenarios: number;
  n_futures: number;
  horizon_seconds: number;
  actual_outcomes: Record<string, number>;
  model: PredictorMetrics;
  baselines: Record<string, PredictorMetrics>;
}
