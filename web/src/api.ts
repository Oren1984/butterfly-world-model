import type { Metrics, PredictResponse, WorldState } from "./types";

// Always same-origin relative URLs: Vite (dev) or nginx (Docker) forwards /api to the backend,
// so no container-internal hostname ever reaches the browser.
async function call<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${method} ${path} failed with ${res.status}`);
  return (await res.json()) as T;
}

export interface ConfigChange {
  wind_strength?: number;
  wind_direction_deg?: number;
  randomness?: number;
  flowers?: { id: string; x: number; y: number }[];
}

export const api = {
  state: () => call<WorldState>("GET", "/api/world/state"),
  step: (steps: number) => call<WorldState>("POST", "/api/world/step", { steps }),
  reset: (newScenario: boolean) => call<WorldState>("POST", "/api/world/reset", { new_scenario: newScenario }),
  next: () => call<WorldState>("POST", "/api/world/next", {}),
  config: (change: ConfigChange) => call<WorldState>("POST", "/api/world/config", change),
  committed: () => call<PredictResponse>("GET", "/api/world/prediction"),
  predict: () => call<PredictResponse>("POST", "/api/predict", {}),
  evaluate: () => call<Metrics>("POST", "/api/evaluate", {}),
};
