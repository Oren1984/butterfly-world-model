"""Reproducible evaluation: predict first, freeze, then let the actual world run."""
import time
from typing import Callable, Optional

import numpy as np

from ..contracts import Observation
from ..params import NO_LANDING, PARAMS
from ..prediction import baseline
from ..prediction.predictor import DEFAULT_HEADING_KAPPA, predict
from ..simulation.world import PREDICTOR_STREAM, ButterflyWorld, make_scenario
from .metrics import brier_score, position_error, summarise, top_outcome

EVAL_BASE_SEED = 0  # evaluation scenarios use seeds 0 .. n-1
TUNING_BASE_SEED = 100_000  # predictor tuning uses seeds 100000+ (disjoint from evaluation)

PREDICTORS = ("model", "constant_velocity", "nearest_flower")


def _score(row: dict, prefix: str, probs: dict, predicted_pos, actual: str, actual_pos, latency: float) -> None:
    top = top_outcome(probs)
    row[f"{prefix}_top"] = top
    row[f"{prefix}_confidence"] = probs[top]
    row[f"{prefix}_correct"] = top == actual
    row[f"{prefix}_p_actual"] = probs[actual]
    row[f"{prefix}_brier"] = brier_score(probs, actual)
    # Defined only when a flower was predicted and the butterfly actually landed.
    row[f"{prefix}_position_error"] = position_error(predicted_pos if top != NO_LANDING else None, actual_pos)
    row[f"{prefix}_latency_s"] = latency


def run_scenario(
    seed: int,
    n_futures: int = 20,
    horizon_steps: int = PARAMS.default_horizon_steps,
    kappa: float = DEFAULT_HEADING_KAPPA,
) -> dict:
    scenario = make_scenario(seed)
    world = ButterflyWorld(scenario, seed)
    for _ in range(PARAMS.min_airborne_steps):  # warm-up flight; landing is not possible yet
        world.step()
    obs: Observation = world.observe()  # the prediction snapshot

    # All predictions are produced and frozen before the world advances any further.
    timed: dict[str, tuple] = {}
    calls: dict[str, Callable] = {
        "model": lambda: (
            lambda pr: (pr.probabilities, pr.predicted_landing_position)
        )(predict(obs, n_futures, horizon_steps, np.random.default_rng([seed, PREDICTOR_STREAM]), kappa=kappa)),
        "constant_velocity": lambda: baseline.constant_velocity(obs, horizon_steps),
        "nearest_flower": lambda: baseline.nearest_flower(obs),
    }
    for name, call in calls.items():
        t0 = time.perf_counter()
        probs, pos = call()
        timed[name] = (probs, pos, time.perf_counter() - t0)

    steps = 0
    while steps < horizon_steps and world.landed_on is None:
        world.step()
        steps += 1
    actual = world.landed_on or NO_LANDING
    actual_pos: Optional[tuple[float, float]] = (world.x, world.y) if world.landed_on else None

    row = {
        "seed": seed,
        "n_flowers": len(scenario.flowers),
        "wind_strength": scenario.wind_strength,
        "wind_direction_deg": scenario.wind_direction_deg,
        "randomness": scenario.randomness,
        "actual": actual,
        "actual_steps": steps if world.landed_on else None,
    }
    for name, (probs, pos, latency) in timed.items():
        _score(row, name, probs, pos, actual, actual_pos, latency)
    return row


def _by_bucket(rows: list[dict], key: str, edges: list[float], labels: list[str]) -> list[dict]:
    out = []
    for i, label in enumerate(labels):
        sub = [r for r in rows if edges[i] <= r[key] < edges[i + 1] or (i == len(labels) - 1 and r[key] == edges[-1])]
        entry = {"bucket": label, "range": [edges[i], edges[i + 1]], "count": len(sub)}
        for name in PREDICTORS:
            entry[f"{name}_accuracy"] = sum(r[f"{name}_correct"] for r in sub) / len(sub) if sub else None
        out.append(entry)
    return out


def run_experiment(
    n_scenarios: int = 500,
    base_seed: int = EVAL_BASE_SEED,
    n_futures: int = 20,
    horizon_steps: int = PARAMS.default_horizon_steps,
    kappa: float = DEFAULT_HEADING_KAPPA,
) -> tuple[dict, list[dict]]:
    rows = [run_scenario(base_seed + i, n_futures, horizon_steps, kappa) for i in range(n_scenarios)]
    no_landing = sum(r["actual"] == NO_LANDING for r in rows)
    w = PARAMS.max_wind
    metrics = {
        "n_scenarios": n_scenarios,
        "seeds": {"first": base_seed, "last": base_seed + n_scenarios - 1},
        "n_futures": n_futures,
        "horizon_steps": horizon_steps,
        "horizon_seconds": horizon_steps * PARAMS.dt,
        "heading_kappa": kappa,
        "actual_outcomes": {"landed": n_scenarios - no_landing, NO_LANDING: no_landing},
        "model": summarise(rows, "model"),
        "baselines": {name: summarise(rows, name) for name in PREDICTORS[1:]},
        "accuracy_by_wind": _by_bucket(rows, "wind_strength", [0, w / 3, 2 * w / 3, w], ["low", "medium", "high"]),
        "accuracy_by_randomness": _by_bucket(rows, "randomness", [0, 0.4, 0.7, 1.0], ["low", "medium", "high"]),
    }
    return metrics, rows
