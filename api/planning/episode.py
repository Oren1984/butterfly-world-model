"""Run one goal-directed episode: observe, plan, act, repeat on a fixed cadence."""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ..evaluation.metrics import latency_stats
from .garden import LAYOUTS, ControlledFlight
from .models import ExactModel
from .planners import PLANNERS, REPLAN_EVERY

PLANNER_STREAM = 13


def make_planner(name: str, layout_name: str, seed: int, learned_weights: Optional[dict] = None):
    if name not in PLANNERS:
        raise KeyError(name)
    cls, kind = PLANNERS[name]
    layout = LAYOUTS[layout_name]
    if kind == "exact":
        model = ExactModel(layout)
    else:
        from .learned import load, model_from

        weights = learned_weights if learned_weights is not None else load()[0]
        model = model_from(weights, layout_name)
    # Same planner random stream for every arm on a given seed.
    rng = np.random.default_rng(np.random.SeedSequence([seed, PLANNER_STREAM]))
    return cls(layout, model, rng)


def _pts(a) -> list[list[float]]:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in a]


def run_episode(
    layout_name: str,
    planner_name: str,
    seed: int,
    learned_weights: Optional[dict] = None,
    record: bool = True,
) -> dict:
    layout = LAYOUTS[layout_name]
    world = ControlledFlight(layout, seed)
    planner = make_planner(planner_name, layout_name, seed, learned_weights)
    replans, latencies, evals = [], [], []
    while not world.done:
        obs = world.observe()
        t0 = time.perf_counter()
        step = planner.plan(obs)
        latencies.append(time.perf_counter() - t0)
        evals.append(step.evaluations)
        if record:
            replans.append(
                {
                    "decision": obs.decision,
                    "sim_step": world.t,
                    "position": [round(obs.x, 2), round(obs.y, 2)],
                    "route": _pts(step.route),
                    "waypoint": None if step.waypoint is None else [round(v, 2) for v in step.waypoint],
                    "high_route": None if step.high_route is None else _pts(step.high_route),
                }
            )
        for a in step.actions[:REPLAN_EVERY]:
            world.step(a)
            if world.done:
                break
    landed = world.status == "landed"
    total = {k: int(sum(e[k] for e in evals)) for k in ("high", "low", "total")}
    out = {
        "layout": layout_name,
        "planner": planner_name,
        "seed": seed,
        "outcome": world.status,
        "landed": landed,
        "collided": world.status == "collided",
        "decisions": world.decision,
        "seconds_to_land": round(world.t * 0.1, 2) if landed else None,
        "replans": len(latencies),
        "evaluations_per_replan": evals[0]["total"] if evals else 0,
        "evaluations_total": total["total"],
        "evaluations_high_total": total["high"],
        "latency": latency_stats(latencies) if latencies else None,
        "wind": world.wind_info,
    }
    if record:
        out["path"] = _pts(world.path)
        out["plans"] = replans
        out["layout_spec"] = layout.to_dict()
    return out
