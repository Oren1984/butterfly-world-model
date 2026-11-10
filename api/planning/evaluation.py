"""Reproducible comparison of the goal-directed planners on frozen evaluation seeds."""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from ..evaluation.metrics import wilson_interval
from .episode import run_episode
from .garden import LAYOUTS, LEARNED_LAYOUTS, MAX_DECISIONS, N_ACTIONS
from .planners import BUDGET, FLAT, PLANNERS, REPLAN_EVERY, SUBGOAL, WAYPOINT
from .seeds import PLANNING_EVAL_SEEDS

EXACT = ("flat_exact", "waypoint_exact")
LEARNED = ("flat_learned", "waypoint_learned", "subgoal_learned")


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    landed = [r for r in rows if r["landed"]]
    lo, hi = wilson_interval(len(landed), n)
    lat = np.array([r["latency_p50_ms"] for r in rows])
    return {
        "episodes": n,
        "landing_rate": len(landed) / n,
        "landing_rate_ci95": [lo, hi],
        "collision_rate": sum(r["collided"] for r in rows) / n,
        "timeout_rate": sum(r["outcome"] == "timeout" for r in rows) / n,
        "mean_seconds_to_land": float(np.mean([r["seconds_to_land"] for r in landed])) if landed else None,
        "median_seconds_to_land": float(np.median([r["seconds_to_land"] for r in landed])) if landed else None,
        "mean_decisions": float(np.mean([r["decisions"] for r in rows])),
        "evaluations_per_replan": int(rows[0]["evaluations_per_replan"]),
        "high_level_share_of_evaluations": float(
            sum(r["evaluations_high_total"] for r in rows) / max(sum(r["evaluations_total"] for r in rows), 1)
        ),
        "mean_evaluations_per_episode": float(np.mean([r["evaluations_total"] for r in rows])),
        "planning_latency_ms_per_replan": {"p50": float(np.median(lat)), "mean": float(lat.mean())},
    }


def run_comparison(
    seeds=PLANNING_EVAL_SEEDS,
    learned: Optional[tuple[dict, dict]] = None,
    log: Callable[[str], None] = print,
) -> tuple[dict, list[dict]]:
    """Run every planner on identical (layout, seed) pairs. `learned` = (weights, meta) or None."""
    seeds = list(seeds)
    t0 = time.perf_counter()
    plan: list[tuple[str, str]] = [(lay, p) for lay in ("direct", "detour") for p in EXACT]
    plan += [("enclosed", p) for p in EXACT]
    if learned is not None:
        plan += [(lay, p) for lay in LEARNED_LAYOUTS for p in LEARNED]
    rows: list[dict] = []
    for layout, planner in plan:
        for seed in seeds:
            r = run_episode(layout, planner, seed, learned_weights=learned[0] if learned else None, record=False)
            rows.append(
                {
                    "layout": layout,
                    "planner": planner,
                    "model": PLANNERS[planner][1],
                    "seed": seed,
                    "outcome": r["outcome"],
                    "landed": r["landed"],
                    "collided": r["collided"],
                    "decisions": r["decisions"],
                    "seconds_to_land": r["seconds_to_land"],
                    "replans": r["replans"],
                    "evaluations_per_replan": r["evaluations_per_replan"],
                    "evaluations_total": r["evaluations_total"],
                    "evaluations_high_total": r["evaluations_high_total"],
                    "latency_p50_ms": r["latency"]["p50_ms"],
                    "wind_mean_strength": r["wind"]["mean_strength"],
                }
            )
        s = summarise([r for r in rows if r["layout"] == layout and r["planner"] == planner])
        log(f"{layout:<9}{planner:<18} landed {s['landing_rate']:6.1%}  collided {s['collision_rate']:5.1%}"
            f"  p50 {s['planning_latency_ms_per_replan']['p50']:5.1f} ms/replan")

    def group(names):
        out = {}
        for layout in ("direct", "detour", "enclosed"):
            sub = {p: summarise(rs) for p in names if (rs := [r for r in rows if r["layout"] == layout and r["planner"] == p])}
            if sub:
                out[layout] = sub
        return out

    metrics = {
        "experiment": "goal-directed planning (HWM-inspired)",
        "app_version": "1.1.0",
        "seeds": {"first": seeds[0], "last": seeds[-1], "count": len(seeds), "range_name": "PLANNING_EVAL_SEEDS"},
        "config": {
            "budget_evaluations_per_replan": BUDGET,
            "replan_every_decisions": REPLAN_EVERY,
            "decision_seconds": 0.2,
            "max_decisions": MAX_DECISIONS,
            "n_actions": N_ACTIONS,
            "flat": FLAT,
            "waypoint": WAYPOINT,
            "subgoal": SUBGOAL,
            "layouts": {k: v.to_dict() for k, v in LAYOUTS.items()},
        },
        "exact_dynamics": group(EXACT),
        "learned_model": group(LEARNED) if learned else None,
        "learned_model_version": learned[1]["model_version"] if learned else None,
        "learned_model_validation": learned[1]["metrics"] if learned else None,
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return metrics, rows
