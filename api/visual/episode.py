"""Run one visual-planning episode. The planner sees pixels and a goal image only."""
from __future__ import annotations

import base64
import time
from typing import Optional

import numpy as np

from ..evaluation.metrics import latency_stats
from ..planning.garden import LAYOUTS
from ..planning.planners import REPLAN_EVERY
from .jepa import VisualWorldModel
from .planners import VISUAL_PLANNERS
from .scene import FRAME_H, FRAME_SIZE, FRAME_W, VisualFlight, goal_image

VISUAL_PLANNER_STREAM = 23


def _b64(frame: np.ndarray) -> str:
    return base64.b64encode((np.clip(frame, 0, 1) * 255).astype(np.uint8).tobytes()).decode("ascii")


def run_visual_episode(
    layout_name: str,
    planner_name: str,
    seed: int,
    with_distractor: bool,
    model: VisualWorldModel,
    probes: Optional[dict] = None,
    record: bool = True,
) -> dict:
    flight = VisualFlight(layout_name, seed, with_distractor)
    goal = goal_image(flight.layout)
    rng = np.random.default_rng(np.random.SeedSequence([seed, VISUAL_PLANNER_STREAM]))
    planner = VISUAL_PLANNERS[planner_name](model, goal, rng)
    w_pos = None if probes is None else probes.get("probe_z_pos")
    latencies, evals, replans = [], [], []
    while not flight.world.done:
        px = flight.pixels()
        t0 = time.perf_counter()
        plan = planner.plan(px)
        latencies.append(time.perf_counter() - t0)
        evals.append(plan.evaluations)
        if record:
            entry = {"decision": flight.world.decision, "frame": _b64(px[FRAME_SIZE:])}
            if w_pos is not None:  # display only: the frozen diagnostic probe decodes imagined latents
                dec = np.concatenate([plan.latents, np.ones((len(plan.latents), 1))], axis=1) @ w_pos
                entry["imagined_route"] = [[round(float(x), 2), round(float(y), 2)] for x, y in dec]
                if plan.subgoal is not None:
                    sx, sy = np.append(plan.subgoal, 1.0) @ w_pos
                    entry["subgoal_decoded"] = [round(float(sx), 2), round(float(sy), 2)]
            replans.append(entry)
        for a in plan.actions[:REPLAN_EVERY]:
            flight.step(a)
            if flight.world.done:
                break
    w = flight.world
    out = {
        "layout": layout_name,
        "planner": planner_name,
        "seed": seed,
        "distractor": with_distractor,
        "outcome": w.status,
        "landed": w.status == "landed",
        "collided": w.status == "collided",
        "decisions": w.decision,
        "seconds_to_land": round(w.t * 0.1, 2) if w.status == "landed" else None,
        "replans": len(latencies),
        "evaluations_per_replan": evals[0]["total"] if evals else 0,
        "evaluations_total": int(sum(e["total"] for e in evals)),
        "latency": latency_stats(latencies) if latencies else None,
    }
    if record:
        out["frame_size"] = [FRAME_H, FRAME_W]
        out["goal_frame"] = _b64(goal[FRAME_SIZE:])
        out["path"] = [[round(x, 2), round(y, 2)] for x, y in w.path]
        out["distractor_path"] = [[round(x, 2), round(y, 2)] for x, y in flight.distractor_path]
        out["plans"] = replans
        out["layout_spec"] = LAYOUTS[layout_name].to_dict()
    return out
