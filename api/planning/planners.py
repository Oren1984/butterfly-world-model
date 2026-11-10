"""Goal-directed planners: flat, explicit-waypoint (hierarchical) and HWM-inspired subgoal.

All planners receive only a `FlightObservation` and a transition model. They
return the next actions to execute plus what they imagined, so the route and
the intermediate waypoints can be visualised.

Budgets are matched: every planner spends the same number of transition-model
evaluations per replan (`BUDGET`). For the hierarchical planners this count
includes waypoint or subgoal generation. One evaluation = one model prediction
for one candidate (one decision for the short model, one macro-action for the
longer-horizon model).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .garden import HEADINGS, N_ACTIONS, FlightObservation, Layout
from .models import MACRO_K

BUDGET = 960
REPLAN_EVERY = 2  # decisions executed between replans; identical for every planner
COLLISION_COST = 1000.0  # times the peak per-decision collision probability of a rollout
WAYPOINT_REACHED = 4.0
KEEP_WAYPOINT_BONUS = 5.0  # hysteresis: keep the current waypoint unless another is clearly better

FLAT = {"n_samples": 64, "horizon": 15}  # 64 * 15 = 960
WAYPOINT = {"n_candidates": 16, "macro_len": 30, "n_samples": 32, "horizon": 15}  # 480 + 480
SUBGOAL = {"n_sequences": 160, "n_macros": 3, "n_samples": 32, "horizon": 15}  # 480 + 480
assert FLAT["n_samples"] * FLAT["horizon"] == BUDGET
assert WAYPOINT["n_candidates"] * WAYPOINT["macro_len"] + WAYPOINT["n_samples"] * WAYPOINT["horizon"] == BUDGET
assert SUBGOAL["n_sequences"] * SUBGOAL["n_macros"] + SUBGOAL["n_samples"] * SUBGOAL["horizon"] == BUDGET


@dataclass
class PlanStep:
    actions: list[int]
    route: np.ndarray  # (H+1, 2) imagined positions of the chosen low-level plan
    waypoint: Optional[tuple[float, float]] = None  # current waypoint / subgoal, if any
    high_route: Optional[np.ndarray] = None  # imagined coarse route through the waypoint
    evaluations: dict = field(default_factory=dict)


def sample_sequences(rng: np.random.Generator, n: int, h: int, warm: Optional[np.ndarray] = None) -> np.ndarray:
    """Random action sequences that tend to hold an action for a few decisions."""
    change = rng.random((n, h)) < 0.3
    change[:, 0] = True
    vals = rng.integers(0, N_ACTIONS, (n, h))
    idx = np.maximum.accumulate(np.where(change, np.arange(h), 0), axis=1)
    seqs = np.take_along_axis(vals, idx, axis=1)
    if warm is not None:
        seqs[0] = warm
    return seqs


def shift_warm(prev: Optional[np.ndarray], h: int) -> Optional[np.ndarray]:
    if prev is None:
        return None
    rest = prev[REPLAN_EVERY:]
    return np.concatenate([rest, np.full(h - len(rest), rest[-1] if len(rest) else prev[-1])])[:h]


def greedy_actions(s: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Point the butterfly at the goal; slow down when close (a simple controller used inside rollouts)."""
    d = goal - s[:, :2]
    heading = np.round(np.arctan2(d[:, 1], d[:, 0]) / (2 * math.pi / HEADINGS)).astype(int) % HEADINGS
    slow = np.hypot(d[:, 0], d[:, 1]) < 8.0
    return heading + HEADINGS * slow


def shoot(model, s0, wind, seqs, goal, land: bool):
    """Roll every action sequence through the model and score it.

    land=True: reach and land on the target (landing sooner is better).
    land=False: pass close to a waypoint (closer and sooner is better).
    """
    n, h = seqs.shape
    s = np.repeat(s0[None, :], n, axis=0)
    risk = np.zeros(n)  # peak per-decision collision probability along each rollout
    cost = np.zeros(n)
    done = np.zeros(n, dtype=bool)
    land_t = np.zeros(n)
    min_d = np.full(n, np.inf)
    min_t = np.zeros(n)
    paths = np.empty((n, h + 1, 2))
    paths[:, 0] = s0[:2]
    for t in range(h):
        nxt, p_col, landed = model.step(s, seqs[:, t], wind)
        nxt = np.where(done[:, None], s, nxt)
        p_col = np.where(done, 0.0, p_col)
        risk = np.maximum(risk, p_col)
        if land:
            newly = landed & ~done
            land_t[newly] = t + 1
            done |= newly
        s = nxt
        paths[:, t + 1] = s[:, :2]
        d = np.hypot(s[:, 0] - goal[0], s[:, 1] - goal[1])
        better = d < min_d
        min_d = np.where(better, d, min_d)
        min_t = np.where(better, t + 1, min_t)
    cost += COLLISION_COST * risk
    if land:
        final_d = np.hypot(s[:, 0] - goal[0], s[:, 1] - goal[1])
        cost += np.where(done, land_t, h + 2.0 * final_d + 0.3 * min_d)
    else:
        cost += min_d + 0.1 * min_t
    return cost, paths


class FlatPlanner:
    """Scores action sequences toward the target directly (random shooting with warm start)."""

    kind = "flat"

    def __init__(self, layout: Layout, model, rng: np.random.Generator, cfg: dict = FLAT):
        self.layout, self.model, self.rng, self.cfg = layout, model, rng, cfg
        self._prev: Optional[np.ndarray] = None

    def plan(self, obs: FlightObservation) -> PlanStep:
        n, h = self.cfg["n_samples"], self.cfg["horizon"]
        seqs = sample_sequences(self.rng, n, h, shift_warm(self._prev, h))
        cost, paths = shoot(self.model, obs.state, obs.wind, seqs, np.array(self.layout.target), land=True)
        best = int(np.argmin(cost))
        self._prev = seqs[best]
        return PlanStep(
            actions=[int(a) for a in seqs[best, :REPLAN_EVERY]],
            route=paths[best],
            evaluations={"high": 0, "low": n * h, "total": n * h},
        )


def macro_rollouts(model, s0, wind, waypoints, target, length):
    """Fly toward each candidate waypoint, then toward the target, with the greedy controller."""
    m = len(waypoints)
    s = np.repeat(s0[None, :], m, axis=0)
    reached = np.hypot(*(waypoints - s0[:2]).T) < WAYPOINT_REACHED
    risk = np.zeros(m)
    cost = np.zeros(m)
    done = np.zeros(m, dtype=bool)
    land_t = np.zeros(m)
    paths = np.empty((m, length + 1, 2))
    paths[:, 0] = s0[:2]
    for t in range(length):
        goal = np.where(reached[:, None], target[None, :], waypoints)
        nxt, p_col, landed = model.step(s, greedy_actions(s, goal), wind)
        nxt = np.where(done[:, None], s, nxt)
        p_col = np.where(done, 0.0, p_col)
        risk = np.maximum(risk, p_col)
        newly = landed & ~done
        land_t[newly] = t + 1
        done |= newly
        s = nxt
        reached |= np.hypot(*(waypoints - s[:, :2]).T) < WAYPOINT_REACHED
        paths[:, t + 1] = s[:, :2]
    cost += COLLISION_COST * risk
    final_d = np.hypot(s[:, 0] - target[0], s[:, 1] - target[1])
    cost += np.where(done, land_t, length + 2.0 * final_d)
    return cost, paths


class WaypointPlanner:
    """Explicit-waypoint hierarchy: propose waypoints, score coarse routes through them, refine with shooting."""

    kind = "waypoint"

    def __init__(self, layout: Layout, model, rng: np.random.Generator, cfg: dict = WAYPOINT):
        self.layout, self.model, self.rng, self.cfg = layout, model, rng, cfg
        self.waypoint: Optional[np.ndarray] = None
        self._prev: Optional[np.ndarray] = None

    def _candidates(self) -> np.ndarray:
        k = self.cfg["n_candidates"]
        lo, hi = np.array([3.0, 3.0]), np.array([self.layout.width - 3.0, self.layout.height - 3.0])
        cands = [np.array(self.layout.target, dtype=float)]
        if self.waypoint is not None:
            cands.append(self.waypoint)
        cands += list(self.rng.uniform(lo, hi, (k - len(cands), 2)))
        return np.array(cands)

    def plan(self, obs: FlightObservation) -> PlanStep:
        target = np.array(self.layout.target, dtype=float)
        cands = self._candidates()
        hcost, hpaths = macro_rollouts(self.model, obs.state, obs.wind, cands, target, self.cfg["macro_len"])
        if self.waypoint is not None:
            hcost[1] -= KEEP_WAYPOINT_BONUS
        best = int(np.argmin(hcost))
        wp = cands[best]
        direct = best == 0 or math.dist(wp, (obs.x, obs.y)) < WAYPOINT_REACHED
        self.waypoint = None if direct else wp
        goal = target if direct else wp
        n, h = self.cfg["n_samples"], self.cfg["horizon"]
        seqs = sample_sequences(self.rng, n, h, shift_warm(self._prev, h))
        cost, paths = shoot(self.model, obs.state, obs.wind, seqs, goal, land=direct)
        b = int(np.argmin(cost))
        self._prev = seqs[b]
        high = len(cands) * self.cfg["macro_len"]
        return PlanStep(
            actions=[int(a) for a in seqs[b, :REPLAN_EVERY]],
            route=paths[b],
            waypoint=None if direct else (float(wp[0]), float(wp[1])),
            high_route=hpaths[best],
            evaluations={"high": high, "low": n * h, "total": high + n * h},
        )


class SubgoalPlanner:
    """HWM-inspired: plan over macro-actions with the longer-horizon learned model; the predicted
    state after the first macro-action becomes the subgoal for the short-horizon planner."""

    kind = "subgoal"
    near_target = 18.0  # within this distance the low level plans straight to the target

    def __init__(self, layout: Layout, model, rng: np.random.Generator, cfg: dict = SUBGOAL):
        if not hasattr(model, "macro"):
            raise ValueError("the subgoal planner needs a model with a longer-horizon (macro) head")
        self.layout, self.model, self.rng, self.cfg = layout, model, rng, cfg
        self._prev: Optional[np.ndarray] = None

    def plan(self, obs: FlightObservation) -> PlanStep:
        target = np.array(self.layout.target, dtype=float)
        m, k = self.cfg["n_sequences"], self.cfg["n_macros"]
        seqs = self.rng.integers(0, N_ACTIONS, (m, k))
        s = np.repeat(obs.state[None, :], m, axis=0)
        risk, cost, best_d = np.zeros(m), np.zeros(m), np.full(m, np.inf)
        ends = np.empty((m, k + 1, 2))
        ends[:, 0] = obs.state[:2]
        for b in range(k):
            s, p_col = self.model.macro(s, seqs[:, b], obs.wind)
            risk = np.maximum(risk, p_col)
            ends[:, b + 1] = s[:, :2]
            d = np.hypot(s[:, 0] - target[0], s[:, 1] - target[1])
            best_d = np.minimum(best_d, d + 3.0 * (b + 1))
        cost += COLLISION_COST * risk + best_d
        best = int(np.argmin(cost))
        direct = math.dist((obs.x, obs.y), target) < self.near_target
        sub = ends[best, 1]
        goal = target if direct else sub
        n, h = self.cfg["n_samples"], self.cfg["horizon"]
        lseqs = sample_sequences(self.rng, n, h, shift_warm(self._prev, h))
        lcost, paths = shoot(self.model, obs.state, obs.wind, lseqs, goal, land=direct)
        b = int(np.argmin(lcost))
        self._prev = lseqs[b]
        return PlanStep(
            actions=[int(a) for a in lseqs[b, :REPLAN_EVERY]],
            route=paths[b],
            waypoint=None if direct else (float(sub[0]), float(sub[1])),
            high_route=ends[best],
            evaluations={"high": m * k, "low": n * h, "total": m * k + n * h},
        )


# name -> (planner class, transition model kind)
PLANNERS: dict[str, tuple[type, str]] = {
    "flat_exact": (FlatPlanner, "exact"),
    "waypoint_exact": (WaypointPlanner, "exact"),
    "flat_learned": (FlatPlanner, "learned"),
    "waypoint_learned": (WaypointPlanner, "learned"),
    "subgoal_learned": (SubgoalPlanner, "learned"),
}
PLANNER_LABELS = {
    "flat_exact": "Flat planner (exact dynamics)",
    "waypoint_exact": "Hierarchical: explicit waypoints (exact dynamics)",
    "flat_learned": "Flat planner (learned model)",
    "waypoint_learned": "Hierarchical: explicit waypoints (learned model)",
    "subgoal_learned": "HWM-inspired: learned long-horizon subgoals",
}
MACRO_DECISIONS = MACRO_K
