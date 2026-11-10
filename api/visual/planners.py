"""Latent-space planners for the visual experiment. Inputs: pixels and a goal image only.

One-level: random shooting over action sequences with the low-level latent
predictor; cost = distance to the goal embedding.

Two-level (H-JEPA-inspired): random shooting over macro-actions with the
higher-level predictor toward the goal's high-level embedding. The predicted
high-level state after the first macro-action is projected into the low-level
space and used as a subgoal for the low-level planner.

Budgets are matched (`VISUAL_BUDGET` predictor evaluations per replan); the
two-level count includes the high-level search.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..planning.garden import N_ACTIONS
from ..planning.planners import REPLAN_EVERY, sample_sequences, shift_warm
from .jepa import VisualWorldModel

VISUAL_BUDGET = 768
ONE_LEVEL = {"n_samples": 64, "horizon": 12}  # 768
TWO_LEVEL = {"n_sequences": 64, "n_macros": 2, "n_samples": 64, "horizon": 10}  # 128 + 640
assert ONE_LEVEL["n_samples"] * ONE_LEVEL["horizon"] == VISUAL_BUDGET
assert TWO_LEVEL["n_sequences"] * TWO_LEVEL["n_macros"] + TWO_LEVEL["n_samples"] * TWO_LEVEL["horizon"] == VISUAL_BUDGET
RISK_COST = 2.0  # times the peak predicted collision probability
GOAL_WEIGHT = 0.5


@dataclass
class LatentPlan:
    actions: list[int]
    latents: np.ndarray  # (H+1, Z) imagined low-level latents of the chosen plan
    subgoal: Optional[np.ndarray] = None  # (Z,) low-level subgoal latent, two-level only
    evaluations: dict = field(default_factory=dict)


def _d(z: np.ndarray, g: np.ndarray) -> np.ndarray:
    return ((z - g) ** 2).mean(axis=-1)


def latent_shoot(model: VisualWorldModel, z0, seqs, target, goal):
    n, h = seqs.shape
    z = np.repeat(z0[None], n, axis=0)
    risk, best = np.zeros(n), np.full(n, np.inf)
    traj = np.empty((n, h + 1, z0.shape[0]), dtype=np.float32)
    traj[:, 0] = z0
    for t in range(h):
        z, p = model.step(z, seqs[:, t])
        risk = np.maximum(risk, p)
        best = np.minimum(best, _d(z, target))
        traj[:, t + 1] = z
    return best + GOAL_WEIGHT * _d(z, goal) + RISK_COST * risk, traj


class OneLevelPlanner:
    kind = "one_level"

    def __init__(self, model: VisualWorldModel, goal_pixels: np.ndarray, rng: np.random.Generator, cfg=ONE_LEVEL):
        self.model, self.rng, self.cfg = model, rng, cfg
        self.z_goal = model.encode(goal_pixels[None])[0]
        self._prev = None

    def plan(self, pixels: np.ndarray) -> LatentPlan:
        z0 = self.model.encode(pixels[None])[0]
        n, h = self.cfg["n_samples"], self.cfg["horizon"]
        seqs = sample_sequences(self.rng, n, h, shift_warm(self._prev, h))
        cost, traj = latent_shoot(self.model, z0, seqs, self.z_goal, self.z_goal)
        b = int(np.argmin(cost))
        self._prev = seqs[b]
        return LatentPlan([int(a) for a in seqs[b, :REPLAN_EVERY]], traj[b], None, {"high": 0, "low": n * h, "total": n * h})


class TwoLevelPlanner:
    kind = "two_level"

    def __init__(self, model: VisualWorldModel, goal_pixels: np.ndarray, rng: np.random.Generator, cfg=TWO_LEVEL):
        if model.levels != 2:
            raise ValueError("the two-level planner needs a two-level model")
        self.model, self.rng, self.cfg = model, rng, cfg
        self.z_goal = model.encode(goal_pixels[None])[0]
        self.h_goal = model.encode_high(self.z_goal[None])[0]
        self._prev = None

    def plan(self, pixels: np.ndarray) -> LatentPlan:
        m, k = self.cfg["n_sequences"], self.cfg["n_macros"]
        z0 = self.model.encode(pixels[None])[0]
        h = np.repeat(self.model.encode_high(z0[None]), m, axis=0)
        seqs = self.rng.integers(0, N_ACTIONS, (m, k))
        risk, best = np.zeros(m), np.full(m, np.inf)
        first = None
        for b in range(k):
            h, p = self.model.step_high(h, seqs[:, b])
            risk = np.maximum(risk, p)
            best = np.minimum(best, _d(h, self.h_goal) + 0.05 * b)
            if b == 0:
                first = h.copy()
        hb = int(np.argmin(best + RISK_COST * risk))
        sub = self.model.subgoal(first[hb][None])[0]
        n, hh = self.cfg["n_samples"], self.cfg["horizon"]
        lseqs = sample_sequences(self.rng, n, hh, shift_warm(self._prev, hh))
        cost, traj = latent_shoot(self.model, z0, lseqs, sub, self.z_goal)
        b = int(np.argmin(cost))
        self._prev = lseqs[b]
        return LatentPlan(
            [int(a) for a in lseqs[b, :REPLAN_EVERY]], traj[b], sub, {"high": m * k, "low": n * hh, "total": m * k + n * hh}
        )


VISUAL_PLANNERS = {"one_level": OneLevelPlanner, "two_level": TwoLevelPlanner}
