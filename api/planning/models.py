"""Transition models used by the planners.

A transition model answers: from these states, if this action is held for one
decision (0.2 s), where does the butterfly end up, did it hit an obstacle, and
did it land? Planners only talk to this interface, never to the simulator.

- `ExactModel` reuses the documented physics with the wind observed now held
  constant (future gusts are unknown to the agent).
- `LearnedModel` is a small MLP trained from seeded observation-action
  trajectories (see `learned.py`). It also has a longer-horizon head that
  predicts the state after a macro-action (one action held for `MACRO_K`
  decisions), used by the HWM-inspired subgoal planner.
"""
from __future__ import annotations

import numpy as np

from ..learning.nn import MLP, sigmoid
from ..params import PARAMS
from .garden import ACTION_REPEAT, ACTION_VELOCITY, LEARNED_LAYOUTS, N_ACTIONS, Layout, substep

MACRO_K = 8  # decisions per macro-action (1.6 s)


class ExactModel:
    kind = "exact"

    def __init__(self, layout: Layout):
        self.layout = layout

    def step(self, s: np.ndarray, actions: np.ndarray, wind: np.ndarray):
        desired = ACTION_VELOCITY[actions]
        collided = np.zeros(len(s), dtype=bool)
        landed = np.zeros(len(s), dtype=bool)
        for _ in range(ACTION_REPEAT):
            nxt, c, lnd = substep(self.layout, s.copy(), desired, wind)
            active = ~(collided | landed)
            s = np.where(active[:, None], nxt, s)
            collided |= active & c
            landed |= active & lnd
        return s, collided.astype(float), landed


# ---- learned model --------------------------------------------------------

STATE_SCALE = np.array([PARAMS.width, PARAMS.height, PARAMS.max_speed, PARAMS.max_speed])
WIND_SCALE = PARAMS.max_wind
N_FEATURES = 4 + 2 + len(LEARNED_LAYOUTS) + N_ACTIONS


def features(s: np.ndarray, wind: np.ndarray, layout_name: str, actions: np.ndarray) -> np.ndarray:
    n = len(s)
    out = np.zeros((n, N_FEATURES), dtype=np.float32)
    out[:, :4] = s / STATE_SCALE
    out[:, 4:6] = np.broadcast_to(wind, (n, 2)) / WIND_SCALE
    out[:, 6 + LEARNED_LAYOUTS.index(layout_name)] = 1.0
    out[np.arange(n), 6 + len(LEARNED_LAYOUTS) + np.asarray(actions)] = 1.0
    return out


class LearnedModel:
    kind = "learned"

    def __init__(self, layout: Layout, short: MLP, long: MLP, norms: dict[str, np.ndarray]):
        if layout.name not in LEARNED_LAYOUTS:
            raise ValueError(f"the learned model was not trained on layout '{layout.name}'")
        self.layout, self.short, self.long, self.norms = layout, short, long, norms

    def _apply(self, net: MLP, key: str, s, actions, wind):
        out = net.forward(features(s, wind, self.layout.name, actions))
        delta = out[:, :4] * self.norms[f"{key}_std"] + self.norms[f"{key}_mean"]
        nxt = s + delta
        nxt[:, 0] = np.clip(nxt[:, 0], 0, self.layout.width)
        nxt[:, 1] = np.clip(nxt[:, 1], 0, self.layout.height)
        p_col = sigmoid(out[:, 4] - self.norms[f"{key}_logit_offset"][0])
        return nxt, p_col

    def _landed(self, s: np.ndarray) -> np.ndarray:
        tx, ty = self.layout.target
        near = np.hypot(s[:, 0] - tx, s[:, 1] - ty) <= self.layout.target_radius
        return near & (np.hypot(s[:, 2], s[:, 3]) <= PARAMS.land_max_speed)

    def step(self, s: np.ndarray, actions: np.ndarray, wind: np.ndarray):
        nxt, p_col = self._apply(self.short, "short", s, actions, wind)
        return nxt, p_col, self._landed(nxt)

    def macro(self, s: np.ndarray, actions: np.ndarray, wind: np.ndarray):
        """Longer-horizon prediction: the state after holding `actions` for MACRO_K decisions."""
        return self._apply(self.long, "long", s, actions, wind)
