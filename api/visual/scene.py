"""Pixels for the visual-planning experiment (Phase 2).

The same controlled garden as Phase 1, rendered as small grayscale frames.
Planners and models only ever receive frames (and the actions they choose);
simulator coordinates are used to render, to score the outcome, and to fit the
diagnostic probe - never as model or planner input.

An optional distractor is a bright blob that wanders through the garden on its
own seeded random walk, independent of the agent's actions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..planning.garden import (
    ACTION_REPEAT,
    ACTION_VELOCITY,
    LAYOUTS,
    LEARNED_LAYOUTS,
    N_ACTIONS,
    ControlledFlight,
    Layout,
    inside_obstacle,
    substep,
    wind_schedule,
)

FRAME_H, FRAME_W = 30, 50  # 2 world units per pixel
PIXEL = 2.0
FRAME_SIZE = FRAME_H * FRAME_W
INPUT_SIZE = 2 * FRAME_SIZE  # previous + current frame (velocity is visible from the pair)
VISUAL_WIND_SCALE = 0.5
DISTRACTOR_STREAM = 21
VISUAL_DATA_STREAM = 22
MACRO_K = 8
N_MACROS = 6
MAX_DECISIONS = 100

BUTTERFLY_SIGMA, BUTTERFLY_AMP = 3.0, 1.0  # ~1.5 px: small blobs are not even linearly decodable from pixels
DISTRACTOR_SIGMA, DISTRACTOR_AMP = 3.0, 0.75  # same size, slightly dimmer: a real confound
OBSTACLE_VALUE, FLOWER_VALUE = 0.35, 0.65

_XS = (np.arange(FRAME_W) + 0.5) * PIXEL
_YS = 60.0 - (np.arange(FRAME_H) + 0.5) * PIXEL
_GX, _GY = np.meshgrid(_XS, _YS)
_BACKGROUNDS: dict[str, np.ndarray] = {}


def background(layout: Layout) -> np.ndarray:
    if layout.name not in _BACKGROUNDS:
        bg = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        bg[inside_obstacle(layout, _GX, _GY)] = OBSTACLE_VALUE
        tx, ty = layout.target
        bg[np.hypot(_GX - tx, _GY - ty) <= layout.target_radius + 0.5] = FLOWER_VALUE
        _BACKGROUNDS[layout.name] = bg
    return _BACKGROUNDS[layout.name]


def _blob(x: np.ndarray, y: np.ndarray, sigma: float, amp: float) -> np.ndarray:
    d2 = (_GX[None] - x[:, None, None]) ** 2 + (_GY[None] - y[:, None, None]) ** 2
    return (amp * np.exp(-d2 / (2 * sigma * sigma))).astype(np.float32)


def render(layout: Layout, pos: np.ndarray, distractor: np.ndarray | None = None) -> np.ndarray:
    """Frames for a batch of butterfly positions (N, 2); distractor (N, 2) with NaN = absent."""
    pos = np.atleast_2d(pos)
    out = np.maximum(background(layout)[None], _blob(pos[:, 0], pos[:, 1], BUTTERFLY_SIGMA, BUTTERFLY_AMP))
    if distractor is not None:
        distractor = np.atleast_2d(distractor)
        present = ~np.isnan(distractor[:, 0])
        if present.any():
            d = _blob(distractor[present, 0], distractor[present, 1], DISTRACTOR_SIGMA, DISTRACTOR_AMP)
            out[present] = np.maximum(out[present], d)
    return out


def goal_image(layout: Layout) -> np.ndarray:
    """The goal: the butterfly resting on the target flower, no distractor. Returned as an input pair."""
    f = render(layout, np.array([layout.target]))[0].reshape(-1)
    return np.concatenate([f, f])


class Distractor:
    """A blob on its own seeded random walk, bouncing off the garden edges."""

    def __init__(self, seed: int, width: float = 100.0, height: float = 60.0):
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, DISTRACTOR_STREAM]))
        self.w, self.h = width, height
        self.p = self.rng.uniform([10, 10], [width - 10, height - 10])
        a = self.rng.uniform(0, 2 * math.pi)
        self.v = 8.0 * np.array([math.cos(a), math.sin(a)])

    def step(self, dt: float = 0.1) -> None:
        self.v += self.rng.normal(0, 6.0, 2) * dt
        speed = float(np.hypot(*self.v))
        self.v *= min(1.0, 10.0 / max(speed, 1e-9))
        self.p = self.p + self.v * dt
        for k, lim in ((0, self.w), (1, self.h)):
            if self.p[k] < 0 or self.p[k] > lim:
                self.p[k] = min(max(self.p[k], 0.0), lim)
                self.v[k] = -self.v[k]


@dataclass
class VisualFlight:
    """One visual-planning episode. The planner sees `pixels()` only."""

    layout_name: str
    seed: int
    with_distractor: bool

    def __post_init__(self) -> None:
        self.layout = LAYOUTS[self.layout_name]
        self.world = ControlledFlight(self.layout, self.seed, wind_scale=VISUAL_WIND_SCALE, max_decisions=MAX_DECISIONS)
        self.distractor = Distractor(self.seed) if self.with_distractor else None
        self.distractor_path: list[tuple[float, float]] = []
        self._frames = [self._frame()]
        self._frames.insert(0, self._frames[0])

    def _frame(self) -> np.ndarray:
        d = None if self.distractor is None else self.distractor.p[None]
        if self.distractor is not None:
            self.distractor_path.append((float(self.distractor.p[0]), float(self.distractor.p[1])))
        return render(self.layout, self.world.s[:, :2], d)[0].reshape(-1)

    def pixels(self) -> np.ndarray:
        """Previous and current frame, flattened: the planner's entire observation."""
        return np.concatenate(self._frames[-2:])

    def step(self, action: int) -> None:
        if self.world.done:
            return
        self.world.step(action)
        if self.distractor is not None:
            for _ in range(ACTION_REPEAT):
                self.distractor.step()
        self._frames = [self._frames[-1], self._frame()]


def collect(seeds, n_macros: int = N_MACROS) -> dict[str, np.ndarray]:
    """Exploration trajectories (positions, actions, collisions, distractor) for training and probing.

    Half the trajectories carry a distractor. Frames are rendered on demand from these arrays.
    """
    seeds = list(seeds)
    n, T = len(seeds), n_macros * MACRO_K
    t_steps = T * ACTION_REPEAT + 1
    layout_idx = np.array([s % len(LEARNED_LAYOUTS) for s in seeds])
    s = np.zeros((n, 4))
    winds = np.zeros((n, t_steps, 2))
    macro = np.zeros((n, n_macros), dtype=int)
    has_d = np.zeros(n, dtype=bool)
    dists = []
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, VISUAL_DATA_STREAM]))
        layout = LAYOUTS[LEARNED_LAYOUTS[layout_idx[i]]]
        while True:
            x, y = rng.uniform(2, layout.width - 2), rng.uniform(2, layout.height - 2)
            if not inside_obstacle(layout, np.array(x), np.array(y)):
                break
        ang, spd = rng.uniform(0, 2 * np.pi), rng.uniform(0, 11)
        s[i] = (x, y, spd * np.cos(ang), spd * np.sin(ang))
        winds[i], _ = wind_schedule(rng, t_steps, VISUAL_WIND_SCALE)
        macro[i] = rng.integers(0, N_ACTIONS, n_macros)
        has_d[i] = rng.random() < 0.5
        dists.append(Distractor(seed) if has_d[i] else None)

    actions = np.repeat(macro, MACRO_K, axis=1)  # (n, T)
    pos = np.zeros((n, T + 1, 2))
    dpos = np.full((n, T + 1, 2), np.nan)
    pos[:, 0] = s[:, :2]
    collided_at = np.full(n, -1)
    alive = np.ones(n, dtype=bool)
    t = 0
    for i, d in enumerate(dists):
        if d is not None:
            dpos[i, 0] = d.p
    for k in range(T):
        hit = np.zeros(n, dtype=bool)
        for _ in range(ACTION_REPEAT):
            for li, name in enumerate(LEARNED_LAYOUTS):
                sel = (layout_idx == li) & alive & ~hit
                if sel.any():
                    nxt, col, _ = substep(LAYOUTS[name], s[sel].copy(), ACTION_VELOCITY[actions[sel, k]], winds[sel, t])
                    s[sel] = nxt
                    hit[np.flatnonzero(sel)[col]] = True
            for d in dists:
                if d is not None:
                    d.step()
            t += 1
        pos[:, k + 1] = s[:, :2]
        for i, d in enumerate(dists):
            if d is not None:
                dpos[i, k + 1] = d.p
        collided_at[hit & alive] = k + 1
        alive &= ~hit
    return {
        "seeds": np.array(seeds),
        "layout": layout_idx,
        "pos": pos,
        "actions": actions,
        "distractor": dpos,
        "collided_at": collided_at,
    }


def frames_for(data: dict, traj: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Input pairs (previous, current frame) for trajectory indices and decision indices (t >= 1)."""
    out = np.empty((len(traj), INPUT_SIZE), dtype=np.float32)
    for li, name in enumerate(LEARNED_LAYOUTS):
        sel = data["layout"][traj] == li
        if not sel.any():
            continue
        tr, tt = traj[sel], t[sel]
        prev = render(LAYOUTS[name], data["pos"][tr, tt - 1], data["distractor"][tr, tt - 1]).reshape(len(tr), -1)
        cur = render(LAYOUTS[name], data["pos"][tr, tt], data["distractor"][tr, tt]).reshape(len(tr), -1)
        out[sel] = np.concatenate([prev, cur], axis=1)
    return out
