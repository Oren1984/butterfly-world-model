"""Controlled flight garden for the V1.1 planning experiments.

This is a separate, opt-in simulator. In the original prediction mode the
butterfly chooses where to go and the model only watches. Here an agent steers
the butterfly with a finite set of actions and tries to land on one designated
target flower while a seeded wind schedule pushes it around.

Separation of concerns:
- `ControlledFlight` is the actual world. It owns the true state and the full
  wind schedule (mean wind plus a gust process). Planners never see it.
- `substep` is the documented physics. The simulator uses it with the true
  wind; the exact transition model (`models.ExactModel`) reuses it with the
  wind the agent can currently observe, held constant, because the agent
  cannot see future gusts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..params import PARAMS

FLIGHT_STREAM = 11  # random stream id for controlled-flight episodes (prediction mode uses 0-2)

DT = PARAMS.dt
ACTION_REPEAT = 2  # simulator steps per decision: one steering action is held for 0.2 s
MAX_DECISIONS = 120  # 24 s episode limit
HEADINGS = 8
SPEEDS = (11.0, 4.0)  # cruise, slow (landing needs speed <= PARAMS.land_max_speed)
N_ACTIONS = HEADINGS * len(SPEEDS)
MAX_MEAN_WIND = 6.0


def _action_table() -> np.ndarray:
    out = []
    for speed in SPEEDS:
        for h in range(HEADINGS):
            a = 2 * math.pi * h / HEADINGS
            out.append((speed * math.cos(a), speed * math.sin(a)))
    return np.array(out, dtype=np.float64)


ACTION_VELOCITY = _action_table()  # (N_ACTIONS, 2): desired velocity for each discrete action


def action_label(a: int) -> str:
    names = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
    return f"{'cruise' if a < HEADINGS else 'slow'} {names[a % HEADINGS]}"


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class Layout:
    name: str
    description: str
    start: tuple[float, float]
    target: tuple[float, float]
    obstacles: tuple[Rect, ...] = ()
    reachable: bool = True
    target_radius: float = PARAMS.flower_radius
    width: float = PARAMS.width
    height: float = PARAMS.height

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "start": list(self.start),
            "target": {"x": self.target[0], "y": self.target[1], "radius": self.target_radius},
            "obstacles": [r.__dict__ for r in self.obstacles],
            "reachable": self.reachable,
            "width": self.width,
            "height": self.height,
        }


LAYOUTS: dict[str, Layout] = {
    "direct": Layout(
        "direct",
        "Simple direct route: a bush sits off to the side; flying straight to the flower works.",
        start=(15.0, 30.0),
        target=(85.0, 30.0),
        obstacles=(Rect(44.0, 46.0, 56.0, 60.0),),
    ),
    "detour": Layout(
        "detour",
        "Detour route: a hedge blocks the straight line; the gaps are at the top and bottom edges.",
        start=(15.0, 30.0),
        target=(85.0, 30.0),
        obstacles=(Rect(47.0, 9.0, 53.0, 51.0),),
    ),
    "enclosed": Layout(
        "enclosed",
        "Unreachable target: the flower is walled in. Used to check that planners fail safely.",
        start=(15.0, 30.0),
        target=(80.0, 30.0),
        obstacles=(
            Rect(68.0, 18.0, 92.0, 21.0),
            Rect(68.0, 39.0, 92.0, 42.0),
            Rect(68.0, 21.0, 71.0, 39.0),
            Rect(89.0, 21.0, 92.0, 39.0),
        ),
        reachable=False,
    ),
}
LEARNED_LAYOUTS = ("direct", "detour")  # layouts the learned models are trained on (one-hot input)


def inside_obstacle(layout: Layout, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    hit = np.zeros(np.shape(x), dtype=bool)
    for r in layout.obstacles:
        hit |= (x >= r.x0) & (x <= r.x1) & (y >= r.y0) & (y <= r.y1)
    return hit


def substep(layout: Layout, s: np.ndarray, desired: np.ndarray, wind: np.ndarray):
    """One simulator step for a batch of states.

    s: (N, 4) = x, y, vx, vy.  desired: (N, 2) desired velocity.  wind: (N, 2) or (2,).
    Returns new states, a collision mask and a landing mask.
    """
    p = PARAMS
    pos, vel = s[:, :2], s[:, 2:]
    acc = p.steer_gain * (desired - vel) + p.wind_coupling * wind
    norm = np.hypot(acc[:, 0], acc[:, 1])[:, None]
    acc = acc * np.minimum(1.0, p.max_accel / np.maximum(norm, 1e-9))
    vel = vel + acc * DT
    speed = np.hypot(vel[:, 0], vel[:, 1])[:, None]
    vel = vel * np.minimum(1.0, p.max_speed / np.maximum(speed, 1e-9))
    pos = pos + vel * DT
    # Garden edges reflect, as in the prediction-mode simulator.
    for k, limit in ((0, layout.width), (1, layout.height)):
        out = (pos[:, k] < 0) | (pos[:, k] > limit)
        pos[:, k] = np.clip(pos[:, k], 0.0, limit)
        vel[:, k] = np.where(out, -vel[:, k] * p.wall_restitution, vel[:, k])
    collided = inside_obstacle(layout, pos[:, 0], pos[:, 1])
    tx, ty = layout.target
    near = np.hypot(pos[:, 0] - tx, pos[:, 1] - ty) <= layout.target_radius
    landed = near & (np.hypot(vel[:, 0], vel[:, 1]) <= p.land_max_speed) & ~collided
    return np.concatenate([pos, vel], axis=1), collided, landed


def wind_schedule(rng: np.random.Generator, n_steps: int, wind_scale: float = 1.0) -> tuple[np.ndarray, dict]:
    """Mean wind plus an Ornstein-Uhlenbeck gust, one vector per simulator step."""
    p = PARAMS
    strength = float(rng.uniform(0.0, MAX_MEAN_WIND * wind_scale))
    direction = float(rng.uniform(0.0, 360.0))
    mean = np.array([strength * math.cos(math.radians(direction)), strength * math.sin(math.radians(direction))])
    sigma = (p.gust_sigma_base + p.gust_sigma_rel * strength) * wind_scale
    k = sigma * math.sqrt(2 * DT / p.gust_tau)
    gust = np.zeros(2)
    out = np.empty((n_steps, 2))
    for t in range(n_steps):
        gust += -gust / p.gust_tau * DT + k * rng.standard_normal(2)
        out[t] = mean + gust
    return out, {"mean_strength": strength, "mean_direction_deg": direction}


@dataclass(frozen=True)
class FlightObservation:
    """What a state-based planner may see: the butterfly, the wind right now, the target."""

    x: float
    y: float
    vx: float
    vy: float
    wx: float
    wy: float
    decision: int

    @property
    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.vx, self.vy])

    @property
    def wind(self) -> np.ndarray:
        return np.array([self.wx, self.wy])


@dataclass
class ControlledFlight:
    """The actual world for one goal-directed episode. Deterministic given (layout, seed)."""

    layout: Layout
    seed: int
    wind_scale: float = 1.0
    max_decisions: int = MAX_DECISIONS
    start_jitter: float = 3.0
    status: str = field(init=False, default="flying")  # flying | landed | collided | timeout

    def __post_init__(self) -> None:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, FLIGHT_STREAM]))
        sx, sy = self.layout.start
        for _ in range(50):
            x = sx + rng.uniform(-self.start_jitter, self.start_jitter)
            y = sy + rng.uniform(-self.start_jitter, self.start_jitter)
            if not inside_obstacle(self.layout, np.array(x), np.array(y)):
                break
        angle = rng.uniform(0, 2 * math.pi)
        self.s = np.array([[x, y, 2.0 * math.cos(angle), 2.0 * math.sin(angle)]])
        self.wind, self.wind_info = wind_schedule(rng, self.max_decisions * ACTION_REPEAT + 1, self.wind_scale)
        self.t = 0  # simulator steps
        self.decision = 0
        self.path: list[tuple[float, float]] = [(float(x), float(y))]

    def observe(self) -> FlightObservation:
        x, y, vx, vy = (float(v) for v in self.s[0])
        wx, wy = (float(v) for v in self.wind[self.t])
        return FlightObservation(x, y, vx, vy, wx, wy, self.decision)

    @property
    def done(self) -> bool:
        return self.status != "flying"

    def step(self, action: int) -> None:
        if self.done:
            return
        desired = ACTION_VELOCITY[int(action)][None, :]
        for _ in range(ACTION_REPEAT):
            self.s, collided, landed = substep(self.layout, self.s, desired, self.wind[self.t])
            self.t += 1
            self.path.append((float(self.s[0, 0]), float(self.s[0, 1])))
            if collided[0]:
                self.status = "collided"
                break
            if landed[0]:
                self.status = "landed"
                self.s[0, 2:] = 0.0
                break
        self.decision += 1
        if not self.done and self.decision >= self.max_decisions:
            self.status = "timeout"
