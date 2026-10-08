"""Simple, untuned baseline predictors. Both return a one-hot distribution."""
import math
from typing import Optional

from ..contracts import Observation
from ..params import NO_LANDING, PARAMS

BaselineResult = tuple[dict[str, float], Optional[tuple[float, float]]]


def _one_hot(obs: Observation, winner: str) -> dict[str, float]:
    return {o: float(o == winner) for o in [f.id for f in obs.flowers] + [NO_LANDING]}


def constant_velocity(obs: Observation, horizon_steps: int = PARAMS.default_horizon_steps) -> BaselineResult:
    """Extend the current velocity in a straight line; the first landing zone it crosses wins."""
    b = obs.butterfly
    t_max = horizon_steps * obs.dt
    speed2 = b.vx * b.vx + b.vy * b.vy
    best_t, best_id = math.inf, NO_LANDING
    for f in obs.flowers:
        ox, oy = b.x - f.x, b.y - f.y
        c = ox * ox + oy * oy - f.radius * f.radius
        if c <= 0:
            t = 0.0  # already inside the zone
        elif speed2 < 1e-12:
            continue
        else:
            half_b = ox * b.vx + oy * b.vy
            disc = half_b * half_b - speed2 * c
            if disc < 0 or half_b >= 0:
                continue
            t = (-half_b - math.sqrt(disc)) / speed2
        x, y = b.x + b.vx * t, b.y + b.vy * t
        if t <= t_max and 0 <= x <= obs.width and 0 <= y <= obs.height and t < best_t:
            best_t, best_id = t, f.id
    if best_id == NO_LANDING:
        return _one_hot(obs, NO_LANDING), None
    return _one_hot(obs, best_id), (b.x + b.vx * best_t, b.y + b.vy * best_t)


def nearest_flower(obs: Observation) -> BaselineResult:
    """Always predict the closest flower, landing at its centre."""
    b = obs.butterfly
    f = min(obs.flowers, key=lambda f: math.hypot(f.x - b.x, f.y - b.y))
    return _one_hot(obs, f.id), (f.x, f.y)
