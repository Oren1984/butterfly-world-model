"""The actual world: a small stochastic, discrete-time butterfly simulator.

Hidden state (never exposed through `observe()`): the flower the butterfly
currently intends to visit, the wind gust, and the random generator.
"""
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..contracts import Butterfly, Flower, Observation, Wind
from ..params import PARAMS, WorldParams

SCENARIO_STREAM, WORLD_STREAM, PREDICTOR_STREAM = 0, 1, 2


@dataclass
class Scenario:
    flowers: list[Flower]
    x: float
    y: float
    vx: float
    vy: float
    wind_strength: float
    wind_direction_deg: float
    randomness: float


def make_scenario(seed: int, params: WorldParams = PARAMS) -> Scenario:
    """Deterministically build a garden layout and initial conditions from a seed."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, SCENARIO_STREAM]))
    margin, min_sep = 8.0, 14.0
    n_flowers = int(rng.integers(3, 7))
    pts: list[tuple[float, float]] = []
    while len(pts) < n_flowers:
        p = (rng.uniform(margin, params.width - margin), rng.uniform(margin, params.height - margin))
        if all(math.dist(p, q) >= min_sep for q in pts):
            pts.append(p)
    flowers = [Flower(id=chr(65 + i), x=x, y=y) for i, (x, y) in enumerate(pts)]

    # Start as far from the flowers as a few random tries allow (at least 25 if possible).
    best, best_d = (params.width / 2, params.height / 2), -1.0
    for _ in range(40):
        p = (rng.uniform(5, params.width - 5), rng.uniform(5, params.height - 5))
        d = min(math.dist(p, q) for q in pts)
        if d > best_d:
            best, best_d = p, d
        if d >= 25.0:
            break
    angle = rng.uniform(0, 2 * math.pi)
    speed = 0.5 * params.cruise_speed
    return Scenario(
        flowers=flowers,
        x=best[0],
        y=best[1],
        vx=speed * math.cos(angle),
        vy=speed * math.sin(angle),
        wind_strength=float(rng.uniform(0, params.max_wind)),
        wind_direction_deg=float(rng.uniform(0, 360)),
        randomness=float(rng.uniform(0.1, 1.0)),
    )


class ButterflyWorld:
    def __init__(self, scenario: Scenario, seed: int, params: WorldParams = PARAMS):
        self.params = params
        self._rng = np.random.default_rng(np.random.SeedSequence([seed, WORLD_STREAM]))
        self.flowers = list(scenario.flowers)
        self.x, self.y = scenario.x, scenario.y
        self.vx, self.vy = scenario.vx, scenario.vy
        self.wind_strength = scenario.wind_strength
        self.wind_direction_deg = scenario.wind_direction_deg % 360
        self.randomness = scenario.randomness
        self.step_count = 0
        self.airborne_steps = 0
        self.landed_on: Optional[str] = None
        # Hidden state.
        self._gust = [0.0, 0.0]
        self._target = self._sample_target()

    # ---- public interface -------------------------------------------------

    def observe(self) -> Observation:
        return Observation(
            step=self.step_count,
            butterfly=Butterfly(x=self.x, y=self.y, vx=self.vx, vy=self.vy),
            wind=Wind(strength=self.wind_strength, direction_deg=self.wind_direction_deg),
            randomness=self.randomness,
            flowers=tuple(self.flowers),
            airborne_steps=self.airborne_steps,
            landed_on=self.landed_on,
            width=self.params.width,
            height=self.params.height,
            dt=self.params.dt,
        )

    def set_wind(self, strength: Optional[float] = None, direction_deg: Optional[float] = None) -> None:
        if strength is not None:
            self.wind_strength = min(max(strength, 0.0), self.params.max_wind)
        if direction_deg is not None:
            self.wind_direction_deg = direction_deg % 360

    def set_randomness(self, randomness: float) -> None:
        self.randomness = min(max(randomness, 0.0), 1.0)

    def move_flower(self, flower_id: str, x: float, y: float) -> None:
        for i, f in enumerate(self.flowers):
            if f.id == flower_id:
                r = f.radius
                x = min(max(x, r), self.params.width - r)
                y = min(max(y, r), self.params.height - r)
                self.flowers[i] = f.model_copy(update={"x": x, "y": y})
                return
        raise KeyError(flower_id)

    def takeoff(self) -> None:
        """Leave the current flower and start a new flight toward a different one."""
        if self.landed_on is None:
            return
        left = self.landed_on
        self.landed_on = None
        self.airborne_steps = 0
        if len(self.flowers) > 1:
            self._target = self._sample_target(exclude=left)
        angle = self._rng.uniform(0, 2 * math.pi)
        self.vx, self.vy = 2.0 * math.cos(angle), 2.0 * math.sin(angle)

    def step(self) -> None:
        if self.landed_on is not None:
            return
        p, rng, dt = self.params, self._rng, self.params.dt

        # 1. The butterfly occasionally changes its mind.
        rate = p.switch_rate_base + p.switch_rate_random * self.randomness
        if rng.random() < rate * dt:
            self._target = self._sample_target()

        # 2. Gust evolves as an OU process around the mean wind.
        sigma = p.gust_sigma_base + p.gust_sigma_rel * self.wind_strength
        k = sigma * math.sqrt(2 * dt / p.gust_tau)
        self._gust[0] += -self._gust[0] / p.gust_tau * dt + k * rng.standard_normal()
        self._gust[1] += -self._gust[1] / p.gust_tau * dt + k * rng.standard_normal()
        wdir = math.radians(self.wind_direction_deg)
        wx = self.wind_strength * math.cos(wdir) + self._gust[0]
        wy = self.wind_strength * math.sin(wdir) + self._gust[1]

        # 3. Steering toward the intended flower, plus wind and flutter.
        tf = self._flower(self._target)
        dx, dy = tf.x - self.x, tf.y - self.y
        dist = math.hypot(dx, dy)
        v_des = min(p.cruise_speed, p.arrive_gain * dist)
        ux, uy = (dx / dist, dy / dist) if dist > 1e-9 else (0.0, 0.0)
        flutter = p.flutter_accel * self.randomness
        ax = p.steer_gain * (ux * v_des - self.vx) + p.wind_coupling * wx + flutter * rng.standard_normal()
        ay = p.steer_gain * (uy * v_des - self.vy) + p.wind_coupling * wy + flutter * rng.standard_normal()
        a = math.hypot(ax, ay)
        if a > p.max_accel:
            ax, ay = ax * p.max_accel / a, ay * p.max_accel / a

        # 4. Integrate with a speed limit and reflecting walls.
        self.vx += ax * dt
        self.vy += ay * dt
        speed = math.hypot(self.vx, self.vy)
        if speed > p.max_speed:
            self.vx, self.vy = self.vx * p.max_speed / speed, self.vy * p.max_speed / speed
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x < 0 or self.x > p.width:
            self.x = min(max(self.x, 0.0), p.width)
            self.vx = -self.vx * p.wall_restitution
        if self.y < 0 or self.y > p.height:
            self.y = min(max(self.y, 0.0), p.height)
            self.vy = -self.vy * p.wall_restitution
        self.step_count += 1
        self.airborne_steps += 1

        # 5. Landing decision.
        if self.airborne_steps > p.min_airborne_steps and math.hypot(self.vx, self.vy) <= p.land_max_speed:
            zone = self._landing_zone()
            if zone is not None:
                prob = p.land_prob_target if zone == self._target else p.land_prob_other
                if rng.random() < prob:
                    self.landed_on = zone
                    self.vx = self.vy = 0.0

    # ---- hidden helpers ---------------------------------------------------

    def _flower(self, flower_id: str) -> Flower:
        return next(f for f in self.flowers if f.id == flower_id)

    def _sample_target(self, exclude: Optional[str] = None) -> str:
        cands = [f for f in self.flowers if f.id != exclude]
        d = np.array([math.hypot(f.x - self.x, f.y - self.y) for f in cands])
        w = np.exp(-(d - d.min()) / self.params.attraction_scale)
        return cands[int(self._rng.choice(len(cands), p=w / w.sum()))].id

    def _landing_zone(self) -> Optional[str]:
        """Intended flower if inside its zone, otherwise the nearest other zone we are inside."""
        best, best_d = None, math.inf
        for f in self.flowers:
            d = math.hypot(f.x - self.x, f.y - self.y)
            if d <= f.radius:
                if f.id == self._target:
                    return f.id
                if d < best_d:
                    best, best_d = f.id, d
        return best
