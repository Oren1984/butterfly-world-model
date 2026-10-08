"""Prediction engine: imagine N stochastic futures from one observation.

This is an independent, vectorised implementation of the documented dynamics
assumptions in `api.params`. It never sees the simulator object. The hidden
quantities of the actual world (intended flower, gust, random draws) are not
in the observation, so they are *inferred or sampled* here:

- intended flower: sampled per future from a belief built from distance and
  from how well the wind-corrected velocity points at each flower;
- gust: sampled from its stationary distribution;
- everything afterwards: rolled forward with the predictor's own generator.
"""
import math
from typing import Optional

import numpy as np

from ..contracts import Future, Observation, Prediction, Uncertainty
from ..params import NO_LANDING, PARAMS, WorldParams

# Weight of the heading evidence in the intent belief. Chosen on a separate
# tuning seed range by experiments/tune.py (never on the evaluation seeds).
DEFAULT_HEADING_KAPPA = 12.0


def intent_belief(obs: Observation, params: WorldParams = PARAMS, kappa: float = DEFAULT_HEADING_KAPPA) -> np.ndarray:
    """P(intended flower | observation) for each flower, in observation order."""
    fl = np.array([[f.x, f.y] for f in obs.flowers])
    pos = np.array([obs.butterfly.x, obs.butterfly.y])
    delta = fl - pos
    dist = np.hypot(delta[:, 0], delta[:, 1])
    logit = -dist / params.attraction_scale
    # Remove the steady-state wind drift so the heading reflects intent, not wind.
    wdir = math.radians(obs.wind.direction_deg)
    drift = params.wind_coupling / params.steer_gain * obs.wind.strength
    v_air = np.array([obs.butterfly.vx - drift * math.cos(wdir), obs.butterfly.vy - drift * math.sin(wdir)])
    s = float(np.hypot(*v_air))
    if s > 1e-6:
        cos = (delta @ v_air) / (np.maximum(dist, 1e-9) * s)
        logit = logit + kappa * min(1.0, s / params.cruise_speed) * cos
    w = np.exp(logit - logit.max())
    return w / w.sum()


def predict(
    obs: Observation,
    n_futures: int = 20,
    horizon_steps: int = PARAMS.default_horizon_steps,
    rng: Optional[np.random.Generator] = None,
    params: WorldParams = PARAMS,
    kappa: float = DEFAULT_HEADING_KAPPA,
    path_stride: int = 2,
) -> Prediction:
    if obs.landed_on is not None:
        raise ValueError("cannot predict a landing for a butterfly that has already landed")
    rng = rng if rng is not None else np.random.default_rng()
    p, dt, n = params, params.dt, n_futures
    ids = [f.id for f in obs.flowers]
    fl = np.array([[f.x, f.y] for f in obs.flowers])
    radius = np.array([f.radius for f in obs.flowers])
    n_fl = len(ids)

    belief = intent_belief(obs, p, kappa)
    target = rng.choice(n_fl, size=n, p=belief)
    sigma_g = p.gust_sigma_base + p.gust_sigma_rel * obs.wind.strength
    gust = rng.normal(0.0, sigma_g, size=(n, 2))
    wdir = math.radians(obs.wind.direction_deg)
    wind = obs.wind.strength * np.array([math.cos(wdir), math.sin(wdir)])

    pos = np.tile([obs.butterfly.x, obs.butterfly.y], (n, 1)).astype(float)
    vel = np.tile([obs.butterfly.vx, obs.butterfly.vy], (n, 1)).astype(float)
    alive = np.ones(n, dtype=bool)
    landed_flower = np.full(n, -1)
    landed_step = np.full(n, -1)
    history = np.empty((horizon_steps + 1, n, 2))
    history[0] = pos

    switch_p = (p.switch_rate_base + p.switch_rate_random * obs.randomness) * dt
    flutter = p.flutter_accel * obs.randomness
    gust_k = sigma_g * math.sqrt(2 * dt / p.gust_tau)
    bounds = np.array([p.width, p.height])
    rows = np.arange(n)

    for s in range(1, horizon_steps + 1):
        # Intent switches (same documented rule as the world, different random draws).
        switching = alive & (rng.random(n) < switch_p)
        if switching.any():
            d = np.hypot(*(fl[None, :, :] - pos[switching, None, :]).transpose(2, 0, 1))
            w = np.exp(-(d - d.min(axis=1, keepdims=True)) / p.attraction_scale)
            cdf = np.cumsum(w / w.sum(axis=1, keepdims=True), axis=1)
            pick = (rng.random((len(d), 1)) > cdf).sum(axis=1)
            target[switching] = np.minimum(pick, n_fl - 1)

        gust += -gust / p.gust_tau * dt + gust_k * rng.standard_normal((n, 2))

        delta = fl[target] - pos
        dist = np.hypot(delta[:, 0], delta[:, 1])
        v_des = np.minimum(p.cruise_speed, p.arrive_gain * dist)
        unit = delta / np.maximum(dist, 1e-9)[:, None]
        acc = (
            p.steer_gain * (unit * v_des[:, None] - vel)
            + p.wind_coupling * (wind + gust)
            + flutter * rng.standard_normal((n, 2))
        )
        a = np.hypot(acc[:, 0], acc[:, 1])
        acc *= np.minimum(1.0, p.max_accel / np.maximum(a, 1e-9))[:, None]

        new_vel = vel + acc * dt
        speed = np.hypot(new_vel[:, 0], new_vel[:, 1])
        new_vel *= np.minimum(1.0, p.max_speed / np.maximum(speed, 1e-9))[:, None]
        new_pos = pos + new_vel * dt
        out = (new_pos < 0) | (new_pos > bounds)
        new_pos = np.clip(new_pos, 0.0, bounds)
        new_vel = np.where(out, -new_vel * p.wall_restitution, new_vel)

        # Futures that already landed stay where they landed.
        pos = np.where(alive[:, None], new_pos, pos)
        vel = np.where(alive[:, None], new_vel, vel)
        history[s] = pos

        if obs.airborne_steps + s > p.min_airborne_steps:
            d_all = np.hypot(pos[:, None, 0] - fl[None, :, 0], pos[:, None, 1] - fl[None, :, 1])
            inside = d_all <= radius[None, :]
            in_target = inside[rows, target]
            nearest = np.where(inside, d_all, np.inf).argmin(axis=1)
            zone = np.where(in_target, target, nearest)
            prob = np.where(in_target, p.land_prob_target, p.land_prob_other)
            slow = np.hypot(vel[:, 0], vel[:, 1]) <= p.land_max_speed
            lands = alive & inside.any(axis=1) & slow & (rng.random(n) < prob)
            landed_flower[lands] = zone[lands]
            landed_step[lands] = s
            alive &= ~lands
            if not alive.any():
                history[s + 1 :] = pos
                break

    # ---- summarise the imagined futures -----------------------------------
    outcomes = ids + [NO_LANDING]
    counts = np.bincount(np.where(landed_flower >= 0, landed_flower, n_fl), minlength=n_fl + 1)
    probs = counts / n
    top = int(probs.argmax())  # ties: first in [flowers..., no_landing] order
    on_top = landed_flower == top
    if top < n_fl:
        landing_pts = pos[on_top]
        mean_pt = landing_pts.mean(axis=0)
        predicted_position = (float(mean_pt[0]), float(mean_pt[1]))
        expected_step = float(landed_step[on_top].mean())
        landing_spread = float(np.sqrt(((landing_pts - mean_pt) ** 2).sum(axis=1).mean()))
    else:
        predicted_position = expected_step = landing_spread = None

    nz = probs[probs > 0]
    entropy = float(-(nz * np.log(nz)).sum() / math.log(len(outcomes)))
    sample_idx = list(range(0, horizon_steps + 1, path_stride))
    cloud = history[sample_idx]
    spread = np.sqrt(((cloud - cloud.mean(axis=1, keepdims=True)) ** 2).sum(axis=2).mean(axis=1))

    futures = []
    for i in range(n):
        end = int(landed_step[i]) if landed_flower[i] >= 0 else horizon_steps
        idx = list(range(0, end, path_stride)) + [end]
        landed = landed_flower[i] >= 0
        futures.append(
            Future(
                outcome=ids[landed_flower[i]] if landed else NO_LANDING,
                landing_step=end if landed else None,
                landing_position=(float(pos[i, 0]), float(pos[i, 1])) if landed else None,
                path=tuple((round(float(x), 2), round(float(y), 2)) for x, y in history[idx, i]),
            )
        )

    return Prediction(
        observation_step=obs.step,
        n_futures=n,
        horizon_steps=horizon_steps,
        path_stride=path_stride,
        probabilities={o: float(pr) for o, pr in zip(outcomes, probs)},
        top_outcome=outcomes[top],
        confidence=float(probs[top]),
        predicted_landing_position=predicted_position,
        expected_landing_step=expected_step,
        uncertainty=Uncertainty(
            entropy=max(entropy, 0.0),
            landing_spread=landing_spread,
            spread_over_time=tuple(round(float(v), 3) for v in spread),
            intent_belief={i: float(b) for i, b in zip(ids, belief)},
        ),
        futures=tuple(futures),
    )
