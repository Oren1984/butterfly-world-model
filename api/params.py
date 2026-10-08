"""Public, documented dynamics assumptions.

These constants describe how the synthetic garden works. They are the only
thing the simulator and the prediction engine share: both may read them, but
neither shares runtime state, random streams or outcomes with the other.
"""
from dataclasses import dataclass

NO_LANDING = "no_landing_within_horizon"


@dataclass(frozen=True)
class WorldParams:
    # World geometry (world units, y pointing up) and timestep (seconds).
    width: float = 100.0
    height: float = 60.0
    dt: float = 0.1

    # Flight limits.
    cruise_speed: float = 11.0
    max_speed: float = 18.0
    max_accel: float = 45.0

    # Steering toward the intended flower: a = steer_gain * (v_desired - v).
    steer_gain: float = 2.5
    # Desired speed shrinks near the flower: min(cruise_speed, arrive_gain * distance).
    arrive_gain: float = 2.0

    # Wind pushes the butterfly: a += wind_coupling * (mean wind + gust).
    wind_coupling: float = 1.2
    max_wind: float = 12.0
    # Gust is an Ornstein-Uhlenbeck process around the mean wind.
    gust_tau: float = 2.0
    gust_sigma_base: float = 0.3
    gust_sigma_rel: float = 0.15

    # Behavioural noise, both scaled by the public "randomness" setting in [0, 1].
    flutter_accel: float = 14.0
    switch_rate_base: float = 0.02  # intent changes per second
    switch_rate_random: float = 0.25

    # Flower attraction: P(intent = f) is proportional to exp(-distance_f / attraction_scale).
    attraction_scale: float = 18.0

    # Landing rules.
    flower_radius: float = 3.0
    land_prob_target: float = 0.35  # per step, inside the intended flower's zone
    land_prob_other: float = 0.03  # per step, inside any other flower's zone
    land_max_speed: float = 9.0
    min_airborne_steps: int = 10  # no landing during the first second of a flight

    wall_restitution: float = 0.5
    default_horizon_steps: int = 60  # 6 seconds


PARAMS = WorldParams()
