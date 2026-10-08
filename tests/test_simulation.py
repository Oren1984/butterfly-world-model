import math

import numpy as np
import pytest

from api.contracts import Flower, Observation
from api.params import PARAMS
from api.simulation.world import ButterflyWorld, Scenario, make_scenario


def run(world, steps):
    out = []
    for _ in range(steps):
        world.step()
        out.append((world.x, world.y, world.vx, world.vy, world.landed_on))
    return out


def calm_scenario(**overrides):
    base = dict(
        flowers=[Flower(id="A", x=80, y=30)],
        x=20.0, y=30.0, vx=0.0, vy=0.0,
        wind_strength=0.0, wind_direction_deg=0.0, randomness=0.0,
    )
    base.update(overrides)
    return Scenario(**base)


def test_same_seed_is_deterministic_and_different_seed_is_not():
    a = run(ButterflyWorld(make_scenario(3), 3), 80)
    b = run(ButterflyWorld(make_scenario(3), 3), 80)
    c = run(ButterflyWorld(make_scenario(3), 4), 80)
    assert a == b
    assert a != c


def test_scenario_generation_is_deterministic_and_valid():
    s1, s2 = make_scenario(11), make_scenario(11)
    assert s1 == s2
    assert 3 <= len(s1.flowers) <= 6
    for f in s1.flowers:
        assert 0 < f.x < PARAMS.width and 0 < f.y < PARAMS.height


@pytest.mark.parametrize("seed", range(8))
def test_physical_constraints_hold(seed):
    """Stays in bounds, never exceeds max speed, and never jumps further than max_speed * dt."""
    scenario = make_scenario(seed)
    scenario.wind_strength, scenario.randomness = PARAMS.max_wind, 1.0  # harshest conditions
    world = ButterflyWorld(scenario, seed)
    px, py = world.x, world.y
    for _ in range(300):
        if world.landed_on:
            world.takeoff()
        world.step()
        assert 0 <= world.x <= PARAMS.width and 0 <= world.y <= PARAMS.height
        assert math.hypot(world.vx, world.vy) <= PARAMS.max_speed + 1e-9
        assert math.hypot(world.x - px, world.y - py) <= PARAMS.max_speed * PARAMS.dt + 1e-9
        px, py = world.x, world.y


def test_acceleration_is_bounded():
    scenario = calm_scenario(wind_strength=PARAMS.max_wind, randomness=1.0)
    world = ButterflyWorld(scenario, 0)
    vx, vy = world.vx, world.vy
    for _ in range(100):
        world.step()
        if world.landed_on:
            break
        # A wall bounce reverses velocity, so only check steps away from the walls.
        if 1 < world.x < PARAMS.width - 1 and 1 < world.y < PARAMS.height - 1:
            assert math.hypot(world.vx - vx, world.vy - vy) <= PARAMS.max_accel * PARAMS.dt + 1e-9
        vx, vy = world.vx, world.vy


def test_wind_pushes_the_butterfly_downwind():
    """Identical seeds, flower straight ahead: an upward wind must lift the flight path."""
    calm = ButterflyWorld(calm_scenario(), 5)
    windy = ButterflyWorld(calm_scenario(wind_strength=8.0, wind_direction_deg=90.0), 5)
    run(calm, 25)
    run(windy, 25)
    assert abs(calm.y - 30.0) < 0.5
    assert windy.y > calm.y + 3.0


def test_butterfly_lands_on_its_only_flower_and_stops():
    world = ButterflyWorld(calm_scenario(), 1)
    history = run(world, 200)
    assert world.landed_on == "A"
    assert math.hypot(world.x - 80, world.y - 30) <= PARAMS.flower_radius
    assert (world.vx, world.vy) == (0.0, 0.0)
    landed_at = next(i for i, h in enumerate(history) if h[4])
    assert history[landed_at][:2] == history[-1][:2]  # no movement after landing


def test_no_landing_during_first_second_even_on_a_flower():
    world = ButterflyWorld(calm_scenario(x=80.0, y=30.0), 2)
    for _ in range(PARAMS.min_airborne_steps):
        world.step()
        assert world.landed_on is None
    run(world, 50)
    assert world.landed_on == "A"


def test_takeoff_starts_a_new_flight():
    scenario = calm_scenario(flowers=[Flower(id="A", x=80, y=30), Flower(id="B", x=30, y=40)])
    world = ButterflyWorld(scenario, 1)
    run(world, 300)
    first = world.landed_on
    assert first is not None
    world.takeoff()
    assert world.landed_on is None and world.airborne_steps == 0
    run(world, 300)
    assert world.landed_on is not None


def test_observation_exposes_no_hidden_state():
    world = ButterflyWorld(make_scenario(0), 0)
    dumped = world.observe().model_dump()
    assert set(dumped) == set(Observation.model_fields)
    text = str(dumped).lower()
    for hidden in ("target", "gust", "rng", "intent", "seed"):
        assert hidden not in text


def test_moving_a_flower_is_clamped_to_the_world():
    world = ButterflyWorld(make_scenario(0), 0)
    world.move_flower("A", -50, 500)
    f = world.observe().flowers[0]
    assert (f.x, f.y) == (f.radius, PARAMS.height - f.radius)
    with pytest.raises(KeyError):
        world.move_flower("Z", 1, 1)
