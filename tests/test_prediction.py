import inspect
import math

import numpy as np
import pytest

from api.contracts import Butterfly, Flower, Observation, Prediction, Wind
from api.params import NO_LANDING, PARAMS
from api.prediction import baseline, predictor
from api.prediction.predictor import predict
from api.simulation.world import ButterflyWorld, make_scenario


def obs_at(seed=0, warmup=PARAMS.min_airborne_steps):
    world = ButterflyWorld(make_scenario(seed), seed)
    for _ in range(warmup):
        world.step()
    return world, world.observe()


def simple_obs(**kw):
    base = dict(
        step=10,
        butterfly=Butterfly(x=20, y=30, vx=8, vy=0),
        wind=Wind(strength=0, direction_deg=0),
        randomness=0.2,
        flowers=(Flower(id="A", x=60, y=30), Flower(id="B", x=20, y=55)),
        airborne_steps=10,
    )
    base.update(kw)
    return Observation(**base)


def rng(seed=0):
    return np.random.default_rng(seed)


def test_default_is_exactly_20_futures():
    assert inspect.signature(predict).parameters["n_futures"].default == 20
    pr = predict(simple_obs(), rng=rng())
    assert isinstance(pr, Prediction)
    assert pr.n_futures == 20 and len(pr.futures) == 20


def test_output_schema_is_visualisation_ready():
    obs = simple_obs()
    pr = predict(obs, rng=rng())
    ids = {f.id for f in obs.flowers}
    assert set(pr.probabilities) == ids | {NO_LANDING}
    assert pr.top_outcome in pr.probabilities
    assert pr.confidence == pr.probabilities[pr.top_outcome] == max(pr.probabilities.values())
    assert 0 <= pr.uncertainty.entropy <= 1
    assert set(pr.uncertainty.intent_belief) == ids
    assert len(pr.uncertainty.spread_over_time) == pr.horizon_steps // pr.path_stride + 1
    for fut in pr.futures:
        assert fut.path[0] == (obs.butterfly.x, obs.butterfly.y)
        assert fut.outcome in pr.probabilities
        for x, y in fut.path:
            assert 0 <= x <= obs.width and 0 <= y <= obs.height
        if fut.outcome == NO_LANDING:
            assert fut.landing_step is None and fut.landing_position is None
        else:
            flower = next(f for f in obs.flowers if f.id == fut.outcome)
            assert math.dist(fut.landing_position, (flower.x, flower.y)) <= flower.radius + 1e-9
            assert 1 <= fut.landing_step <= pr.horizon_steps


@pytest.mark.parametrize("seed", range(6))
def test_probabilities_are_counts_and_sum_to_one(seed):
    _, obs = obs_at(seed)
    pr = predict(obs, n_futures=20, rng=rng(seed))
    assert sum(pr.probabilities.values()) == pytest.approx(1.0)
    for outcome, p in pr.probabilities.items():
        assert p == pytest.approx(sum(f.outcome == outcome for f in pr.futures) / 20)


def test_no_landing_mass_is_kept_not_renormalised():
    """With a horizon too short to reach any flower, all mass must sit on no-landing."""
    pr = predict(simple_obs(), horizon_steps=5, rng=rng())
    assert pr.probabilities[NO_LANDING] == 1.0
    assert pr.top_outcome == NO_LANDING
    assert pr.predicted_landing_position is None and pr.expected_landing_step is None
    assert all(f.outcome == NO_LANDING for f in pr.futures)
    # A partial case: some futures land, the rest stay explicitly unlanded.
    mixed = predict(simple_obs(), horizon_steps=38, n_futures=200, rng=rng(1))
    assert 0 < mixed.probabilities[NO_LANDING] < 1
    assert sum(mixed.probabilities.values()) == pytest.approx(1.0)


def test_same_observation_and_seed_gives_identical_prediction():
    obs = simple_obs()
    assert predict(obs, rng=rng(5)) == predict(obs, rng=rng(5))
    assert predict(obs, rng=rng(5)) != predict(obs, rng=rng(6))


def test_futures_are_distinct_samples():
    pr = predict(simple_obs(randomness=0.8), rng=rng())
    assert len({f.path for f in pr.futures}) == 20


def test_predictor_cannot_take_the_simulator():
    """The only world information accepted is the Observation contract."""
    params = inspect.signature(predict).parameters
    assert params["obs"].annotation is Observation
    source = inspect.getsource(predictor)
    assert "ButterflyWorld" not in source and "import" not in "".join(
        line for line in source.splitlines() if "simulation" in line
    )


def test_predicting_does_not_disturb_the_actual_world():
    """RNG independence: the world's trajectory is identical with or without predictions."""
    plain = ButterflyWorld(make_scenario(9), 9)
    watched = ButterflyWorld(make_scenario(9), 9)
    for _ in range(80):
        plain.step()
        if watched.landed_on is None:
            predict(watched.observe(), rng=rng(123))
        watched.step()
        assert (plain.x, plain.y, plain.landed_on) == (watched.x, watched.y, watched.landed_on)


def test_prediction_depends_only_on_the_observation_not_hidden_state():
    """Two worlds with the same observable state but different hidden state and futures."""
    w1, obs = obs_at(4)
    w2, _ = obs_at(4)
    w2._target = next(f.id for f in w2.flowers if f.id != w1._target)  # different hidden intent
    w2._gust = [5.0, -5.0]
    w2._rng = np.random.default_rng(999)
    assert w1.observe() == w2.observe()
    assert predict(w1.observe(), rng=rng(1)) == predict(w2.observe(), rng=rng(1))
    for _ in range(60):
        w1.step()
        w2.step()
    assert (w1.x, w1.y) != (w2.x, w2.y)  # the actual futures really did differ


def test_predictor_is_not_artificially_perfect():
    """Imagined futures must not replay the actual future."""
    world, obs = obs_at(2)
    pr = predict(obs, rng=rng(0))
    actual = []
    for _ in range(pr.horizon_steps):
        world.step()
        actual.append((round(world.x, 2), round(world.y, 2)))
    for fut in pr.futures:
        assert fut.path[1:4] != tuple(actual[1:6:2])


def test_prediction_changes_when_wind_changes():
    calm = predict(simple_obs(), n_futures=200, rng=rng(3))
    gale = predict(simple_obs(wind=Wind(strength=12, direction_deg=180)), n_futures=200, rng=rng(3))
    assert calm.probabilities != gale.probabilities
    assert calm.futures[0].path != gale.futures[0].path
    # A strong headwind makes the flower straight ahead harder to reach in time.
    assert gale.probabilities["A"] < calm.probabilities["A"]


def test_prediction_follows_a_moved_flower():
    near = predict(simple_obs(), n_futures=200, rng=rng(3))
    moved = simple_obs(flowers=(Flower(id="A", x=95, y=5), Flower(id="B", x=30, y=32)))
    after = predict(moved, n_futures=200, rng=rng(3))
    assert near.top_outcome == "A" and after.top_outcome == "B"


def test_more_randomness_means_more_spread():
    low = predict(simple_obs(randomness=0.0), n_futures=200, rng=rng(1))
    high = predict(simple_obs(randomness=1.0), n_futures=200, rng=rng(1))
    assert high.uncertainty.spread_over_time[8] > low.uncertainty.spread_over_time[8]


def test_landed_observation_is_rejected():
    with pytest.raises(ValueError):
        predict(simple_obs(landed_on="A"), rng=rng())


def test_intent_belief_favours_the_flower_ahead():
    belief = predictor.intent_belief(simple_obs())
    assert belief.sum() == pytest.approx(1.0)
    assert belief[0] > belief[1]  # flying straight at A


# ---- baselines ---------------------------------------------------------------


def test_constant_velocity_hits_the_flower_straight_ahead():
    probs, pos = baseline.constant_velocity(simple_obs())
    assert probs == {"A": 1.0, "B": 0.0, NO_LANDING: 0.0}
    assert pos == pytest.approx((57.0, 30.0))  # entry point of A's landing zone


def test_constant_velocity_reports_no_landing_when_it_misses_or_runs_out_of_time():
    away = simple_obs(butterfly=Butterfly(x=20, y=30, vx=-8, vy=0))
    probs, pos = baseline.constant_velocity(away)
    assert probs[NO_LANDING] == 1.0 and pos is None
    probs, _ = baseline.constant_velocity(simple_obs(), horizon_steps=10)  # 8 units travelled < 37 needed
    assert probs[NO_LANDING] == 1.0
    still = simple_obs(butterfly=Butterfly(x=20, y=30, vx=0, vy=0))
    assert baseline.constant_velocity(still)[0][NO_LANDING] == 1.0


def test_nearest_flower_baseline():
    probs, pos = baseline.nearest_flower(simple_obs())
    assert probs == {"A": 0.0, "B": 1.0, NO_LANDING: 0.0}
    assert pos == (20, 55)
