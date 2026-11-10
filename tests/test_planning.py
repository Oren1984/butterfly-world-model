import json

import numpy as np
import pytest

from api.planning import learned
from api.planning.episode import run_episode
from api.planning.garden import ACTION_REPEAT, LAYOUTS, ControlledFlight, inside_obstacle
from api.planning.models import ExactModel
from api.planning.planners import BUDGET, PLANNERS, REPLAN_EVERY
from api.planning.seeds import ALL_RANGES


def _strip(ep: dict) -> dict:
    ep = dict(ep)
    ep.pop("latency")
    return ep


def test_seed_ranges_are_disjoint_within_each_experiment():
    for prefix in ("planning", "visual"):
        ranges = {k: set(v) for k, v in ALL_RANGES.items() if k.startswith(prefix)}
        names = list(ranges)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                assert not ranges[a] & ranges[b], (a, b)


def test_controlled_flight_is_seeded_and_keeps_its_wind_private():
    a, b, c = (ControlledFlight(LAYOUTS["detour"], s) for s in (3, 3, 4))
    assert np.array_equal(a.wind, b.wind) and not np.array_equal(a.wind, c.wind)
    obs = a.observe()
    assert set(vars(obs)) == {"x", "y", "vx", "vy", "wx", "wy", "decision"}  # no schedule, no future gusts


@pytest.mark.parametrize("planner", ["flat_exact", "waypoint_exact"])
def test_episodes_are_repeatable_for_a_seed(planner):
    first = run_episode("detour", planner, 5)
    again = run_episode("detour", planner, 5)
    assert _strip(first) == _strip(again)


def test_every_planner_spends_the_same_matched_budget_per_replan(tmp_path):
    weights = learned.train(seed=0, n_train=60, n_val=20, log=lambda *_: None)["weights"]
    for name in PLANNERS:
        ep = run_episode("direct", name, 1, learned_weights=weights, record=False)
        assert ep["evaluations_per_replan"] == BUDGET, name
        assert ep["evaluations_total"] == BUDGET * ep["replans"], name
        # Same replanning cadence for every arm.
        assert ep["replans"] == -(-ep["decisions"] // REPLAN_EVERY), name
        if PLANNERS[name][0].kind != "flat":
            assert ep["evaluations_high_total"] > 0  # waypoint/subgoal generation is counted


def test_unreachable_target_fails_safely():
    for planner in ("flat_exact", "waypoint_exact"):
        ep = run_episode("enclosed", planner, 0, record=False)
        assert not ep["landed"] and ep["outcome"] in ("timeout", "collided")


def test_hierarchy_beats_flat_on_the_detour_with_exact_dynamics():
    seeds = range(50_000, 50_006)  # tuning seeds, never the reported evaluation seeds
    flat = sum(run_episode("detour", "flat_exact", s, record=False)["landed"] for s in seeds)
    hier = sum(run_episode("detour", "waypoint_exact", s, record=False)["landed"] for s in seeds)
    assert hier > flat


def test_exact_model_matches_the_simulator_without_gusts():
    lay = LAYOUTS["direct"]
    w = ControlledFlight(lay, 2, wind_scale=0.0)
    model = ExactModel(lay)
    s = w.observe().state[None]
    for a in (0, 1, 2, 9):
        s, col, landed = model.step(s, np.array([a]), np.zeros(2))
        w.step(a)
        assert np.allclose(s[0], w.s[0]) and w.t % ACTION_REPEAT == 0


def test_learned_data_split_is_by_whole_trajectory():
    tr = learned.collect(range(200_000, 200_030))
    va = learned.collect(range(300_000, 300_010))
    assert not set(tr["seeds"]) & set(va["seeds"])
    assert set(np.unique(tr["short_traj"])) <= set(range(30))
    # No example starts inside an obstacle and trajectories stop at their first collision.
    for name_idx, name in enumerate(("direct", "detour")):
        sel = tr["short_layout"] == name_idx
        assert not inside_obstacle(LAYOUTS[name], tr["short_s"][sel, 0], tr["short_s"][sel, 1]).any()
    for t in np.unique(tr["short_traj"]):
        col = tr["short_collided"][tr["short_traj"] == t]
        assert col.sum() <= 1 and (col.sum() == 0 or col[-1])


def test_missing_or_incompatible_artifact_is_reported_clearly(tmp_path):
    with pytest.raises(learned.ArtifactMissing, match="train_planning"):
        learned.load(tmp_path)
    (tmp_path / "dynamics.json").write_text(json.dumps({"format": "other"}))
    (tmp_path / "dynamics.npz").write_bytes(b"")
    with pytest.raises(learned.ArtifactMissing, match="not compatible"):
        learned.load(tmp_path)
