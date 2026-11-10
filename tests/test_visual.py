import numpy as np
import pytest

from api.planning.garden import LAYOUTS
from api.visual import jepa
from api.visual.episode import run_visual_episode
from api.visual.planners import VISUAL_BUDGET
from api.visual.scene import INPUT_SIZE, Distractor, VisualFlight, collect, goal_image, render


@pytest.fixture(scope="module")
def tiny():
    tr, va = collect(range(400_000, 400_024)), collect(range(500_000, 500_010))
    cfg = dict(jepa.TRAIN_CONFIG, steps=12, batch=32, log_every=6)
    models = {lv: jepa.train(lv, 0, tr, va, cfg, log=lambda *_: None) for lv in (1, 2)}
    return tr, va, models


def test_frames_have_fixed_size_and_show_butterfly_and_distractor():
    lay = LAYOUTS["detour"]
    clean = render(lay, np.array([[20.0, 30.0]]))[0]
    with_d = render(lay, np.array([[20.0, 30.0]]), np.array([[70.0, 40.0]]))[0]
    assert clean.shape == (30, 50) and clean.max() <= 1.0
    assert (with_d - clean).max() > 0.5  # the distractor is visible
    assert goal_image(lay).shape == (INPUT_SIZE,)


def test_the_planner_only_receives_pixels():
    flight = VisualFlight("direct", 3, True)
    px = flight.pixels()
    assert px.shape == (INPUT_SIZE,) and px.dtype == np.float32


def test_distractor_motion_is_independent_of_actions():
    a, b = VisualFlight("direct", 9, True), VisualFlight("direct", 9, True)
    for _ in range(10):
        a.step(0)
        b.step(12)
    assert a.distractor_path == b.distractor_path
    assert np.allclose(Distractor(9).p, Distractor(9).p)


def test_visual_splits_are_whole_trajectories_and_disjoint(tiny):
    tr, va, _ = tiny
    assert not set(tr["seeds"]) & set(va["seeds"])
    pairs = jepa.valid_pairs(tr)
    assert (pairs[:, 1] >= 1).all()
    for i, c in enumerate(tr["collided_at"]):
        if c > 0:
            assert pairs[pairs[:, 0] == i, 1].max() < c  # never train past a collision


def test_smoke_training_and_episode_on_cpu(tiny):
    _, _, models = tiny
    for lv, art in models.items():
        meta = art["meta"]
        assert meta["training_curve"] and not any("non-finite" in f for f in meta["flags"])
        model = jepa.VisualWorldModel.from_state(art["weights"], lv)
        probes = {k: v for k, v in art["weights"].items() if k.startswith("probe_")}
        name = "one_level" if lv == 1 else "two_level"
        ep = run_visual_episode("direct", name, 0, True, model, probes)
        again = run_visual_episode("direct", name, 0, True, model, probes)
        assert ep["evaluations_per_replan"] == VISUAL_BUDGET
        assert ep["outcome"] == again["outcome"] and ep["path"] == again["path"]  # seeded repeatability
        assert "imagined_route" in ep["plans"][0]


def test_missing_visual_artifact_is_reported_clearly(tmp_path):
    with pytest.raises(jepa.VisualArtifactMissing, match="train_visual"):
        jepa.load(1, 0, tmp_path)
