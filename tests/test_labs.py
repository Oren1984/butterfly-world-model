from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import labs
from api.main import app
from api.planning import learned
from api.visual import jepa

client = TestClient(app)


def test_prediction_stays_the_default_mode_and_both_new_modes_are_listed():
    info = client.get("/api/modes").json()
    assert info["default"] == "prediction"
    assert info["modes"] == ["prediction", "planning", "visual"]
    assert set(info["planning"]["planners"]) == {
        "flat_exact", "waypoint_exact", "flat_learned", "waypoint_learned", "subgoal_learned"
    }


def test_original_prediction_routes_are_untouched():
    client.post("/api/world/reset", json={"seed": 7})
    assert client.get("/api/world/state").json()["status"] == "observing"
    assert client.post("/api/predict", json={"seed": 1}).status_code == 200


def test_live_exact_planning_episode_records_route_waypoints_and_flight():
    ep = client.post("/api/planning/episode", json={"layout": "detour", "planner": "waypoint_exact", "seed": 3}).json()
    assert ep["source"] == "generated live" and ep["model_version"] == "exact dynamics"
    assert ep["path"] and ep["plans"] and ep["plans"][0]["route"]
    assert any(p["waypoint"] for p in ep["plans"])
    assert ep["outcome"] in ("landed", "collided", "timeout")


def test_learned_planner_reports_a_missing_model(monkeypatch):
    def missing(*_, **__):
        raise learned.ArtifactMissing("No learned planning model found. Train one with: python -m experiments.train_planning")

    monkeypatch.setattr(learned, "load", missing)
    labs._planning_weights.cache_clear()
    r = client.post("/api/planning/episode", json={"layout": "detour", "planner": "flat_learned", "seed": 0})
    assert r.status_code == 409 and "train_planning" in r.json()["detail"]
    assert client.get("/api/modes").json()["planning"]["learned_model"]["available"] is False
    labs._planning_weights.cache_clear()


def test_visual_planner_reports_a_missing_model(monkeypatch):
    def missing(levels, seed, directory=None):
        raise jepa.VisualArtifactMissing("No trained visual model found. Train with: python -m experiments.train_visual")

    monkeypatch.setattr(labs.jepa, "load", missing)
    labs._visual_model.cache_clear()
    r = client.post("/api/visual/episode", json={"layout": "direct", "variant": "two_level", "seed": 0})
    assert r.status_code == 409 and "train_visual" in r.json()["detail"]
    labs._visual_model.cache_clear()


def test_recorded_results_are_served_or_clearly_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(labs, "RESULTS", tmp_path)
    for name in ("planning", "visual"):
        r = client.get(f"/api/{name}/results")
        assert r.status_code == 404 and "experiments.run_" in r.json()["detail"]


@pytest.mark.skipif(not (Path(labs.RESULTS) / "planning" / "metrics.json").exists(), reason="no recorded results")
def test_recorded_planning_results_report_exact_and_learned_separately():
    r = client.get("/api/planning/results").json()
    assert "exact_dynamics" in r and "learned_model" in r
    budgets = {s["evaluations_per_replan"] for g in ("exact_dynamics", "learned_model") if r[g]
               for layout in r[g].values() for s in layout.values()}
    assert budgets == {r["config"]["budget_evaluations_per_replan"]}


def test_requests_are_validated():
    assert client.post("/api/planning/episode", json={"layout": "nowhere"}).status_code == 422
    assert client.post("/api/planning/episode", json={"planner": "waypoint_exact", "seed": -1}).status_code == 422
    assert client.post("/api/visual/episode", json={"variant": "three_level"}).status_code == 422
    assert client.post("/api/planning/episode", json={"layout": "enclosed", "planner": "flat_learned"}).status_code == 422
