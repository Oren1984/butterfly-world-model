import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.params import NO_LANDING, PARAMS

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_world():
    client.post("/api/world/reset", json={"seed": 7})


def fly_until(status_set, max_steps=400):
    for _ in range(max_steps):
        state = client.post("/api/world/step", json={"steps": 1}).json()
        if state["status"] in status_set:
            return state
    raise AssertionError(f"never reached {status_set}")


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_reset_is_deterministic_and_new_scenario_changes_the_garden():
    a = client.post("/api/world/reset", json={"seed": 7}).json()
    run_a = [client.post("/api/world/step", json={"steps": 5}).json()["observation"] for _ in range(4)]
    b = client.post("/api/world/reset", json={}).json()  # same seed again
    run_b = [client.post("/api/world/step", json={"steps": 5}).json()["observation"] for _ in range(4)]
    assert a["observation"] == b["observation"] and run_a == run_b
    c = client.post("/api/world/reset", json={"new_scenario": True}).json()
    assert c["seed"] == 8 and c["observation"]["flowers"] != a["observation"]["flowers"]


def test_state_hides_simulator_internals():
    text = client.get("/api/world/state").text.lower()
    for hidden in ("target", "gust", "rng", "intent"):
        assert hidden not in text


def test_step_returns_the_path_travelled():
    state = client.post("/api/world/step", json={"steps": 3}).json()
    assert len(state["path"]) == 3 and state["observation"]["step"] == 3
    last = state["path"][-1]
    assert last == [state["observation"]["butterfly"]["x"], state["observation"]["butterfly"]["y"]]


def test_full_cycle_commits_then_scores_then_continues():
    state = client.get("/api/world/state").json()
    assert state["status"] == "observing" and state["committed"] is None
    assert client.get("/api/world/prediction").status_code == 404

    state = fly_until({"flying"})
    assert state["observation"]["airborne_steps"] == PARAMS.min_airborne_steps
    committed = client.get("/api/world/prediction").json()["prediction"]
    assert len(committed["futures"]) == 20
    assert committed["top_outcome"] == state["committed"]["top_outcome"]

    # Live re-imagining must never alter the frozen, scored prediction.
    live = client.post("/api/predict", json={}).json()
    assert live["config_version"] == state["config_version"]
    assert client.get("/api/world/prediction").json()["prediction"] == committed

    done = fly_until({"landed", "no_landing"})
    result = done["result"]
    assert result["predicted"] == committed["top_outcome"]
    assert result["confidence"] == committed["confidence"]
    assert result["correct"] == (result["predicted"] == result["actual"])
    if done["status"] == "landed":
        assert result["actual"] == done["observation"]["landed_on"]
    else:
        assert result["actual"] == NO_LANDING and result["position_error"] is None
    assert done["session"]["cycles"] == 1

    # Stepping a finished flight does nothing; next starts another cycle.
    assert client.post("/api/world/step", json={"steps": 5}).json()["path"] == []
    again = client.post("/api/world/next").json()
    assert again["status"] == "observing" and again["result"] is None
    assert fly_until({"flying"})["committed"] is not None


def test_config_change_invalidates_and_recommits_the_prediction():
    before = fly_until({"flying"})
    old = client.get("/api/world/prediction").json()
    after = client.post("/api/world/config", json={"wind_strength": 12, "wind_direction_deg": 200}).json()
    assert after["config_version"] > before["config_version"]
    assert after["observation"]["wind"] == {"strength": 12, "direction_deg": 200}
    new = client.get("/api/world/prediction").json()
    assert new["prediction"]["futures"] != old["prediction"]["futures"]
    assert new["prediction"]["observation_step"] == after["observation"]["step"]
    assert after["steps_remaining"] == after["horizon_steps"]


def test_moving_a_flower_updates_observation_and_prediction():
    before = fly_until({"flying"})
    b = before["observation"]["butterfly"]
    target = before["observation"]["flowers"][0]["id"]
    x = min(max(b["x"] + 6, 3), 97)
    y = min(max(b["y"], 3), 57)
    after = client.post("/api/world/config", json={"flowers": [{"id": target, "x": x, "y": y}]}).json()
    moved = next(f for f in after["observation"]["flowers"] if f["id"] == target)
    assert (moved["x"], moved["y"]) == (x, y)
    assert after["committed"]["probabilities"] != before["committed"]["probabilities"] or \
        after["config_version"] > before["config_version"]
    pred = client.get("/api/world/prediction").json()["prediction"]
    assert pred["observation_step"] == after["observation"]["step"]


def test_predict_with_explicit_observation_is_deterministic():
    obs = client.get("/api/world/state").json()["observation"]
    body = {"observation": obs, "seed": 42, "horizon_steps": 60}
    a = client.post("/api/predict", json=body).json()
    b = client.post("/api/predict", json=body).json()
    assert a == b and a["config_version"] is None
    probs = a["prediction"]["probabilities"]
    assert NO_LANDING in probs and sum(probs.values()) == pytest.approx(1.0)
    assert len(client.post("/api/predict", json={**body, "n_futures": 50}).json()["prediction"]["futures"]) == 50


def test_predict_after_landing_is_a_conflict():
    done = fly_until({"landed", "no_landing"})
    for _ in range(10):
        if done["status"] == "landed":
            break
        client.post("/api/world/next")
        done = fly_until({"landed", "no_landing"})
    assert done["status"] == "landed"
    assert client.post("/api/predict", json={}).status_code == 409


def test_evaluate_endpoint():
    m = client.post("/api/evaluate", json={"n_scenarios": 12}).json()
    assert m["n_scenarios"] == 12
    assert 0 <= m["model"]["top1_accuracy"] <= 1
    assert "constant_velocity" in m["baselines"]


@pytest.mark.parametrize(
    "path, body",
    [
        ("/api/world/step", {"steps": -1}),
        ("/api/world/step", {"steps": 0}),
        ("/api/world/step", {"steps": 10_000}),
        ("/api/world/step", {"dt": 0.5}),
        ("/api/world/config", {"wind_strength": -1}),
        ("/api/world/config", {"wind_strength": 99}),
        ("/api/world/config", {"wind_direction_deg": 720}),
        ("/api/world/config", {"randomness": 1.5}),
        ("/api/world/config", {"flowers": [{"id": "A", "x": -5, "y": 10}]}),
        ("/api/world/config", {"flowers": [{"id": "A", "x": 50, "y": 1000}]}),
        ("/api/world/config", {"flowers": [{"id": "ZZ", "x": 50, "y": 30}]}),
        ("/api/world/config", {"flowers": [{"id": "A", "x": "left", "y": 30}]}),
        ("/api/world/config", {"gravity": 9.8}),
        ("/api/world/reset", {"seed": -3}),
        ("/api/world/reset", {"seed": "abc"}),
        ("/api/predict", {"n_futures": 0}),
        ("/api/predict", {"n_futures": 100_000}),
        ("/api/predict", {"horizon_steps": -5}),
        ("/api/predict", {"observation": {"step": 0}}),
        ("/api/evaluate", {"n_scenarios": 0}),
        ("/api/evaluate", {"n_scenarios": 1_000_000}),
    ],
)
def test_invalid_requests_are_rejected(path, body):
    before = client.get("/api/world/state").json()
    assert client.post(path, json=body).status_code == 422
    assert client.get("/api/world/state").json() == before  # nothing was changed


def test_malformed_json_is_rejected():
    r = client.post("/api/world/config", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
