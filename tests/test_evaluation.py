import pytest

from api.evaluation import metrics
from api.evaluation.experiment import EVAL_BASE_SEED, TUNING_BASE_SEED, run_experiment, run_scenario
from api.params import NO_LANDING


def test_brier_score_known_values():
    assert metrics.brier_score({"A": 1.0, "B": 0.0, NO_LANDING: 0.0}, "A") == 0.0
    assert metrics.brier_score({"A": 1.0, "B": 0.0, NO_LANDING: 0.0}, "B") == 2.0
    # (0.6-1)^2 + 0.3^2 + 0.1^2 = 0.16 + 0.09 + 0.01
    assert metrics.brier_score({"A": 0.6, "B": 0.3, NO_LANDING: 0.1}, "A") == pytest.approx(0.26)
    # The no-landing class is scored like any other outcome.
    assert metrics.brier_score({"A": 0.6, "B": 0.3, NO_LANDING: 0.1}, NO_LANDING) == pytest.approx(1.26)
    uniform = {"A": 1 / 3, "B": 1 / 3, NO_LANDING: 1 / 3}
    assert metrics.brier_score(uniform, "B") == pytest.approx(2 / 3)


def test_top_outcome_and_tie_break():
    assert metrics.top_outcome({"A": 0.2, "B": 0.5, NO_LANDING: 0.3}) == "B"
    assert metrics.top_outcome({"A": 0.4, "B": 0.4, NO_LANDING: 0.2}) == "A"


def test_position_error():
    assert metrics.position_error((0, 0), (3, 4)) == 5.0
    assert metrics.position_error(None, (3, 4)) is None
    assert metrics.position_error((0, 0), None) is None


def test_calibration_known_example():
    # Bin [0.8, 1.0]: confidence 0.9, accuracy 0.5 -> gap 0.4. Bin [0.4, 0.6): conf 0.5, acc 0.5 -> gap 0.
    conf = [0.9, 0.9, 0.5, 0.5]
    hit = [True, False, True, False]
    cal = metrics.calibration(conf, hit, n_bins=5)
    assert cal["expected_calibration_error"] == pytest.approx(0.5 * 0.4)
    assert [b["count"] for b in cal["bins"]] == [0, 0, 2, 0, 2]
    assert cal["bins"][4]["accuracy"] == 0.5 and cal["bins"][0]["accuracy"] is None
    perfect = metrics.calibration([1.0, 1.0], [True, True])
    assert perfect["expected_calibration_error"] == 0.0


def test_wilson_interval_contains_the_estimate():
    lo, hi = metrics.wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3) and hi == pytest.approx(0.5962, abs=1e-3)


def test_latency_stats():
    stats = metrics.latency_stats([0.001, 0.002, 0.003, 0.004, 0.010])
    assert stats["p50_ms"] == pytest.approx(3.0)
    assert stats["max_ms"] == pytest.approx(10.0)
    assert stats["mean_ms"] == pytest.approx(4.0)


def test_summarise_on_synthetic_rows():
    rows = [
        {"m_correct": True, "m_brier": 0.0, "m_confidence": 1.0, "m_position_error": 1.0, "m_latency_s": 0.001},
        {"m_correct": False, "m_brier": 2.0, "m_confidence": 1.0, "m_position_error": 3.0, "m_latency_s": 0.001},
        {"m_correct": True, "m_brier": 0.5, "m_confidence": 0.5, "m_position_error": None, "m_latency_s": 0.001},
        {"m_correct": False, "m_brier": 0.5, "m_confidence": 0.5, "m_position_error": None, "m_latency_s": 0.001},
    ]
    s = metrics.summarise(rows, "m")
    assert s["top1_accuracy"] == 0.5
    assert s["brier_score"] == 0.75
    assert s["landing_position_error"] == {"coverage": 0.5, "n_defined": 2, "mean": 2.0, "median": 2.0,
                                           "p90": pytest.approx(2.8), "mean_when_flower_correct": 1.0}
    assert s["calibration"]["expected_calibration_error"] == pytest.approx(0.25)


def test_tuning_and_evaluation_seeds_do_not_overlap():
    assert TUNING_BASE_SEED > EVAL_BASE_SEED + 2000


def test_scenario_rows_are_reproducible_and_complete():
    a, b = run_scenario(3), run_scenario(3)
    drop = lambda r: {k: v for k, v in r.items() if not k.endswith("latency_s")}
    assert drop(a) == drop(b)
    for name in ("model", "constant_velocity", "nearest_flower"):
        assert a[f"{name}_correct"] == (a[f"{name}_top"] == a["actual"])
        assert 0 <= a[f"{name}_brier"] <= 2
    if a["actual"] == NO_LANDING:
        assert a["model_position_error"] is None


def test_small_experiment_aggregates_consistently():
    m, rows = run_experiment(n_scenarios=30)
    assert m["n_scenarios"] == len(rows) == 30
    assert m["seeds"] == {"first": 0, "last": 29}
    assert sum(m["actual_outcomes"].values()) == 30
    assert m["model"]["top1_accuracy"] == pytest.approx(sum(r["model_correct"] for r in rows) / 30)
    assert set(m["baselines"]) == {"constant_velocity", "nearest_flower"}
    assert sum(b["count"] for b in m["accuracy_by_wind"]) == 30
    assert sum(b["count"] for b in m["accuracy_by_randomness"]) == 30
    cov = m["model"]["landing_position_error"]
    assert cov["n_defined"] == sum(r["model_position_error"] is not None for r in rows)
