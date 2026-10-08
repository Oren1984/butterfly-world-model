"""Choose the predictor's one free parameter (heading_kappa) on tuning seeds only.

Usage:  python -m experiments.tune
The tuning seeds (100000+) never overlap the evaluation seeds (0..499).
The selected value is then written by hand into api/prediction/predictor.py.
"""
from api.evaluation.experiment import TUNING_BASE_SEED, run_experiment

GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
N_TUNING = 300

if __name__ == "__main__":
    print(f"tuning on seeds {TUNING_BASE_SEED}..{TUNING_BASE_SEED + N_TUNING - 1}")
    print(f"{'kappa':>6} {'top1':>7} {'brier':>7} {'ece':>7}")
    results = []
    for kappa in GRID:
        m, _ = run_experiment(N_TUNING, TUNING_BASE_SEED, kappa=kappa)
        mm = m["model"]
        ece = mm["calibration"]["expected_calibration_error"]
        print(f"{kappa:6.1f} {mm['top1_accuracy']:7.3f} {mm['brier_score']:7.3f} {ece:7.3f}")
        results.append((kappa, mm["brier_score"]))
    # Selection rule: the smallest kappa whose Brier score is within 0.01 of the best
    # (the curve flattens, so we prefer the weaker assumption).
    best = min(b for _, b in results)
    print(f"selected kappa = {next(k for k, b in results if b <= best + 0.01)}")
