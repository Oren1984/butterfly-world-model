"""Frozen seed ranges for the V1.1 experiments.

These ranges were fixed before any planner or model tuning and must not change
afterwards. Tuning uses only the TUNE ranges; training only TRAIN/VAL; the
reported results only EVAL. `tests/test_planning.py` checks they are disjoint.
"""

# Phase 1: goal-directed planning (state-based).
PLANNING_EVAL_SEEDS = range(0, 50)  # per layout
PLANNING_TUNE_SEEDS = range(50_000, 50_010)
PLANNING_TRAIN_SEEDS = range(200_000, 203_000)  # learned-dynamics training trajectories
PLANNING_VAL_SEEDS = range(300_000, 300_400)

# Phase 2: visual planning (pixels).
VISUAL_EVAL_SEEDS = range(0, 20)  # per layout and condition
VISUAL_TUNE_SEEDS = range(60_000, 60_010)
VISUAL_TRAIN_SEEDS = range(400_000, 400_700)
VISUAL_VAL_SEEDS = range(500_000, 500_120)
VISUAL_MODEL_SEEDS = (0, 1, 2)  # independent training runs reported for each variant

ALL_RANGES = {
    "planning_eval": PLANNING_EVAL_SEEDS,
    "planning_tune": PLANNING_TUNE_SEEDS,
    "planning_train": PLANNING_TRAIN_SEEDS,
    "planning_val": PLANNING_VAL_SEEDS,
    "visual_eval": VISUAL_EVAL_SEEDS,
    "visual_tune": VISUAL_TUNE_SEEDS,
    "visual_train": VISUAL_TRAIN_SEEDS,
    "visual_val": VISUAL_VAL_SEEDS,
}
