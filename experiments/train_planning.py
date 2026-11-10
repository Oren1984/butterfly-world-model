"""Train the learned transition model for goal-directed planning (Phase 1 follow-up).

Usage:  python -m experiments.train_planning [--seed 0] [--train-trajectories N] [--out artifacts/planning]

Writes dynamics.npz (weights) and dynamics.json (model version, seed, config,
data seeds and validation metrics). CPU only; about a minute on a laptop.
"""
import argparse
from pathlib import Path

from api.planning.learned import ARTIFACT_DIR, save, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-trajectories", type=int, default=None, help="default: the full frozen training range")
    ap.add_argument("--val-trajectories", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    args = ap.parse_args()
    artifact = train(args.seed, args.train_trajectories, args.val_trajectories)
    out = save(artifact, args.out)
    meta = artifact["meta"]
    for head in ("short", "long"):
        m = meta["metrics"][head]
        print(f"{head:<6} position RMSE {m['position_rmse']:.2f} units  collision recall {m['collision_detection_recall']:.3f}"
              f"  precision {m['collision_detection_precision']:.3f}  (val collision rate {m['collision_rate_val']:.3f},"
              f" mean predicted {m['collision_mean_predicted_probability']:.3f})")
    print(f"model {meta['model_version']} written to {out}/ in {meta['training_seconds']} s")


if __name__ == "__main__":
    main()
