"""Train the visual latent world models (Phase 2): one-level and two-level, several seeds.

Usage:  python -m experiments.train_visual [--seeds 0 1 2] [--levels 1 2] [--steps 3000] [--out artifacts/visual]

Data: the frozen VISUAL_TRAIN_SEEDS trajectories (split by whole trajectory);
VISUAL_VAL_SEEDS are held out for monitoring and the diagnostic probe. Both
variants see exactly the same trajectories, batch size and number of steps.
Writes <variant>_seed<k>.npz/.json (model version, seed, config, training
curve, collapse checks, probe scores). CPU only; a few minutes per model.
"""
import argparse
from pathlib import Path

from api.visual.jepa import ARTIFACT_DIR, TRAIN_CONFIG, collect_splits, save, train
from api.planning.seeds import VISUAL_MODEL_SEEDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(VISUAL_MODEL_SEEDS))
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2], choices=[1, 2])
    ap.add_argument("--steps", type=int, default=TRAIN_CONFIG["steps"])
    ap.add_argument("--train-trajectories", type=int, default=None, help="default: the full frozen training range")
    ap.add_argument("--out", type=Path, default=ARTIFACT_DIR)
    args = ap.parse_args()
    cfg = dict(TRAIN_CONFIG, steps=args.steps)
    train_data, val_data = collect_splits(args.train_trajectories)
    print(f"trajectories: {len(train_data['seeds'])} train, {len(val_data['seeds'])} validation")
    for seed in args.seeds:
        for levels in args.levels:
            art = train(levels, seed, train_data, val_data, cfg)
            save(art, args.out)
            m = art["meta"]
            probe = ", ".join(f"{k}: position R2 {v['butterfly_position_r2']:.2f} / distractor R2 {v['distractor_position_r2']:.2f}"
                              for k, v in m["probe"].items())
            print(f"{m['model_version']}  {m['training_seconds']} s  probe [{probe}]  flags {m['flags'] or 'none'}")


if __name__ == "__main__":
    main()
