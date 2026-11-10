"""Visual planning comparison (Phase 2): one-level vs two-level latent planning from pixels.

Usage:  python -m experiments.run_visual [--seeds 20] [--model-seeds 0 1 2] [--out experiments/results/visual]

For every trained model seed, both variants plan on identical (layout, episode
seed, condition) triples from the frozen VISUAL_EVAL_SEEDS, in clean scenes and
with an independent moving distractor, under a matched budget. Outputs:
metrics.json, episodes.csv, comparison.svg.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from api.evaluation.metrics import wilson_interval
from api.planning.garden import LEARNED_LAYOUTS
from api.planning.seeds import VISUAL_EVAL_SEEDS, VISUAL_MODEL_SEEDS
from api.visual.episode import run_visual_episode
from api.visual.jepa import TRAIN_CONFIG, VisualArtifactMissing, load
from api.visual.planners import ONE_LEVEL, TWO_LEVEL, VISUAL_BUDGET
from api.visual.scene import FRAME_H, FRAME_W, MAX_DECISIONS, VISUAL_WIND_SCALE

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"one_level": "#eb6834", "two_level": "#1baf7a"}
LABELS = {"one_level": "One-level latent planner", "two_level": "Two-level (H-JEPA-inspired)"}
VARIANTS = ("one_level", "two_level")


def _summ(rows):
    n = len(rows)
    landed = [r for r in rows if r["landed"]]
    lo, hi = wilson_interval(len(landed), n)
    return {
        "episodes": n,
        "landing_rate": len(landed) / n,
        "landing_rate_ci95": [lo, hi],
        "collision_rate": sum(r["collided"] for r in rows) / n,
        "mean_seconds_to_land": float(np.mean([r["seconds_to_land"] for r in landed])) if landed else None,
        "evaluations_per_replan": rows[0]["evaluations_per_replan"],
        "planning_latency_ms_p50": float(np.median([r["latency_p50_ms"] for r in rows])),
    }


def _text(x, y, s, size=12, fill=INK, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def render_chart(m: dict) -> str:
    """Landing rate per variant, clean vs distractor, per layout; dots = individual model seeds."""
    W, H = 760, 400
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, -apple-system, '
        'Segoe UI, Helvetica, Arial, sans-serif" role="img" aria-label="Visual planning results">',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        _text(24, 30, f"Visual planning from pixels: {m['seeds']['count']} frozen seeds per cell, "
              f"{len(m['model_seeds'])} training seeds per variant, {m['config']['budget_evaluations_per_replan']} "
              "evaluations per replan", 14, weight="600"),
        _text(24, 50, "Target landing rate. Bars: mean over training seeds. Dots: each training seed.", 11, MUTED),
    ]
    base, height = 330, 220
    groups = [(lay, cond) for lay in LEARNED_LAYOUTS for cond in ("clean", "distractor")]
    gw = (W - 80) / len(groups)
    parts.append(f'<line x1="50" y1="{base}" x2="{W - 24}" y2="{base}" stroke="{MUTED}"/>')
    for t in (0.25, 0.5, 0.75, 1.0):
        y = base - height * t
        parts.append(f'<line x1="50" y1="{y:.1f}" x2="{W - 24}" y2="{y:.1f}" stroke="{GRID}"/>')
        parts.append(_text(44, y + 4, f"{t:.0%}", 10, MUTED, "end"))
    for g, (lay, cond) in enumerate(groups):
        gx = 60 + g * gw
        for i, v in enumerate(VARIANTS):
            cell = m["by_variant"][v][lay][cond]
            x = gx + 18 + i * 64
            h = max(height * cell["mean_landing_rate"], 1)
            parts.append(f'<rect x="{x:.1f}" y="{base - h:.1f}" width="50" height="{h:.1f}" rx="3" fill="{COLORS[v]}" opacity="0.85"/>')
            parts.append(_text(x + 25, base - h - 6, f"{cell['mean_landing_rate']:.0%}", 11, INK, "middle", "600"))
            for k, rate in enumerate(cell["per_model_seed"].values()):
                parts.append(f'<circle cx="{x + 12 + 13 * k:.1f}" cy="{base - height * rate:.1f}" r="3.2" '
                             f'fill="{SURFACE}" stroke="{INK}" stroke-width="1.2"/>')
        parts.append(_text(gx + 75, base + 18, f"{lay.capitalize()} route", 12, INK, "middle", "600"))
        parts.append(_text(gx + 75, base + 33, "with distractor" if cond == "distractor" else "clean scene", 11, MUTED, "middle"))
    for i, v in enumerate(VARIANTS):
        x = 60 + i * 260
        parts.append(f'<rect x="{x}" y="{H - 22}" width="10" height="10" rx="2" fill="{COLORS[v]}"/>')
        parts.append(_text(x + 14, H - 13, LABELS[v], 11, MUTED))
    return "\n".join(parts + ["</svg>"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(VISUAL_EVAL_SEEDS))
    ap.add_argument("--model-seeds", type=int, nargs="+", default=list(VISUAL_MODEL_SEEDS))
    ap.add_argument("--out", type=Path, default=Path("experiments/results/visual"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    seeds = list(VISUAL_EVAL_SEEDS)[: args.seeds]
    rows, models = [], {}
    for ms in args.model_seeds:
        for levels, variant in ((1, "one_level"), (2, "two_level")):
            try:
                model, meta, probes = load(levels, ms)
            except VisualArtifactMissing as exc:
                raise SystemExit(str(exc)) from exc
            models[f"{variant}_seed{ms}"] = meta
            for lay in LEARNED_LAYOUTS:
                for dist in (False, True):
                    for s in seeds:
                        ep = run_visual_episode(lay, variant, s, dist, model, record=False)
                        rows.append({
                            "variant": variant, "model_seed": ms, "model_version": meta["model_version"],
                            "layout": lay, "condition": "distractor" if dist else "clean", "seed": s,
                            "outcome": ep["outcome"], "landed": ep["landed"], "collided": ep["collided"],
                            "decisions": ep["decisions"], "seconds_to_land": ep["seconds_to_land"],
                            "evaluations_per_replan": ep["evaluations_per_replan"],
                            "evaluations_total": ep["evaluations_total"],
                            "latency_p50_ms": ep["latency"]["p50_ms"],
                        })
            sub = [r for r in rows if r["variant"] == variant and r["model_seed"] == ms]
            print(f"{variant:<10} model seed {ms}: landed {np.mean([r['landed'] for r in sub]):.1%}  "
                  f"collided {np.mean([r['collided'] for r in sub]):.1%}")

    by_variant = {}
    for v in VARIANTS:
        by_variant[v] = {}
        for lay in LEARNED_LAYOUTS:
            by_variant[v][lay] = {}
            for cond in ("clean", "distractor"):
                cell = [r for r in rows if r["variant"] == v and r["layout"] == lay and r["condition"] == cond]
                per = {str(ms): float(np.mean([r["landed"] for r in cell if r["model_seed"] == ms])) for ms in args.model_seeds}
                by_variant[v][lay][cond] = {
                    **_summ(cell),
                    "per_model_seed": per,
                    "mean_landing_rate": float(np.mean(list(per.values()))),
                    "sd_landing_rate_across_model_seeds": float(np.std(list(per.values()))),
                }
            c, d = by_variant[v][lay]["clean"], by_variant[v][lay]["distractor"]
            by_variant[v][lay]["change_under_distraction"] = d["mean_landing_rate"] - c["mean_landing_rate"]

    probe = {
        v: {
            key: {
                metric: float(np.mean([models[f"{v}_seed{ms}"]["probe"][key][metric] for ms in args.model_seeds]))
                for metric in ("butterfly_position_r2", "distractor_position_r2")
            }
            for key in models[f"{v}_seed{args.model_seeds[0]}"]["probe"]
        }
        for v in VARIANTS
    }
    health = {
        name: {
            "model_version": meta["model_version"],
            "final_z": meta["training_curve"][-1]["z"],
            "final_h": meta["training_curve"][-1].get("h"),
            "final_val_pred": meta["training_curve"][-1]["val_pred"],
            "flags": meta["flags"],
            "training_seconds": meta["training_seconds"],
            "n_parameters": meta["n_parameters"],
        }
        for name, meta in models.items()
    }
    metrics = {
        "experiment": "visual planning (H-JEPA-inspired adaptation)",
        "app_version": "1.1.0",
        "seeds": {"first": seeds[0], "last": seeds[-1], "count": len(seeds), "range_name": "VISUAL_EVAL_SEEDS"},
        "model_seeds": args.model_seeds,
        "config": {
            "budget_evaluations_per_replan": VISUAL_BUDGET,
            "one_level": ONE_LEVEL,
            "two_level": TWO_LEVEL,
            "frame": [FRAME_H, FRAME_W],
            "max_decisions": MAX_DECISIONS,
            "wind_scale": VISUAL_WIND_SCALE,
            "training": TRAIN_CONFIG,
        },
        "by_variant": by_variant,
        "probe_r2_mean_over_model_seeds": probe,
        "training_health": health,
        "failures": {k: v["flags"] for k, v in health.items() if v["flags"]},
        "runtime_seconds": round(time.perf_counter() - t0, 1),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with open(args.out / "episodes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "comparison.svg").write_text(render_chart(metrics), encoding="utf-8")
    for v in VARIANTS:
        for lay in LEARNED_LAYOUTS:
            c = by_variant[v][lay]
            print(f"{v:<10} {lay:<7} clean {c['clean']['mean_landing_rate']:.0%}  distractor "
                  f"{c['distractor']['mean_landing_rate']:.0%}  change {c['change_under_distraction']:+.0%}")
    print(f"probe R2 (mean over seeds): {json.dumps(probe)}")
    print(f"written to {args.out}/ in {metrics['runtime_seconds']} s")


if __name__ == "__main__":
    main()
