"""Run the 500-scenario benchmark and write the result files.

Usage:  python -m experiments.run_evaluation [--scenarios 500] [--out experiments/results]

Outputs: metrics.json, scenarios.csv, results.svg. Everything except the
latency numbers is fully determined by the seeds.
"""
import argparse
import csv
import json
from pathlib import Path

from api.evaluation.experiment import EVAL_BASE_SEED, run_experiment

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("model", "20-future world model", "#2a78d6"),
          ("nearest_flower", "Nearest flower", "#eb6834"),
          ("constant_velocity", "Constant velocity", "#1baf7a")]


def _text(x, y, s, size=12, fill=INK, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def _bar_panel(x0, title, values, fmt, vmax, note):
    """Horizontal bars, one per predictor, labelled directly."""
    out = [_text(x0, 58, title, 14, weight="600"), _text(x0, 76, note, 11, MUTED)]
    left, width = x0 + 132, 190
    for i, ((_, label, color), v) in enumerate(zip(SERIES, values)):
        y = 98 + i * 34
        w = max(width * v / vmax, 1)
        out.append(_text(left - 8, y + 14, label, 11, MUTED, "end"))
        out.append(f'<path d="M{left},{y} h{w - 4:.1f} a4,4 0 0 1 4,4 v12 a4,4 0 0 1 -4,4 h-{w - 4:.1f} z" fill="{color}"/>')
        out.append(_text(left + w + 6, y + 14, fmt(v), 12, weight="600"))
    out.append(f'<line x1="{left}" y1="92" x2="{left}" y2="{98 + 3 * 34 - 8}" stroke="{MUTED}" stroke-width="1"/>')
    return out


def _calibration_panel(x0, model):
    size, top = 150, 250
    out = [_text(x0, top - 30, "Calibration of the world model", 14, weight="600"),
           _text(x0, top - 12, "Stated confidence vs. how often it was right (dot area = scenarios)", 11, MUTED)]
    ox, oy = x0 + 44, top + 8
    for t in (0, 0.5, 1):
        gx, gy = ox + t * size, oy + size - t * size
        out.append(f'<line x1="{ox}" y1="{gy}" x2="{ox + size}" y2="{gy}" stroke="{GRID}"/>')
        out.append(_text(ox - 6, gy + 4, f"{t:.0%}", 10, MUTED, "end"))
        out.append(_text(gx, oy + size + 14, f"{t:.0%}", 10, MUTED, "middle"))
    out.append(f'<line x1="{ox}" y1="{oy + size}" x2="{ox + size}" y2="{oy}" stroke="{MUTED}" stroke-dasharray="3 3"/>')
    out.append(_text(ox + size + 6, oy + 6, "perfect", 10, MUTED))
    pts = [(b["mean_confidence"], b["accuracy"], b["count"]) for b in model["calibration"]["bins"] if b["count"]]
    line = " ".join(f"{ox + c * size:.1f},{oy + size - a * size:.1f}" for c, a, _ in pts)
    out.append(f'<polyline points="{line}" fill="none" stroke="{SERIES[0][2]}" stroke-width="2"/>')
    biggest = max(n for _, _, n in pts)
    for c, a, n in pts:
        r = 4 + 5 * (n / biggest) ** 0.5
        out.append(f'<circle cx="{ox + c * size:.1f}" cy="{oy + size - a * size:.1f}" r="{r:.1f}" '
                   f'fill="{SERIES[0][2]}" stroke="{SURFACE}" stroke-width="2"><title>confidence {c:.0%}, '
                   f'accuracy {a:.0%}, {n} scenarios</title></circle>')
    out.append(_text(ox + size / 2, oy + size + 30, "Stated confidence", 11, MUTED, "middle"))
    out.append(f'<text transform="translate({x0 + 8},{oy + size / 2}) rotate(-90)" font-size="11" fill="{MUTED}" '
               f'text-anchor="middle">Observed accuracy</text>')
    return out


def _wind_panel(x0, metrics):
    """Small multiples: accuracy per wind bucket for each predictor."""
    top = 250
    out = [_text(x0, top - 30, "Accuracy by wind strength", 14, weight="600"),
           _text(x0, top - 12, "Top-1 accuracy within each third of the wind range", 11, MUTED)]
    base, height, group = top + 158, 130, 92
    out.append(f'<line x1="{x0}" y1="{base}" x2="{x0 + 3 * group}" y2="{base}" stroke="{MUTED}"/>')
    for g, bucket in enumerate(metrics["accuracy_by_wind"]):
        gx = x0 + g * group + 8
        for i, (key, _, color) in enumerate(SERIES):
            v = bucket[f"{key}_accuracy"] or 0
            h = max(height * v, 1)
            x = gx + i * 24
            out.append(f'<path d="M{x},{base} v-{h - 4:.1f} a4,4 0 0 1 4,-4 h14 a4,4 0 0 1 4,4 v{h - 4:.1f} z" fill="{color}"/>')
            out.append(_text(x + 11, base - h - 4, f"{v * 100:.0f}", 10, INK, "middle"))
        out.append(_text(gx + 35, base + 15, f'{bucket["bucket"]} (n={bucket["count"]})', 11, MUTED, "middle"))
    return out


def render_chart(metrics: dict) -> str:
    preds = {"model": metrics["model"], **metrics["baselines"]}
    acc = [preds[k]["top1_accuracy"] for k, _, _ in SERIES]
    brier = [preds[k]["brier_score"] for k, _, _ in SERIES]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 450" font-family="system-ui, -apple-system, '
        'Segoe UI, Helvetica, Arial, sans-serif" role="img" aria-label="Evaluation results">',
        f'<rect width="760" height="450" fill="{SURFACE}"/>',
        _text(24, 30, f'Butterfly World Model: {metrics["n_scenarios"]} held-out scenarios, '
              f'{metrics["n_futures"]} imagined futures each', 15, weight="600"),
    ]
    parts += _bar_panel(24, "Top-1 landing accuracy", acc, lambda v: f"{v:.1%}", 1.0, "Higher is better")
    parts += _bar_panel(400, "Multiclass Brier score", brier, lambda v: f"{v:.3f}", 2.0, "Lower is better (0 to 2)")
    parts += _calibration_panel(24, metrics["model"])
    parts += _wind_panel(400, metrics)
    # Legend for the grouped bars (the top panels are labelled directly).
    for (_, label, color), x in zip(SERIES, (400, 544, 650)):
        parts.append(f'<rect x="{x}" y="436" width="10" height="10" rx="2" fill="{color}"/>')
        parts.append(_text(x + 14, 445, label, 10, MUTED))
    return "\n".join(parts + ["</svg>"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=500)
    ap.add_argument("--base-seed", type=int, default=EVAL_BASE_SEED)
    ap.add_argument("--out", type=Path, default=Path("experiments/results"))
    args = ap.parse_args()

    metrics, rows = run_experiment(args.scenarios, args.base_seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with open(args.out / "scenarios.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "results.svg").write_text(render_chart(metrics), encoding="utf-8")

    m = metrics["model"]
    print(f"scenarios: {metrics['n_scenarios']}  actual outcomes: {metrics['actual_outcomes']}")
    print(f"{'predictor':<20}{'top-1':>8}{'brier':>8}{'pos.err':>9}{'coverage':>10}{'p50 ms':>9}")
    for name, s in {"model": m, **metrics["baselines"]}.items():
        e = s["landing_position_error"]
        print(f"{name:<20}{s['top1_accuracy']:>8.3f}{s['brier_score']:>8.3f}{e['mean']:>9.2f}"
              f"{e['coverage']:>10.2f}{s['latency']['p50_ms']:>9.3f}")
    print(f"model ECE: {m['calibration']['expected_calibration_error']:.3f}   written to {args.out}/")


if __name__ == "__main__":
    main()
