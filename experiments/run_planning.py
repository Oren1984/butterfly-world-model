"""Goal-directed planning comparison (Phase 1): flat vs hierarchical planners.

Usage:  python -m experiments.run_planning [--seeds 50] [--out experiments/results/planning] [--no-learned]

Runs every planner on identical start states, targets, wind seeds and budgets
(the frozen PLANNING_EVAL_SEEDS). Exact-dynamics and learned-model results are
reported separately. Outputs: metrics.json, episodes.csv, comparison.svg,
routes.svg. Everything except latency is determined by the seeds and the
learned-model artifact (its version is recorded in metrics.json).
"""
import argparse
import csv
import json
from pathlib import Path

from api.planning.episode import run_episode
from api.planning.evaluation import EXACT, LEARNED, run_comparison
from api.planning.garden import LAYOUTS
from api.planning.learned import ArtifactMissing, load
from api.planning.seeds import PLANNING_EVAL_SEEDS

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SHORT = {
    "flat_exact": "Flat, exact",
    "waypoint_exact": "Explicit waypoints, exact",
    "flat_learned": "Flat, learned",
    "waypoint_learned": "Explicit waypoints, learned",
    "subgoal_learned": "Learned subgoals (HWM-inspired)",
}
COLORS = {"flat": "#eb6834", "waypoint": "#2a78d6", "subgoal": "#1baf7a"}


def _color(planner: str) -> str:
    return COLORS[planner.split("_")[0]]


def _text(x, y, s, size=12, fill=INK, anchor="start", weight="400"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


def _panel(x0, title, group, planners):
    out = [_text(x0, 58, title, 14, weight="600"), _text(x0, 76, "Target landing rate, 95% interval", 11, MUTED)]
    left, width, y = x0 + 150, 170, 96
    for layout in ("direct", "detour"):
        if layout not in group:
            continue
        out.append(_text(x0, y + 4, f"{layout.capitalize()} route", 12, INK, weight="600"))
        y += 14
        for p in planners:
            s = group[layout][p]
            v, (lo, hi) = s["landing_rate"], s["landing_rate_ci95"]
            w = max(width * v, 1)
            label = {"flat": "Flat", "waypoint": "Explicit waypoints", "subgoal": "Learned subgoals"}[p.split("_")[0]]
            out.append(_text(left - 8, y + 13, label, 11, MUTED, "end"))
            out.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="17" rx="3" fill="{_color(p)}"/>')
            out.append(f'<line x1="{left + width * lo:.1f}" y1="{y + 8.5}" x2="{left + width * hi:.1f}" y2="{y + 8.5}" '
                       f'stroke="{INK}" stroke-width="1.2" opacity="0.55"/>')
            out.append(_text(left + max(w, width * hi) + 6, y + 13, f"{v:.0%}", 12, weight="600"))
            y += 24
        y += 14
    return out


def render_comparison(metrics: dict) -> str:
    n = metrics["seeds"]["count"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 330" font-family="system-ui, -apple-system, '
        'Segoe UI, Helvetica, Arial, sans-serif" role="img" aria-label="Goal-directed planning results">',
        f'<rect width="760" height="330" fill="{SURFACE}"/>',
        _text(24, 30, f"Goal-directed planning: {n} frozen seeds per route, matched budget of "
              f"{metrics['config']['budget_evaluations_per_replan']} model evaluations per replan", 14, weight="600"),
    ]
    parts += _panel(24, "Exact dynamics", metrics["exact_dynamics"], EXACT)
    if metrics.get("learned_model"):
        parts += _panel(400, "Learned model", metrics["learned_model"], LEARNED)
    parts.append(_text(24, 318, "Unreachable (walled-in) target: every exact-dynamics planner landed "
                       f"{max(s['landing_rate'] for s in metrics['exact_dynamics']['enclosed'].values()):.0%} "
                       "of the time and failed safely.", 11, MUTED))
    return "\n".join(parts + ["</svg>"])


def render_routes(examples: dict, seed: int) -> str:
    """Detour garden with the actual flights and the last waypoints/subgoals of each planner."""
    lay = LAYOUTS["detour"]
    sc, ox, oy = 6.4, 20, 52
    W, H = lay.width * sc + 2 * ox, lay.height * sc + oy + 56
    X = lambda x: ox + x * sc  # noqa: E731
    Y = lambda y: oy + (lay.height - y) * sc  # noqa: E731
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" font-family="system-ui, -apple-system, '
        'Segoe UI, Helvetica, Arial, sans-serif" role="img" aria-label="Example routes on the detour garden">',
        f'<rect width="{W:.0f}" height="{H:.0f}" fill="{SURFACE}"/>',
        _text(ox, 26, f"Detour route, evaluation seed {seed}: actual flights and intermediate waypoints", 14, weight="600"),
        f'<rect x="{X(0)}" y="{Y(lay.height)}" width="{lay.width * sc}" height="{lay.height * sc}" fill="#f1f3ec" stroke="{GRID}"/>',
    ]
    for r in lay.obstacles:
        parts.append(f'<rect x="{X(r.x0):.1f}" y="{Y(r.y1):.1f}" width="{(r.x1 - r.x0) * sc:.1f}" '
                     f'height="{(r.y1 - r.y0) * sc:.1f}" fill="#5b6b4e" rx="3"/>')
    tx, ty = lay.target
    parts.append(f'<circle cx="{X(tx):.1f}" cy="{Y(ty):.1f}" r="{lay.target_radius * sc:.1f}" fill="#ff8fc2" stroke="#b03a78"/>')
    parts.append(_text(X(tx), Y(ty) - lay.target_radius * sc - 6, "target", 11, MUTED, "middle"))
    sx, sy = lay.start
    parts.append(_text(X(sx), Y(sy) + 24, "start", 11, MUTED, "middle"))
    for i, (name, ep) in enumerate(examples.items()):
        c = _color(name)
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in ep["path"])
        dash = ' stroke-dasharray="5 4"' if ep["planner"].endswith("learned") else ""
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2.4"{dash} opacity="0.9"/>')
        wps = sorted({tuple(p["waypoint"]) for p in ep["plans"] if p["waypoint"]})
        for wx, wy in wps[:40]:
            parts.append(f'<circle cx="{X(wx):.1f}" cy="{Y(wy):.1f}" r="2.6" fill="none" stroke="{c}" stroke-width="1.2" opacity="0.7"/>')
        ex, ey = ep["path"][-1]
        parts.append(f'<circle cx="{X(ex):.1f}" cy="{Y(ey):.1f}" r="4" fill="{c}"/>')
        ly = H - 30 + (i // 3) * 16
        lx = ox + (i % 3) * (W - 2 * ox) / 3
        parts.append(f'<rect x="{lx:.1f}" y="{ly - 9:.1f}" width="10" height="10" rx="2" fill="{c}"/>')
        parts.append(_text(lx + 14, ly, f"{SHORT[name]}: {ep['outcome']}", 11, MUTED))
    parts.append(_text(ox, oy - 10, "Lines: actual flight (dashed = learned model). Small rings: waypoints or subgoals "
                       "the planner chose along the way.", 11, MUTED))
    return "\n".join(parts + ["</svg>"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(PLANNING_EVAL_SEEDS), help="first N of the frozen evaluation seeds")
    ap.add_argument("--out", type=Path, default=Path("experiments/results/planning"))
    ap.add_argument("--no-learned", action="store_true", help="skip the learned-model planners")
    args = ap.parse_args()

    learned = None
    if not args.no_learned:
        try:
            learned = load()
        except ArtifactMissing as exc:
            print(f"{exc}\nContinuing with exact-dynamics planners only.")
    seeds = list(PLANNING_EVAL_SEEDS)[: args.seeds]
    metrics, rows = run_comparison(seeds, learned)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with open(args.out / "episodes.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out / "comparison.svg").write_text(render_comparison(metrics), encoding="utf-8")
    names = list(EXACT) + (list(LEARNED) if learned else [])
    examples = {p: run_episode("detour", p, seeds[0], learned_weights=learned[0] if learned else None) for p in names}
    (args.out / "routes.svg").write_text(render_routes(examples, seeds[0]), encoding="utf-8")
    print(f"written to {args.out}/ in {metrics['runtime_seconds']} s")


if __name__ == "__main__":
    main()
