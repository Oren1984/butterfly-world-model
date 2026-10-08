"""Pure metric functions. No simulator or predictor imports here."""
import math
from typing import Optional, Sequence

import numpy as np


def top_outcome(probabilities: dict[str, float]) -> str:
    """Highest-probability outcome; ties resolve to the first key in dict order."""
    return max(probabilities, key=probabilities.get)


def brier_score(probabilities: dict[str, float], actual: str) -> float:
    """Multiclass Brier score for one forecast: sum over classes of (p - outcome)^2. Range 0..2."""
    return float(sum((p - (o == actual)) ** 2 for o, p in probabilities.items()))


def position_error(
    predicted: Optional[tuple[float, float]], actual: Optional[tuple[float, float]]
) -> Optional[float]:
    """Distance between predicted and actual landing points; None when either is undefined."""
    if predicted is None or actual is None:
        return None
    return math.dist(predicted, actual)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def calibration(confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 5) -> dict:
    """Reliability table and expected calibration error over equal-width confidence bins."""
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        row = {"lower": float(edges[b]), "upper": float(edges[b + 1]), "count": count}
        if count:
            row["mean_confidence"] = float(conf[mask].mean())
            row["accuracy"] = float(hit[mask].mean())
            ece += count / len(conf) * abs(row["accuracy"] - row["mean_confidence"])
        else:
            row["mean_confidence"] = row["accuracy"] = None
        bins.append(row)
    return {"expected_calibration_error": float(ece) if len(conf) else None, "bins": bins}


def latency_stats(seconds: Sequence[float]) -> dict:
    ms = np.asarray(seconds, dtype=float) * 1000.0
    p50, p90, p99 = np.percentile(ms, [50, 90, 99])
    return {
        "mean_ms": float(ms.mean()),
        "p50_ms": float(p50),
        "p90_ms": float(p90),
        "p99_ms": float(p99),
        "max_ms": float(ms.max()),
    }


def summarise(rows: Sequence[dict], prefix: str) -> dict:
    """Aggregate per-scenario rows for one predictor (columns are `<prefix>_...`)."""
    n = len(rows)
    correct = [bool(r[f"{prefix}_correct"]) for r in rows]
    errors = [r[f"{prefix}_position_error"] for r in rows if r[f"{prefix}_position_error"] is not None]
    # Error restricted to scenarios where the predicted flower was the right one.
    hits = [r[f"{prefix}_position_error"] for r in rows
            if r[f"{prefix}_position_error"] is not None and r[f"{prefix}_correct"]]
    lo, hi = wilson_interval(sum(correct), n)
    return {
        "top1_accuracy": sum(correct) / n,
        "top1_accuracy_ci95": [lo, hi],
        "brier_score": float(np.mean([r[f"{prefix}_brier"] for r in rows])),
        "landing_position_error": {
            "coverage": len(errors) / n,
            "n_defined": len(errors),
            "mean": float(np.mean(errors)) if errors else None,
            "median": float(np.median(errors)) if errors else None,
            "p90": float(np.percentile(errors, 90)) if errors else None,
            "mean_when_flower_correct": float(np.mean(hits)) if hits else None,
        },
        "calibration": calibration([r[f"{prefix}_confidence"] for r in rows], correct),
        "latency": latency_stats([r[f"{prefix}_latency_s"] for r in rows]),
    }
