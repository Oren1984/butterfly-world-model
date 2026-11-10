"""Training data and training for the learned transition model (Phase 1 follow-up).

Data: seeded exploration trajectories in the controlled garden. Each trajectory
is a chain of macro-actions (one random action held for MACRO_K decisions)
from a random start. Every decision yields a short-horizon example; every
macro-action yields a longer-horizon example. Splits are by whole trajectory
and by disjoint seed ranges (see `seeds.py`).

Explicit command:  python -m experiments.train_planning
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..learning.nn import MLP, Adam, bce_logits, mse, sigmoid
from .garden import (
    ACTION_REPEAT,
    ACTION_VELOCITY,
    LAYOUTS,
    LEARNED_LAYOUTS,
    N_ACTIONS,
    inside_obstacle,
    substep,
    wind_schedule,
)
from .models import MACRO_K, N_FEATURES, LearnedModel, features
from .seeds import PLANNING_TRAIN_SEEDS, PLANNING_VAL_SEEDS

DATA_STREAM = 12
FORMAT = "bwm-planning-dynamics"
FORMAT_VERSION = 1
N_MACROS = 6
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "planning"

TRAIN_CONFIG = {
    "hidden": [128, 128],
    "batch": 256,
    "epochs_short": 12,
    "epochs_long": 60,
    "lr": 1e-3,
    "collision_weight": 1.0,
    "collision_pos_weight": 10.0,  # collisions are rare, especially over one decision
    "macro_k": MACRO_K,
    "n_macros_per_trajectory": N_MACROS,
}


def collect(seeds, n_macros: int = N_MACROS) -> dict[str, np.ndarray]:
    """Simulate exploration trajectories for a list of seeds (batched, deterministic per seed)."""
    seeds = list(seeds)
    n = len(seeds)
    t_steps = n_macros * MACRO_K * ACTION_REPEAT + 1
    layout_idx = np.array([s % len(LEARNED_LAYOUTS) for s in seeds])
    s = np.zeros((n, 4))
    winds = np.zeros((n, t_steps, 2))
    actions = np.zeros((n, n_macros), dtype=int)
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence([seed, DATA_STREAM]))
        layout = LAYOUTS[LEARNED_LAYOUTS[layout_idx[i]]]
        while True:
            x, y = rng.uniform(2, layout.width - 2), rng.uniform(2, layout.height - 2)
            if not inside_obstacle(layout, np.array(x), np.array(y)):
                break
        ang, spd = rng.uniform(0, 2 * np.pi), rng.uniform(0, 11)
        s[i] = (x, y, spd * np.cos(ang), spd * np.sin(ang))
        winds[i], _ = wind_schedule(rng, t_steps)
        actions[i] = rng.integers(0, N_ACTIONS, n_macros)

    short = {k: [] for k in ("s", "wind", "a", "s_next", "collided", "layout", "traj")}
    long = {k: [] for k in ("s", "wind", "a", "s_next", "collided", "layout", "traj")}
    alive = np.ones(n, dtype=bool)
    t = 0
    for m in range(n_macros):
        m_start, m_wind, m_alive = s.copy(), winds[:, t].copy(), alive.copy()
        m_col = np.zeros(n, dtype=bool)
        for _ in range(MACRO_K):
            d_start, d_wind, d_alive = s.copy(), winds[:, t].copy(), alive.copy()
            d_col = np.zeros(n, dtype=bool)
            for _ in range(ACTION_REPEAT):
                for li, name in enumerate(LEARNED_LAYOUTS):
                    sel = (layout_idx == li) & alive & ~d_col
                    if not sel.any():
                        continue
                    nxt, col, _ = substep(LAYOUTS[name], s[sel].copy(), ACTION_VELOCITY[actions[sel, m]], winds[sel, t])
                    s[sel] = nxt
                    idx = np.flatnonzero(sel)
                    d_col[idx[col]] = True
                t += 1
            for k, v in zip(short, (d_start, d_wind, actions[:, m], s.copy(), d_col, layout_idx, np.arange(n))):
                short[k].append(v[d_alive])
            m_col |= d_col
            alive &= ~d_col
        for k, v in zip(long, (m_start, m_wind, actions[:, m], s.copy(), m_col, layout_idx, np.arange(n))):
            long[k].append(v[m_alive])
    out = {}
    for prefix, d in (("short", short), ("long", long)):
        for k, v in d.items():
            out[f"{prefix}_{k}"] = np.concatenate(v)
    out["seeds"] = np.array(seeds)
    return out


def _inputs(data: dict, prefix: str) -> np.ndarray:
    x = np.zeros((len(data[f"{prefix}_s"]), N_FEATURES), dtype=np.float32)
    for li, name in enumerate(LEARNED_LAYOUTS):
        sel = data[f"{prefix}_layout"] == li
        x[sel] = features(data[f"{prefix}_s"][sel], data[f"{prefix}_wind"][sel], name, data[f"{prefix}_a"][sel])
    return x


def _train_head(data, val, prefix, epochs, rng, log) -> tuple[MLP, dict, dict]:
    cfg = TRAIN_CONFIG
    x = _inputs(data, prefix)
    delta = data[f"{prefix}_s_next"] - data[f"{prefix}_s"]
    col = data[f"{prefix}_collided"].astype(np.float32)
    ok = col == 0
    mean, std = delta[ok].mean(0), delta[ok].std(0) + 1e-6
    y = ((delta - mean) / std).astype(np.float32)
    net = MLP([N_FEATURES, *cfg["hidden"], 5], rng)
    opt = Adam(net.params(), lr=cfg["lr"])
    n = len(x)
    curve = []
    for epoch in range(epochs):
        order = rng.permutation(n)
        losses = []
        for i in range(0, n, cfg["batch"]):
            b = order[i : i + cfg["batch"]]
            net.zero_grad()
            out = net.forward(x[b], train=True)
            m = ok[b][:, None].astype(np.float32)
            l1, g1 = mse(out[:, :4] * m, y[b] * m)
            l2, g2 = bce_logits(out[:, 4], col[b])
            w = 1.0 + (cfg["collision_pos_weight"] - 1.0) * col[b]
            g = np.concatenate([g1 * m, cfg["collision_weight"] * (g2 * w)[:, None]], axis=1)
            net.backward(g)
            opt.step(net.grads())
            losses.append(l1 + l2)
        curve.append(float(np.mean(losses)))
        log(f"  {prefix} epoch {epoch + 1}/{epochs}  loss {curve[-1]:.4f}")
    # Positive weighting shifts the logits by log(weight); undo it so planners get calibrated probabilities.
    offset = np.array([np.log(cfg["collision_pos_weight"])], dtype=np.float32)
    norms = {f"{prefix}_mean": mean, f"{prefix}_std": std, f"{prefix}_logit_offset": offset}
    metrics = _evaluate_head(net, norms, val, prefix)
    metrics["train_loss_curve"] = curve
    return net, norms, metrics


def _evaluate_head(net: MLP, norms: dict, val: dict, prefix: str) -> dict:
    x = _inputs(val, prefix)
    out = net.forward(x)
    pred = val[f"{prefix}_s"] + out[:, :4] * norms[f"{prefix}_std"] + norms[f"{prefix}_mean"]
    ok = ~val[f"{prefix}_collided"]
    err = np.hypot(*(pred[ok, :2] - val[f"{prefix}_s_next"][ok, :2]).T)
    p_raw = sigmoid(out[:, 4])
    p = sigmoid(out[:, 4] - norms[f"{prefix}_logit_offset"][0])
    col = val[f"{prefix}_collided"]
    return {
        "collision_detection_recall": float(((p_raw > 0.5) & col).sum() / max(col.sum(), 1)),
        "collision_detection_precision": float(((p_raw > 0.5) & col).sum() / max((p_raw > 0.5).sum(), 1)),
        "collision_mean_predicted_probability": float(p.mean()),
        "n_val_examples": int(len(x)),
        "position_rmse": float(np.sqrt((err**2).mean())),
        "position_median_error": float(np.median(err)),
        "collision_rate_val": float(col.mean()),
    }


def train(seed: int = 0, n_train: Optional[int] = None, n_val: Optional[int] = None, log=print) -> dict:
    """Train the short- and long-horizon heads. Returns the artifact (weights + metadata)."""
    t0 = time.perf_counter()
    train_seeds = list(PLANNING_TRAIN_SEEDS)[: n_train or None]
    val_seeds = list(PLANNING_VAL_SEEDS)[: n_val or None]
    log(f"collecting {len(train_seeds)} training and {len(val_seeds)} validation trajectories")
    data, val = collect(train_seeds), collect(val_seeds)
    rng = np.random.default_rng(seed)
    short, n_short, m_short = _train_head(data, val, "short", TRAIN_CONFIG["epochs_short"], rng, log)
    long, n_long, m_long = _train_head(data, val, "long", TRAIN_CONFIG["epochs_long"], rng, log)
    weights = {**short.state("short"), **long.state("long"), **n_short, **n_long}
    digest = hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes() for v in weights.values())).hexdigest()[:10]
    meta = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "model_version": f"planning-dynamics-v1-{digest}",
        "training_seed": seed,
        "config": TRAIN_CONFIG,
        "layouts": list(LEARNED_LAYOUTS),
        "n_features": N_FEATURES,
        "n_actions": N_ACTIONS,
        "data": {
            "train_seeds": [train_seeds[0], train_seeds[-1]],
            "val_seeds": [val_seeds[0], val_seeds[-1]],
            "n_train_trajectories": len(train_seeds),
            "n_val_trajectories": len(val_seeds),
            "n_short_train_examples": int(len(data["short_s"])),
            "n_long_train_examples": int(len(data["long_s"])),
            "split": "by whole trajectory; train and validation seed ranges are disjoint from evaluation seeds",
        },
        "metrics": {"short": m_short, "long": m_long},
        "training_seconds": round(time.perf_counter() - t0, 1),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return {"weights": weights, "meta": meta}


def save(artifact: dict, directory: Path = ARTIFACT_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "dynamics.npz", **artifact["weights"])
    (directory / "dynamics.json").write_text(json.dumps(artifact["meta"], indent=2) + "\n", encoding="utf-8")
    return directory


class ArtifactMissing(RuntimeError):
    pass


def load(directory: Path = ARTIFACT_DIR) -> tuple[dict, dict]:
    """Load and validate the learned dynamics. Raises ArtifactMissing with a clear message."""
    meta_path, npz_path = directory / "dynamics.json", directory / "dynamics.npz"
    if not meta_path.exists() or not npz_path.exists():
        raise ArtifactMissing(
            "No learned planning model found. Train one with: python -m experiments.train_planning"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (
        meta.get("format") != FORMAT
        or meta.get("format_version") != FORMAT_VERSION
        or meta.get("n_features") != N_FEATURES
        or meta.get("n_actions") != N_ACTIONS
        or meta.get("layouts") != list(LEARNED_LAYOUTS)
    ):
        raise ArtifactMissing(
            f"The learned planning model in {directory} is not compatible with this code "
            f"(found {meta.get('format')} v{meta.get('format_version')}). Retrain: python -m experiments.train_planning"
        )
    with np.load(npz_path) as data:
        weights = {k: data[k] for k in data.files}
    return weights, meta


def model_from(weights: dict, layout_name: str) -> LearnedModel:
    norms = {k: weights[k] for k in weights if k.startswith(("short_", "long_")) and "." not in k}
    return LearnedModel(LAYOUTS[layout_name], MLP.from_state(weights, "short"), MLP.from_state(weights, "long"), norms)
