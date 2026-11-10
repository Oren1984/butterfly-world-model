"""H-JEPA-inspired latent world models learned from pixels (Phase 2).

One-level model
  encoder E: frame pair -> z (32-d)
  predictor P: (z, action) -> z' (residual) plus a collision logit
  objective: predict the *target encoder's* embedding of the next frame pair
  (a joint-embedding predictive objective, no pixel reconstruction).

Two-level model (trained jointly, same data and steps)
  adds its own higher-level encoder E_h: z -> h (16-d), a higher-level
  predictor P_h: (h, macro-action) -> h after MACRO_K decisions plus a
  collision-within-macro logit, and a projection G: h -> z used to turn
  predicted high-level states into low-level subgoals.

Collapse prevention and monitoring
  - target encoders are exponential moving averages of the online encoders
    and receive no gradient (stop-gradient), as in JEPA-style methods;
  - VICReg-style variance and covariance terms on z and h keep every
    embedding dimension spread out and decorrelated;
  - during training we log the mean per-dimension standard deviation and the
    effective rank of z and h on a fixed validation batch. A run is flagged
    "collapsed" if the mean std falls below 0.1 or the effective rank below 2.

Explicit command:  python -m experiments.train_visual
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..learning.nn import MLP, Adam, bce_logits, mse, sigmoid
from ..planning.garden import N_ACTIONS
from .scene import INPUT_SIZE, MACRO_K, collect, frames_for

FORMAT = "bwm-visual-jepa"
FORMAT_VERSION = 1
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "visual"
Z, HZ = 32, 16

TRAIN_CONFIG = {
    "encoder_hidden": 256,
    "predictor_hidden": 128,
    "high_hidden": 64,
    "latent": Z,
    "high_latent": HZ,
    "macro_k": MACRO_K,
    "batch": 128,
    "steps": 3000,
    "lr": 1e-3,
    "ema_tau": 0.99,
    "w_pred": 1.0,
    "w_var": 1.0,
    "w_cov": 0.04,
    "w_collision": 0.2,
    "w_high_pred": 1.0,
    "w_proj": 0.5,
    "log_every": 250,
}


# ---- VICReg-style regularisers (gradients checked in tests) ------------------

def variance_loss(z: np.ndarray, eps: float = 1e-4) -> tuple[float, np.ndarray]:
    n, d = z.shape
    zc = z - z.mean(0)
    std = np.sqrt((zc * zc).sum(0) / (n - 1) + eps)
    loss = float(np.maximum(0.0, 1.0 - std).mean())
    g = -((std < 1.0) / d)[None, :] * zc / ((n - 1) * std[None, :])
    return loss, g.astype(np.float32)


def covariance_loss(z: np.ndarray) -> tuple[float, np.ndarray]:
    n, d = z.shape
    zc = z - z.mean(0)
    c = zc.T @ zc / (n - 1)
    off = c - np.diag(np.diag(c))
    return float((off * off).sum() / d), (4.0 * zc @ off / (d * (n - 1))).astype(np.float32)


def embedding_health(z: np.ndarray) -> dict:
    zc = z - z.mean(0)
    std = zc.std(0)
    sv = np.linalg.svd(zc, compute_uv=False)
    p = sv / max(sv.sum(), 1e-12)
    p = p[p > 0]
    return {"mean_std": float(std.mean()), "effective_rank": float(np.exp(-(p * np.log(p)).sum()))}


# ---- model -------------------------------------------------------------------

class VisualWorldModel:
    def __init__(self, levels: int, rng: np.random.Generator, cfg: dict = TRAIN_CONFIG):
        assert levels in (1, 2)
        self.levels, self.cfg = levels, cfg
        self.enc = MLP([INPUT_SIZE, cfg["encoder_hidden"], Z], rng)
        self.pred = MLP([Z + N_ACTIONS, cfg["predictor_hidden"], Z + 1], rng)
        if levels == 2:
            self.henc = MLP([Z, cfg["high_hidden"], HZ], rng)
            self.hpred = MLP([HZ + N_ACTIONS, cfg["high_hidden"], HZ + 1], rng)
            self.proj = MLP([HZ, cfg["high_hidden"], Z], rng)
        self._init_targets()

    def _init_targets(self) -> None:
        self.tenc = self.enc.copy()
        if self.levels == 2:
            self.thenc = self.henc.copy()

    def nets(self) -> dict[str, MLP]:
        out = {"enc": self.enc, "pred": self.pred}
        if self.levels == 2:
            out.update(henc=self.henc, hpred=self.hpred, proj=self.proj)
        return out

    # inference
    def encode(self, x: np.ndarray) -> np.ndarray:
        return self.enc.forward(x)

    def encode_high(self, z: np.ndarray) -> np.ndarray:
        return self.henc.forward(z)

    def step(self, z: np.ndarray, actions: np.ndarray):
        a = np.zeros((len(z), N_ACTIONS), dtype=np.float32)
        a[np.arange(len(z)), actions] = 1.0
        out = self.pred.forward(np.concatenate([z, a], axis=1))
        return z + out[:, :Z], sigmoid(out[:, Z])

    def step_high(self, h: np.ndarray, actions: np.ndarray):
        m = np.zeros((len(h), N_ACTIONS), dtype=np.float32)
        m[np.arange(len(h)), actions] = 1.0
        out = self.hpred.forward(np.concatenate([h, m], axis=1))
        return h + out[:, :HZ], sigmoid(out[:, HZ])

    def subgoal(self, h: np.ndarray) -> np.ndarray:
        return self.proj.forward(h)

    # training
    def train_step(self, b: dict, opt: Adam) -> dict:
        cfg = self.cfg
        for net in self.nets().values():
            net.zero_grad()
        z = self.enc.forward(b["x_t"], train=True)
        z_next = self.tenc.forward(b["x_t1"])  # stop-gradient target
        out = self.pred.forward(np.concatenate([z, b["a"]], axis=1), train=True)
        zp = z + out[:, :Z]
        ok = (1.0 - b["col1"])[:, None].astype(np.float32)
        l_pred, g_zp = mse(zp * ok, z_next * ok)
        g_zp = g_zp * ok
        l_col, g_col = bce_logits(out[:, Z], b["col1"])
        g_in = self.pred.backward(np.concatenate([cfg["w_pred"] * g_zp, cfg["w_collision"] * g_col[:, None]], axis=1))
        g_z = cfg["w_pred"] * g_zp + g_in[:, :Z]
        l_var, g_var = variance_loss(z)
        l_cov, g_cov = covariance_loss(z)
        g_z += cfg["w_var"] * g_var + cfg["w_cov"] * g_cov
        losses = {"pred": l_pred, "collision": l_col, "var": l_var, "cov": l_cov}
        if self.levels == 2:
            h = self.henc.forward(z, train=True)
            h_k = self.thenc.forward(self.tenc.forward(b["x_tk"]))
            hout = self.hpred.forward(np.concatenate([h, b["m"]], axis=1), train=True)
            hp = h + hout[:, :HZ]
            okk = b["okk"][:, None]
            l_hp, g_hp = mse(hp * okk, h_k * okk)
            g_hp = g_hp * okk
            l_hcol, g_hcol = bce_logits(hout[:, HZ], b["colk"])
            g_hin = self.hpred.backward(
                np.concatenate([cfg["w_high_pred"] * g_hp, cfg["w_collision"] * g_hcol[:, None]], axis=1)
            )
            g_h = cfg["w_high_pred"] * g_hp + g_hin[:, :HZ]
            l_hvar, g_hvar = variance_loss(h)
            l_hcov, g_hcov = covariance_loss(h)
            g_h += cfg["w_var"] * g_hvar + cfg["w_cov"] * g_hcov
            l_proj, g_pr = mse(self.proj.forward(h, train=True), self.tenc.forward(b["x_t"]))
            g_h += self.proj.backward(cfg["w_proj"] * g_pr)
            g_z += self.henc.backward(g_h)
            losses.update(high_pred=l_hp, high_collision=l_hcol, high_var=l_hvar, high_cov=l_hcov, proj=l_proj)
        self.enc.backward(g_z)
        grads = [g for net in self.nets().values() for g in net.grads()]
        opt.step(grads)
        self.tenc.ema_from(self.enc, cfg["ema_tau"])
        if self.levels == 2:
            self.thenc.ema_from(self.henc, cfg["ema_tau"])
        return losses

    def params(self) -> list[np.ndarray]:
        return [p for net in self.nets().values() for p in net.params()]

    def state(self) -> dict[str, np.ndarray]:
        out = {}
        for name, net in self.nets().items():
            out.update(net.state(name))
        return out

    @classmethod
    def from_state(cls, data, levels: int) -> "VisualWorldModel":
        m = cls.__new__(cls)
        m.levels, m.cfg = levels, TRAIN_CONFIG
        m.enc, m.pred = MLP.from_state(data, "enc"), MLP.from_state(data, "pred")
        if levels == 2:
            m.henc, m.hpred, m.proj = (MLP.from_state(data, k) for k in ("henc", "hpred", "proj"))
        m._init_targets()
        return m


# ---- data batches ------------------------------------------------------------

def valid_pairs(data: dict) -> np.ndarray:
    """(trajectory, t) pairs with a previous frame and at least one more valid frame."""
    T = data["actions"].shape[1]
    out = []
    for i, c in enumerate(data["collided_at"]):
        last = c if c > 0 else T  # the collision frame is the last valid frame
        out += [(i, t) for t in range(1, last)]
    return np.array(out)


def make_batch(data: dict, pairs: np.ndarray) -> dict:
    tr, t = pairs[:, 0], pairs[:, 1]
    T = data["actions"].shape[1]
    c = data["collided_at"][tr]
    last = np.where(c > 0, c, T)
    tk = np.minimum(t + MACRO_K, last)
    a = np.zeros((len(tr), N_ACTIONS), dtype=np.float32)
    a[np.arange(len(tr)), data["actions"][tr, t]] = 1.0
    # Macro-action summary: the mean one-hot of the actions taken over the next MACRO_K decisions.
    m = np.zeros_like(a)
    for k in range(MACRO_K):
        idx = np.minimum(t + k, T - 1)
        m[np.arange(len(tr)), data["actions"][tr, idx]] += 1.0 / MACRO_K
    return {
        "x_t": frames_for(data, tr, t),
        "x_t1": frames_for(data, tr, t + 1),
        "x_tk": frames_for(data, tr, tk),
        "a": a,
        "m": m,
        "col1": (c == t + 1).astype(np.float32),
        "colk": ((c > 0) & (c <= t + MACRO_K)).astype(np.float32),
        # The high-level target is used only when a full macro-action fits before the trajectory ends.
        "okk": ((t + MACRO_K <= last) & ~((c > 0) & (c <= t + MACRO_K))).astype(np.float32),
    }


# ---- diagnostic probe ----------------------------------------------------------

def _ridge(x: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    xa = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    return np.linalg.solve(xa.T @ xa + lam * np.eye(xa.shape[1]), xa.T @ y)


def _apply(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((len(x), 1))], axis=1) @ w


def _r2(pred: np.ndarray, y: np.ndarray) -> float:
    return float(1.0 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def fit_probes(model: VisualWorldModel, train: dict, test: dict, rng: np.random.Generator, n: int = 3000) -> tuple[dict, dict]:
    """Frozen linear probes from the learned representation to butterfly and distractor positions.

    Fitted on training trajectories, scored (R^2) on held-out validation trajectories.
    """
    def sample(data, k):
        pairs = valid_pairs(data)
        pairs = pairs[rng.choice(len(pairs), min(k, len(pairs)), replace=False)]
        x = frames_for(data, pairs[:, 0], pairs[:, 1])
        return x, data["pos"][pairs[:, 0], pairs[:, 1]], data["distractor"][pairs[:, 0], pairs[:, 1]]

    xa, pa, da = sample(train, n)
    xb, pb, db = sample(test, n // 2)
    feats = {"z": (model.encode(xa), model.encode(xb))}
    if model.levels == 2:
        feats["h"] = (model.encode_high(feats["z"][0]), model.encode_high(feats["z"][1]))
    weights, scores = {}, {}
    for name, (fa, fb) in feats.items():
        w_pos = _ridge(fa, pa)
        weights[f"probe_{name}_pos"] = w_pos.astype(np.float32)
        da_ok, db_ok = ~np.isnan(da[:, 0]), ~np.isnan(db[:, 0])
        w_d = _ridge(fa[da_ok], da[da_ok])
        scores[name] = {
            "butterfly_position_r2": _r2(_apply(w_pos, fb), pb),
            "distractor_position_r2": _r2(_apply(w_d, fb[db_ok]), db[db_ok]),
        }
    return weights, scores


# ---- training ------------------------------------------------------------------

def train(levels: int, seed: int, train_data: dict, val_data: dict, cfg: dict = TRAIN_CONFIG, log=print) -> dict:
    t0 = time.perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([seed, levels, 31]))
    model = VisualWorldModel(levels, rng, cfg)
    opt = Adam(model.params(), lr=cfg["lr"])
    pairs = valid_pairs(train_data)
    vpairs = valid_pairs(val_data)
    vbatch = make_batch(val_data, vpairs[np.random.default_rng(7).choice(len(vpairs), min(512, len(vpairs)), replace=False)])
    curve, flags = [], []
    for step in range(1, cfg["steps"] + 1):
        batch = make_batch(train_data, pairs[rng.choice(len(pairs), cfg["batch"], replace=False)])
        losses = model.train_step(batch, opt)
        if not all(np.isfinite(v) for v in losses.values()):
            flags.append(f"non-finite loss at step {step}")
            break
        if step % cfg["log_every"] == 0 or step == cfg["steps"]:
            zv = model.encode(vbatch["x_t"])
            zp, _ = model.step(zv, vbatch["a"].argmax(1))
            ok = vbatch["col1"] == 0
            entry = {
                "step": step,
                **{f"train_{k}": round(float(v), 5) for k, v in losses.items()},
                "val_pred": float(((zp - model.tenc.forward(vbatch["x_t1"])) ** 2)[ok].mean()),
                "z": embedding_health(zv),
            }
            if levels == 2:
                entry["h"] = embedding_health(model.encode_high(zv))
            curve.append(entry)
            log(f"  L{levels} seed {seed} step {step}: pred {losses['pred']:.4f}  z std {entry['z']['mean_std']:.2f}"
                f"  z rank {entry['z']['effective_rank']:.1f}" + (f"  h rank {entry['h']['effective_rank']:.1f}" if levels == 2 else ""))
    final = curve[-1] if curve else {}
    for key in ("z", "h"):
        if key in final and (final[key]["mean_std"] < 0.1 or final[key]["effective_rank"] < 2.0):
            flags.append(f"representation {key} collapsed (std {final[key]['mean_std']:.3f}, rank {final[key]['effective_rank']:.2f})")
    if len(curve) > 2 and curve[-1]["val_pred"] > 3 * min(c["val_pred"] for c in curve):
        flags.append("validation prediction loss rose more than 3x above its best value (unstable)")
    probe_w, probe_scores = fit_probes(model, train_data, val_data, np.random.default_rng(seed + 100))
    weights = {**model.state(), **probe_w}
    digest = hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes() for v in weights.values())).hexdigest()[:10]
    meta = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "levels": levels,
        "variant": "one_level" if levels == 1 else "two_level",
        "model_version": f"visual-jepa-L{levels}-v1-{digest}",
        "training_seed": seed,
        "input_size": INPUT_SIZE,
        "n_actions": N_ACTIONS,
        "config": cfg,
        "n_parameters": int(sum(p.size for p in model.params())),
        "training_curve": curve,
        "probe": probe_scores,
        "flags": flags,
        "training_seconds": round(time.perf_counter() - t0, 1),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return {"weights": weights, "meta": meta}


def artifact_name(levels: int, seed: int) -> str:
    return f"{'one' if levels == 1 else 'two'}_level_seed{seed}"


def save(artifact: dict, directory: Path = ARTIFACT_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    name = artifact_name(artifact["meta"]["levels"], artifact["meta"]["training_seed"])
    np.savez_compressed(directory / f"{name}.npz", **artifact["weights"])
    (directory / f"{name}.json").write_text(json.dumps(artifact["meta"], indent=2) + "\n", encoding="utf-8")


class VisualArtifactMissing(RuntimeError):
    pass


def load(levels: int, seed: int, directory: Path = ARTIFACT_DIR) -> tuple[VisualWorldModel, dict, dict]:
    name = artifact_name(levels, seed)
    meta_path, npz_path = directory / f"{name}.json", directory / f"{name}.npz"
    if not meta_path.exists() or not npz_path.exists():
        raise VisualArtifactMissing(
            f"No trained visual model '{name}' found. Train with: python -m experiments.train_visual"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta.get("format"), meta.get("format_version"), meta.get("input_size"), meta.get("n_actions")) != (
        FORMAT, FORMAT_VERSION, INPUT_SIZE, N_ACTIONS
    ):
        raise VisualArtifactMissing(
            f"The visual model '{name}' is not compatible with this code. Retrain: python -m experiments.train_visual"
        )
    with np.load(npz_path) as data:
        weights = {k: data[k] for k in data.files}
    probes = {k: v for k, v in weights.items() if k.startswith("probe_")}
    return VisualWorldModel.from_state(weights, levels), meta, probes


def collect_splits(n_train: int | None = None, n_val: int | None = None) -> tuple[dict, dict]:
    from ..planning.seeds import VISUAL_TRAIN_SEEDS, VISUAL_VAL_SEEDS

    return (collect(list(VISUAL_TRAIN_SEEDS)[: n_train or None]), collect(list(VISUAL_VAL_SEEDS)[: n_val or None]))
