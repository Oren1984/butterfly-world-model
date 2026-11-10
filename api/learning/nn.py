"""A tiny numpy multilayer perceptron with manual backpropagation and Adam.

Both V1.1 experiments train small models on the CPU. Keeping this in numpy
avoids a deep-learning dependency; the gradients are checked against finite
differences in `tests/test_nn.py`.
"""
from __future__ import annotations

import numpy as np


class MLP:
    """Fully connected ReLU network. The last layer is linear."""

    def __init__(self, sizes: list[int], rng: np.random.Generator):
        self.sizes = list(sizes)
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, dtype=np.float32) for b in sizes[1:]]
        self.zero_grad()
        self._acts: list[np.ndarray] = []

    # ---- parameters -----------------------------------------------------

    def params(self) -> list[np.ndarray]:
        return [*self.W, *self.b]

    def grads(self) -> list[np.ndarray]:
        return [*self.gW, *self.gb]

    def zero_grad(self) -> None:
        self.gW = [np.zeros_like(w) for w in self.W]
        self.gb = [np.zeros_like(b) for b in self.b]

    def copy(self) -> "MLP":
        other = MLP.__new__(MLP)
        other.sizes = list(self.sizes)
        other.W = [w.copy() for w in self.W]
        other.b = [b.copy() for b in self.b]
        other.zero_grad()
        other._acts = []
        return other

    def ema_from(self, online: "MLP", tau: float) -> None:
        """Exponential moving average toward another network (used for target encoders)."""
        for mine, theirs in zip(self.params(), online.params()):
            mine *= tau
            mine += (1.0 - tau) * theirs

    def state(self, prefix: str) -> dict[str, np.ndarray]:
        out = {f"{prefix}.W{i}": w for i, w in enumerate(self.W)}
        out.update({f"{prefix}.b{i}": b for i, b in enumerate(self.b)})
        return out

    @classmethod
    def from_state(cls, data, prefix: str) -> "MLP":
        net = cls.__new__(cls)
        n = sum(1 for k in data.keys() if k.startswith(f"{prefix}.W"))
        net.W = [np.asarray(data[f"{prefix}.W{i}"], dtype=np.float32) for i in range(n)]
        net.b = [np.asarray(data[f"{prefix}.b{i}"], dtype=np.float32) for i in range(n)]
        net.sizes = [net.W[0].shape[0], *[w.shape[1] for w in net.W]]
        net.zero_grad()
        net._acts = []
        return net

    # ---- computation ----------------------------------------------------

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        h = np.asarray(x, dtype=np.float32)
        acts = [h]
        last = len(self.W) - 1
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            h = h @ w + b
            if i < last:
                h = np.maximum(h, 0.0)
            acts.append(h)
        if train:
            self._acts = acts
        return h

    def backward(self, gy: np.ndarray) -> np.ndarray:
        """Accumulate parameter gradients for the last training forward pass; return d loss / d input."""
        g = np.asarray(gy, dtype=np.float32)
        acts = self._acts
        for i in range(len(self.W) - 1, -1, -1):
            if i < len(self.W) - 1:
                g = g * (acts[i + 1] > 0)
            self.gW[i] += acts[i].T @ g
            self.gb[i] += g.sum(axis=0)
            g = g @ self.W[i].T
        return g


class Adam:
    def __init__(self, params: list[np.ndarray], lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray], clip: float = 5.0) -> None:
        self.t += 1
        norm = float(np.sqrt(sum(float((g * g).sum()) for g in grads)))
        scale = min(1.0, clip / (norm + 1e-12))
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            g = g * scale
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= (self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)).astype(p.dtype)


def mse(pred: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean squared error over all elements, and its gradient."""
    d = pred - target
    return float((d * d).mean()), (2.0 * d / d.size).astype(np.float32)


def bce_logits(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Binary cross-entropy on logits (mean), and its gradient."""
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    loss = -(labels * np.log(p + 1e-7) + (1 - labels) * np.log(1 - p + 1e-7)).mean()
    return float(loss), ((p - labels) / logits.size).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
