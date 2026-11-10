import numpy as np

from api.learning.nn import MLP, Adam, bce_logits, mse
from api.visual.jepa import covariance_loss, variance_loss


def _numeric_grad(f, x, eps=1e-3):
    g = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        hi = f(x)
        x[i] = old - eps
        lo = f(x)
        x[i] = old
        g[i] = (hi - lo) / (2 * eps)
    return g


def test_mlp_input_and_weight_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    net = MLP([5, 7, 3], rng)
    x = rng.standard_normal((4, 5)).astype(np.float64)
    y = rng.standard_normal((4, 3))
    for w in net.W:
        w[...] = w.astype(np.float64)
    net.W = [w.astype(np.float64) for w in net.W]
    net.b = [b.astype(np.float64) + 0.1 for b in net.b]
    net.zero_grad()

    def loss_x(xx):
        return mse(net.forward(xx), y)[0]

    out = net.forward(x, train=True)
    _, g = mse(out, y)
    gx = net.backward(g)
    assert np.allclose(gx, _numeric_grad(loss_x, x.copy()), atol=1e-4)

    w0 = net.W[0]
    def loss_w(ww):
        net.W[0] = ww
        return mse(net.forward(x), y)[0]

    assert np.allclose(net.gW[0], _numeric_grad(loss_w, w0.copy()), atol=1e-4)


def test_vicreg_terms_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    z = rng.standard_normal((9, 4)) * 0.5
    for fn in (variance_loss, covariance_loss):
        _, g = fn(z)
        assert np.allclose(g, _numeric_grad(lambda zz: fn(zz)[0], z.copy(), eps=1e-5), atol=1e-4)


def test_bce_gradient_and_adam_reduce_a_simple_loss():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal(6)
    labels = (rng.random(6) > 0.5).astype(float)
    _, g = bce_logits(logits, labels)
    assert np.allclose(g, _numeric_grad(lambda v: bce_logits(v, labels)[0], logits.copy(), eps=1e-5), atol=1e-5)

    net = MLP([2, 16, 1], rng)
    opt = Adam(net.params(), lr=1e-2)
    x = rng.standard_normal((64, 2)).astype(np.float32)
    y = (x[:, :1] * 2 - x[:, 1:]).astype(np.float32)
    first = None
    for _ in range(200):
        net.zero_grad()
        loss, g = mse(net.forward(x, train=True), y)
        first = first if first is not None else loss
        net.backward(g)
        opt.step(net.grads())
    assert loss < first * 0.1
