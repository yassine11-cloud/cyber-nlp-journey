import numpy as np

from phase1_fondamentaux import xor_numpy as xor


def test_forward_shapes():
    params = xor.init_params(2, 4, 1, seed=0)
    out, (a1, a2) = xor.forward(xor.X, params)
    assert out.shape == (4, 1)
    assert a1.shape == (4, 4)
    assert np.all((out > 0) & (out < 1))


def test_gradients_match_numerical_estimate():
    """Vérifie la rétropropagation : gradient analytique ≈ gradient numérique (différences finies)."""
    params = xor.init_params(2, 3, 1, seed=3)
    out, cache = xor.forward(xor.X, params)
    grads = xor.backward(xor.X, xor.Y, params, cache)
    eps = 1e-6
    for name in params:
        idx = (0, 0)
        original = params[name][idx]
        params[name][idx] = original + eps
        loss_plus = xor.loss(xor.forward(xor.X, params)[0], xor.Y)
        params[name][idx] = original - eps
        loss_minus = xor.loss(xor.forward(xor.X, params)[0], xor.Y)
        params[name][idx] = original
        numerical = (loss_plus - loss_minus) / (2 * eps)
        assert abs(numerical - grads[name][idx]) < 1e-6, name


def test_training_reduces_loss_and_learns_xor():
    params, history = xor.train(seed=1, epochs=8000)
    assert history[-1] < history[0] / 100
    assert (xor.predict(xor.X, params) == xor.Y).all()


def test_training_is_reproducible():
    _, h1 = xor.train(seed=7, epochs=500)
    _, h2 = xor.train(seed=7, epochs=500)
    assert h1 == h2
