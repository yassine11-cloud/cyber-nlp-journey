import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phase1_fondamentaux import xor_numpy, xor_pytorch  # noqa: E402


def test_autograd_gradients_match_manual_backpropagation():
    params = xor_numpy.init_params(2, 4, 1, seed=3)
    _, cache = xor_numpy.forward(xor_numpy.X, params)
    manual = xor_numpy.backward(xor_numpy.X, xor_numpy.Y, params, cache)

    model = xor_pytorch.build_model(seed=3)
    loss = torch.nn.MSELoss()(model(torch.from_numpy(xor_numpy.X)), torch.from_numpy(xor_numpy.Y))
    loss.backward()
    np.testing.assert_allclose(model[0].weight.grad.numpy().T, manual["W1"], atol=1e-8)
    np.testing.assert_allclose(model[0].bias.grad.numpy(), manual["b1"][0], atol=1e-8)
    np.testing.assert_allclose(model[2].weight.grad.numpy().T, manual["W2"], atol=1e-8)
    np.testing.assert_allclose(model[2].bias.grad.numpy(), manual["b2"][0], atol=1e-8)


def test_pytorch_and_numpy_follow_the_same_trajectory():
    _, numpy_history = xor_numpy.train(seed=7, epochs=500)
    _, torch_history = xor_pytorch.train(seed=7, epochs=500)
    assert abs(numpy_history[-1] - torch_history[-1]) < 1e-6
