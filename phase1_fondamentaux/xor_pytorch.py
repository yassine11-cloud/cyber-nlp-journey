"""Le même réseau XOR avec PyTorch : la rétropropagation est faite par autograd.

On part exactement des mêmes poids initiaux que la version NumPy, donc les deux versions doivent
suivre la même trajectoire d'apprentissage (c'est ce que vérifie tests/test_xor_pytorch.py).

Lancer (nécessite requirements-deep.txt) :
    python -m phase1_fondamentaux.xor_pytorch
"""
import argparse

import numpy as np
import torch
from torch import nn

from phase1_fondamentaux import xor_numpy


def build_model(seed, n_hidden=4):
    params = xor_numpy.init_params(2, n_hidden, 1, seed)
    model = nn.Sequential(nn.Linear(2, n_hidden), nn.Sigmoid(), nn.Linear(n_hidden, 1), nn.Sigmoid()).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(params["W1"].T))
        model[0].bias.copy_(torch.from_numpy(params["b1"][0]))
        model[2].weight.copy_(torch.from_numpy(params["W2"].T))
        model[2].bias.copy_(torch.from_numpy(params["b2"][0]))
    return model


def train(seed=0, n_hidden=4, lr=1.0, epochs=8000):
    model = build_model(seed, n_hidden)
    X = torch.from_numpy(xor_numpy.X)
    y = torch.from_numpy(xor_numpy.Y)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()                  # autograd calcule les gradients que backward() faisait à la main
        optimizer.step()
        history.append(loss.item())
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 21, 42, 99])
    parser.add_argument("--epochs", type=int, default=8000)
    args = parser.parse_args()
    for seed in args.seeds:
        model, history = train(seed=seed, epochs=args.epochs)
        preds = (model(torch.from_numpy(xor_numpy.X)) > 0.5).int().numpy()
        accuracy = float(np.mean(preds == xor_numpy.Y.astype(int)))
        print(f"seed={seed:>3}  perte finale={history[-1]:.6f}  accuracy={accuracy:.2f}")


if __name__ == "__main__":
    main()
