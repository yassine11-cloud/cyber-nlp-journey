"""Réseau de neurones à une couche cachée écrit from scratch avec NumPy.

Tâche : apprendre la fonction XOR, qu'aucun neurone seul ne peut représenter
(les deux classes ne sont pas séparables par une droite).

Lancer :
    python phase1_fondamentaux/xor_numpy.py
    python phase1_fondamentaux/xor_numpy.py --seeds 1 7 21 42 99 --save-figure reports/figures/xor_loss.png
"""
import argparse
import json

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def init_params(n_in, n_hidden, n_out, seed):
    """Poids aléatoires (pour casser la symétrie), biais nuls."""
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 1.0, (n_in, n_hidden)),
        "b1": np.zeros((1, n_hidden)),
        "W2": rng.normal(0.0, 1.0, (n_hidden, n_out)),
        "b2": np.zeros((1, n_out)),
    }


def forward(X, params):
    """Propagation avant. Retourne la sortie et le cache pour la rétropropagation."""
    a1 = sigmoid(X @ params["W1"] + params["b1"])
    a2 = sigmoid(a1 @ params["W2"] + params["b2"])
    return a2, (a1, a2)


def loss(y_pred, y):
    """Erreur quadratique moyenne."""
    return float(np.mean((y_pred - y) ** 2))


def backward(X, y, params, cache):
    """Rétropropagation : gradients de la perte par rapport à chaque paramètre."""
    a1, a2 = cache
    n = X.shape[0]
    dz2 = (2.0 / n) * (a2 - y) * a2 * (1.0 - a2)       # sigmoid'(z) = a * (1 - a)
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)
    dz1 = (dz2 @ params["W2"].T) * a1 * (1.0 - a1)
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def train(X=X, y=Y, n_hidden=4, lr=1.0, epochs=8000, seed=0):
    """Descente de gradient. Retourne les paramètres appris et l'historique de la perte."""
    params = init_params(X.shape[1], n_hidden, y.shape[1], seed)
    history = []
    for _ in range(epochs):
        y_pred, cache = forward(X, params)
        history.append(loss(y_pred, y))
        grads = backward(X, y, params, cache)
        for k in params:
            params[k] -= lr * grads[k]
    return params, history


def predict(X, params):
    y_pred, _ = forward(X, params)
    return (y_pred > 0.5).astype(int)


def run_seeds(seeds, **kwargs):
    results = []
    for seed in seeds:
        params, history = train(seed=seed, **kwargs)
        accuracy = float(np.mean(predict(X, params) == Y))
        results.append({"seed": seed, "final_loss": history[-1], "accuracy": accuracy, "history": history})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 21, 42, 99])
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--save-figure", default=None)
    parser.add_argument("--output", default=None, help="Fichier JSON de résultats")
    args = parser.parse_args()

    results = run_seeds(args.seeds, n_hidden=args.hidden, lr=args.lr, epochs=args.epochs)
    for r in results:
        print(f"seed={r['seed']:>3}  perte finale={r['final_loss']:.6f}  accuracy={r['accuracy']:.2f}")
    mean_loss = float(np.mean([r["final_loss"] for r in results]))
    print(f"perte finale moyenne : {mean_loss:.6f}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in r.items() if k != "history"} for r in results], f, indent=2)
    if args.save_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for r in results:
            plt.plot(r["history"], label=f"seed {r['seed']}")
        plt.yscale("log")
        plt.xlabel("époque")
        plt.ylabel("perte (MSE)")
        plt.title("Réseau XOR : courbes de perte")
        plt.legend()
        plt.savefig(args.save_figure, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
