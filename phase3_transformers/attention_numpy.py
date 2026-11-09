"""Mécanisme d'attention et attention multi-têtes écrits à la main en NumPy.

Attention : pour chaque position, une moyenne pondérée des valeurs V, avec des poids obtenus par
    softmax(Q·Kᵀ / √d_k)
Le facteur √d_k évite que les produits scalaires grandissent avec la dimension et saturent le softmax.
Le masque causal empêche une position de regarder les positions suivantes (cas d'un modèle génératif).
"""
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)    # stabilité numérique
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n):
    """Matrice booléenne : True là où l'attention est interdite (positions futures)."""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q, K, V : tableaux (n, d). Retourne (sortie (n, d_v), poids d'attention (n, n))."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def init_mha_params(d_model, n_heads, seed=0):
    assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(d_model)
    return {name: rng.normal(0.0, scale, (d_model, d_model)) for name in ("Wq", "Wk", "Wv", "Wo")}


def multi_head_attention(X, params, n_heads, mask=None):
    """X : (n, d_model). Chaque tête travaille sur un sous-espace de taille d_model / n_heads."""
    n, d_model = X.shape
    d_head = d_model // n_heads
    Q, K, V = X @ params["Wq"], X @ params["Wk"], X @ params["Wv"]
    outputs, all_weights = [], []
    for h in range(n_heads):
        sl = slice(h * d_head, (h + 1) * d_head)
        out, w = scaled_dot_product_attention(Q[:, sl], K[:, sl], V[:, sl], mask)
        outputs.append(out)
        all_weights.append(w)
    return np.concatenate(outputs, axis=-1) @ params["Wo"], np.stack(all_weights)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 8))
    params = init_mha_params(8, 2)
    out, weights = multi_head_attention(X, params, n_heads=2, mask=causal_mask(5))
    print("sortie :", out.shape, "| poids :", weights.shape)
    print("poids de la tête 0 (triangulaire à cause du masque causal) :")
    print(np.round(weights[0], 2))
