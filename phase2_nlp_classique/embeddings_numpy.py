"""Word embeddings appris depuis zéro : skip-gram avec échantillonnage négatif, en NumPy.

Idée : deux mots qui apparaissent dans des contextes proches reçoivent des vecteurs proches.
Pour chaque couple (mot central c, mot de contexte o), on maximise la probabilité que (c, o) soit un
vrai couple et que k mots tirés au hasard n'en soient pas :
    perte = -log σ(u_o · v_c) - Σ_k log σ(-u_n · v_c)

Lancer :
    python -m phase2_nlp_classique.embeddings_numpy --n-sentences 600 --epochs 5
"""
import argparse

import numpy as np

from phase2_nlp_classique.data_fr import generate_dataset
from phase2_nlp_classique.text_utils import build_vocab, tokenize


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sentences_to_ids(texts, vocab):
    return [[vocab[w] for w in tokenize(t) if w in vocab] for t in texts]


def build_pairs(sentences, window=2):
    pairs = []
    for s in sentences:
        for i, center in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i:
                    pairs.append((center, s[j]))
    return np.array(pairs, dtype=np.int64)


def train_skipgram(sentences, vocab_size, dim=32, window=2, negatives=5, epochs=5, lr=0.05, seed=0):
    """Retourne la matrice d'embeddings (un vecteur par mot) et la perte moyenne par époque."""
    rng = np.random.default_rng(seed)
    pairs = build_pairs(sentences, window)
    counts = np.bincount(pairs[:, 1], minlength=vocab_size).astype(float)
    probs = counts ** 0.75                      # les mots rares sont un peu sur-échantillonnés
    probs /= probs.sum()
    W_in = rng.normal(0.0, 0.1, (vocab_size, dim))
    W_out = rng.normal(0.0, 0.1, (vocab_size, dim))
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(pairs))
        negs = rng.choice(vocab_size, size=(len(pairs), negatives), p=probs)
        total = 0.0
        for idx in order:
            c, o = pairs[idx]
            n = negs[idx]
            v = W_in[c]
            s_pos = sigmoid(W_out[o] @ v)
            s_neg = sigmoid(W_out[n] @ v)
            total += -np.log(s_pos + 1e-10) - np.sum(np.log(1.0 - s_neg + 1e-10))
            grad_v = (s_pos - 1.0) * W_out[o] + (s_neg[:, None] * W_out[n]).sum(axis=0)
            W_out[o] -= lr * (s_pos - 1.0) * v
            np.subtract.at(W_out, n, lr * s_neg[:, None] * v)
            W_in[c] -= lr * grad_v
        history.append(total / len(pairs))
    return W_in, history


def most_similar(word, embeddings, vocab, k=5):
    """Les k mots les plus proches (similarité cosinus), sans le mot lui-même."""
    inverse = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    unit = embeddings / norms
    sims = unit @ unit[vocab[word]]
    order = [i for i in np.argsort(-sims) if i != vocab[word] and i > 1]
    return [(inverse[i], float(sims[i])) for i in order[:k]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sentences", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dim", type=int, default=32)
    args = parser.parse_args()

    texts = [r["text"] for r in generate_dataset(n_per_class=args.n_sentences // 2)]
    vocab = build_vocab(texts, min_freq=2)
    sentences = sentences_to_ids(texts, vocab)
    embeddings, history = train_skipgram(sentences, len(vocab), dim=args.dim, epochs=args.epochs)
    for epoch, loss in enumerate(history, 1):
        print(f"époque {epoch} : perte moyenne = {loss:.4f}")
    for word in ["compte", "colis", "lien", "mot", "gagne"]:
        if word in vocab:
            print(word, "->", [w for w, _ in most_similar(word, embeddings, vocab)])


if __name__ == "__main__":
    main()
