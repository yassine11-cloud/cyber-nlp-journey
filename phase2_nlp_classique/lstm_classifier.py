"""Classification de texte avec embeddings appris et LSTM (PyTorch).

Pipeline : texte -> identifiants de mots -> couche d'embeddings -> LSTM -> dernier état caché -> score.
Même jeu français et mêmes gabarits de test que le TF-IDF et le Transformer, pour une comparaison juste.

Lancer (nécessite requirements-deep.txt) :
    python -m phase2_nlp_classique.lstm_classifier
Écrit reports/lstm_fr_metrics.json.
"""
import argparse
import json
import random

import numpy as np
import torch
from torch import nn

from phase2_nlp_classique.data_fr import generate_dataset, split_dataset, to_xy
from phase2_nlp_classique.metrics import compute_metrics
from phase2_nlp_classique.text_utils import build_vocab, encode


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=64, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)          # h_n : (1, batch, hidden) = dernier état de chaque séquence
        return self.fc(h_n[-1]).squeeze(1)       # un score (logit) par message


def make_tensors(texts, labels, vocab, max_len):
    encoded = [encode(t, vocab, max_len) for t in texts]
    x = torch.tensor([e[0] for e in encoded], dtype=torch.long)
    lengths = torch.tensor([e[1] for e in encoded], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.float32)
    return x, lengths, y


def predict(model, x, lengths, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            logits = model(x[i:i + batch_size], lengths[i:i + batch_size])
            preds.extend((logits > 0).long().tolist())
    return preds


def run(seed=0, epochs=10, emb_dim=64, hidden=64, lr=0.01, batch_size=32, max_len=32, word_dropout=0.0,
        verbose=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train, val, test = split_dataset(generate_dataset(seed=seed), seed=seed)
    X_train, y_train = to_xy(train)
    X_val, y_val = to_xy(val)
    X_test, y_test = to_xy(test)
    vocab = build_vocab(X_train, min_freq=1)
    xt, lt, yt = make_tensors(X_train, y_train, vocab, max_len)
    xv, lv, _ = make_tensors(X_val, y_val, vocab, max_len)
    xs, ls, _ = make_tensors(X_test, y_test, vocab, max_len)

    model = LSTMClassifier(len(vocab), emb_dim, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(xt))
        total = 0.0
        for i in range(0, len(xt), batch_size):
            idx = perm[i:i + batch_size]
            xb = xt[idx]
            if word_dropout > 0:                      # remplace au hasard des mots par <unk> : force à ne pas
                drop = (torch.rand(xb.shape) < word_dropout) & (xb != 0)   # mémoriser des mots précis
                xb = xb.masked_fill(drop, 1)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, lt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        val_acc = float(np.mean(np.array(predict(model, xv, lv)) == np.array(y_val)))
        history.append({"epoch": epoch, "train_loss": total / len(xt), "val_accuracy": val_acc})
        if verbose:
            print(f"époque {epoch:>2}  perte={total / len(xt):.4f}  accuracy validation={val_acc:.3f}")
    name = "lstm" if word_dropout == 0 else f"lstm_wd{int(round(word_dropout * 100))}"
    metrics = compute_metrics(y_test, predict(model, xs, ls), name, templates=[r["template"] for r in test], seed=seed, epochs=epochs,
                              vocab_size=len(vocab), word_dropout=word_dropout, history=history)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--word-dropout", type=float, default=0.0,
                        help="Probabilité de remplacer un mot par <unk> à l'entraînement (expérience, ex. 0.3)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = run(seed=args.seed, epochs=args.epochs, word_dropout=args.word_dropout)
    if args.output is None:
        args.output = ("reports/lstm_fr_metrics.json" if args.word_dropout == 0
                       else f"reports/lstm_wd{int(round(args.word_dropout * 100))}_fr_metrics.json")
    print({k: v for k, v in metrics.items() if k != "history"})
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
