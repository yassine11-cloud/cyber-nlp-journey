"""Fine-tuning d'un Transformer pré-entraîné (CamemBERT par défaut) pour détecter le smishing en français.

Principe : on garde le modèle pré-entraîné sur du français et on l'entraîne quelques époques de plus,
avec une petite tête de classification, sur notre tâche. Même jeu et mêmes gabarits de test que le
TF-IDF et le LSTM.

Lancer (GPU recommandé, par exemple Google Colab : Exécution > Modifier le type d'exécution > GPU) :
    pip install -r requirements.txt -r requirements-deep.txt
    python -m phase3_transformers.finetune_transformer --epochs 3
Si le nom du modèle est refusé, essaie --model almanach/camembert-base.
Écrit reports/camembert_fr_metrics.json.
"""
import argparse
import json
import math
import random

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from phase2_nlp_classique.data_fr import generate_dataset, split_dataset, to_xy
from phase2_nlp_classique.metrics import compute_metrics


def predict(model, tokenizer, texts, device, max_len, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = tokenizer(texts[i:i + batch_size], padding=True, truncation=True, max_length=max_len,
                              return_tensors="pt").to(device)
            preds.extend(model(**batch).logits.argmax(dim=-1).tolist())
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="camembert-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="reports/camembert_fr_metrics.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("appareil :", device)

    train, val, test = split_dataset(generate_dataset(seed=args.seed), seed=args.seed)
    X_train, y_train = to_xy(train)
    X_val, y_val = to_xy(val)
    X_test, y_test = to_xy(test)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(X_train) / args.batch_size)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train)).tolist()
        total = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            batch = tokenizer([X_train[j] for j in idx], padding=True, truncation=True, max_length=args.max_len,
                              return_tensors="pt").to(device)
            labels = torch.tensor([y_train[j] for j in idx], device=device)
            loss = model(**batch, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total += loss.item() * len(idx)
        val_acc = float(np.mean(np.array(predict(model, tokenizer, X_val, device, args.max_len)) == np.array(y_val)))
        history.append({"epoch": epoch, "train_loss": total / len(perm), "val_accuracy": val_acc})
        print(f"époque {epoch}  perte={total / len(perm):.4f}  accuracy validation={val_acc:.3f}")

    metrics = compute_metrics(y_test, predict(model, tokenizer, X_test, device, args.max_len), "camembert_finetuned",
                              templates=[r["template"] for r in test], base_model=args.model, seed=args.seed, epochs=args.epochs, history=history)
    print({k: v for k, v in metrics.items() if k != "history"})
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
