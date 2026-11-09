"""Baseline TF-IDF + régression logistique sur le jeu français (gabarits de test jamais vus).

Lancer :
    python -m phase2_nlp_classique.benchmark_fr
Écrit reports/tfidf_fr_metrics.json, au même format que le LSTM et le Transformer.
"""
import argparse
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from phase2_nlp_classique.data_fr import generate_dataset, split_dataset, to_xy
from phase2_nlp_classique.metrics import compute_metrics


def run(seed=0, ngram_range=(1, 2)):
    train, val, test = split_dataset(generate_dataset(seed=seed), seed=seed)
    X_train, y_train = to_xy(train)
    X_test, y_test = to_xy(test)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=ngram_range, strip_accents="unicode", lowercase=True),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X_train, y_train)
    return compute_metrics(y_test, model.predict(X_test), "tfidf_logreg", templates=[r["template"] for r in test], seed=seed, ngram_range=list(ngram_range))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="reports/tfidf_fr_metrics.json")
    args = parser.parse_args()
    metrics = run(args.seed)
    print(json.dumps(metrics, indent=2))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
