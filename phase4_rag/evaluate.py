"""Évalue le retrieval (Precision@3, Recall@3, MRR) et le refus hors corpus.

Lancer :
    python -m phase4_rag.evaluate
    python -m phase4_rag.evaluate --output reports/rag_metrics.json
"""
import argparse
import json

from phase4_rag.eval_dataset import IN_CORPUS, NEAR_OUT_OF_CORPUS, OUT_OF_CORPUS
from phase4_rag.rag import LocalRetriever


def evaluate(retriever=None, k=3):
    retriever = retriever or LocalRetriever()
    precisions, recalls, reciprocal_ranks, accepted = [], [], [], 0
    for question, expected in IN_CORPUS:
        ids = [doc_id for doc_id, _ in retriever.search(question, k)]
        hit = expected in ids
        precisions.append((1 if hit else 0) / k)
        recalls.append(1.0 if hit else 0.0)
        reciprocal_ranks.append(1.0 / (ids.index(expected) + 1) if hit else 0.0)
        accepted += 0 if retriever.answer(question, k)["refused"] else 1
    refused_out = sum(1 for q in OUT_OF_CORPUS if retriever.answer(q, k)["refused"])
    near_refused = sum(1 for q in NEAR_OUT_OF_CORPUS if retriever.answer(q, k)["refused"])
    n_in, n_out = len(IN_CORPUS), len(OUT_OF_CORPUS)
    return {
        "k": k,
        "threshold": retriever.threshold,
        "n_questions_in_corpus": n_in,
        "n_questions_out_of_corpus": n_out,
        "precision_at_k": sum(precisions) / n_in,
        "recall_at_k": sum(recalls) / n_in,
        "mrr": sum(reciprocal_ranks) / n_in,
        "in_corpus_accepted": accepted / n_in,
        "out_of_corpus_refused": refused_out / n_out,
        "balanced_accuracy_refusal": 0.5 * (accepted / n_in) + 0.5 * (refused_out / n_out),
        "near_out_of_corpus_refused": near_refused / len(NEAR_OUT_OF_CORPUS),
    }


def score_summary(retriever):
    """Meilleur score de chaque question : sert à choisir le seuil de refus."""
    top = lambda q: retriever.search(q, 1)[0][1]
    in_scores = sorted(top(q) for q, _ in IN_CORPUS)
    return {
        "in_corpus_min": in_scores[0],
        "in_corpus_median": in_scores[len(in_scores) // 2],
        "out_of_corpus_max": max(top(q) for q in OUT_OF_CORPUS),
        "near_out_of_corpus_max": max(top(q) for q in NEAR_OUT_OF_CORPUS),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    metrics = evaluate()
    for key, value in metrics.items():
        print(f"{key:>28} : {value:.3f}" if isinstance(value, float) else f"{key:>28} : {value}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
