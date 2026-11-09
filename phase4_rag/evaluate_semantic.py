"""Compare le retrieval TF-IDF et le retrieval sémantique (embeddings + FAISS) sur les mêmes questions.

Lancer (nécessite requirements-deep.txt, télécharge un modèle au premier lancement) :
    python -m phase4_rag.evaluate_semantic
    python -m phase4_rag.evaluate_semantic --threshold 0.40

Regarde d'abord le résumé des scores : un bon seuil est situé entre « out_of_corpus_max »,
« near_out_of_corpus_max » et « in_corpus_min ». S'ils se chevauchent, aucun seuil ne sépare tout :
c'est une limite à documenter.
"""
import argparse
import json

from phase4_rag.evaluate import evaluate, score_summary
from phase4_rag.rag import LocalRetriever
from phase4_rag.semantic import SemanticRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--output", default="reports/rag_comparison.json")
    args = parser.parse_args()

    retrievers = {"tfidf": LocalRetriever(), "semantique_faiss": SemanticRetriever(threshold=args.threshold)}
    report = {}
    for name, retriever in retrievers.items():
        report[name] = {"metrics": evaluate(retriever), "scores": score_summary(retriever)}
        print(f"\n== {name} ==")
        for key, value in {**report[name]["metrics"], **report[name]["scores"]}.items():
            print(f"{key:>28} : {value:.3f}" if isinstance(value, float) else f"{key:>28} : {value}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
