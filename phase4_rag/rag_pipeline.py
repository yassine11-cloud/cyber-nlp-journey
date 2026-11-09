"""Pipeline RAG complet : retrieval -> prompt -> génération.

1. Le retriever (TF-IDF ou sémantique + FAISS) trouve les passages pertinents, ou refuse.
2. On construit un prompt qui contient uniquement ces passages et demande de citer ses sources.
3. La génération est soit extractive (hors ligne, sans clé API), soit confiée à un LLM si la variable
   d'environnement ANTHROPIC_API_KEY est définie (modèle modifiable avec ANTHROPIC_MODEL).
"""
import os

from phase4_rag.corpus import DOCUMENTS

DOCS_BY_ID = {d["id"]: d for d in DOCUMENTS}
REFUSAL_MESSAGE = "Je ne trouve pas de réponse à cette question dans le corpus."


def build_prompt(question, passages):
    """Prompt qui impose de répondre uniquement à partir des passages fournis."""
    context = "\n\n".join(f"[{p['id']}] {p['title']} : {p['text']}" for p in passages)
    return (
        "Réponds à la question en utilisant uniquement les passages ci-dessous. "
        "Cite l'identifiant de chaque passage utilisé entre crochets. "
        "Si les passages ne contiennent pas la réponse, dis-le.\n\n"
        f"Passages :\n{context}\n\nQuestion : {question}"
    )


def extractive_answer(passages):
    """Réponse sans LLM : la première phrase du passage le mieux classé, avec sa source."""
    best = passages[0]
    first_sentence = best["text"].split(". ")[0].rstrip(".") + "."
    return f"{first_sentence} [{best['id']}]"


def generate_with_llm(prompt, max_tokens=400):
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def rag_answer(question, retriever, k=3, use_llm=None):
    result = retriever.answer(question, k)
    if result["refused"]:
        return {"answer": REFUSAL_MESSAGE, "refused": True, "sources": []}
    passages = [DOCS_BY_ID[doc_id] for doc_id, _ in result["sources"]]
    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    answer = generate_with_llm(build_prompt(question, passages)) if use_llm else extractive_answer(passages)
    return {"answer": answer, "refused": False, "sources": result["sources"]}


def main():
    import argparse
    from phase4_rag.rag import LocalRetriever

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--semantic", action="store_true", help="Utilise embeddings + FAISS (requirements-deep.txt)")
    parser.add_argument("--threshold", type=float, default=None, help="Seuil de refus (par défaut : celui du retriever)")
    parser.add_argument("--show-scores", action="store_true", help="Affiche les 3 meilleurs scores, même en cas de refus")
    args = parser.parse_args()
    options = {} if args.threshold is None else {"threshold": args.threshold}
    if args.semantic:
        from phase4_rag.semantic import SemanticRetriever
        retriever = SemanticRetriever(**options)
    else:
        retriever = LocalRetriever(**options)
    if args.show_scores:
        for doc_id, score in retriever.search(args.question, 3):
            print(f"  score : {doc_id} = {score:.3f} (seuil de refus : {retriever.threshold})")
    out = rag_answer(args.question, retriever)
    print(out["answer"])
    for doc_id, score in out["sources"]:
        print(f"  source : {doc_id} (score {score:.3f})")


if __name__ == "__main__":
    main()
