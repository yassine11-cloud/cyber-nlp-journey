from phase4_rag.evaluate import evaluate
from phase4_rag.rag import LocalRetriever


def test_search_finds_expected_document():
    retriever = LocalRetriever()
    top = retriever.search("Comment se défendre contre un rançongiciel ?", k=3)
    assert top[0][0] == "ransomware"


def test_out_of_corpus_question_is_refused():
    retriever = LocalRetriever()
    assert retriever.answer("Donne-moi une recette de tarte aux pommes.")["refused"]


def test_in_corpus_question_is_answered_with_sources():
    retriever = LocalRetriever()
    result = retriever.answer("À quoi sert un SIEM ?")
    assert not result["refused"]
    assert result["sources"][0][0] == "siem"


def test_evaluation_metrics():
    metrics = evaluate()
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] >= 0.9
    assert metrics["balanced_accuracy_refusal"] >= 0.9
