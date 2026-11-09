from phase4_rag.evaluate import evaluate, score_summary
from phase4_rag.rag import LocalRetriever
from phase4_rag.rag_pipeline import REFUSAL_MESSAGE, build_prompt, extractive_answer, rag_answer


def test_rag_answer_refuses_out_of_corpus():
    out = rag_answer("Quelle est la capitale de l'Australie ?", LocalRetriever(), use_llm=False)
    assert out["refused"] and out["answer"] == REFUSAL_MESSAGE and out["sources"] == []


def test_rag_answer_cites_source_without_llm():
    out = rag_answer("À quoi sert un SIEM ?", LocalRetriever(), use_llm=False)
    assert not out["refused"] and "[siem]" in out["answer"]


def test_prompt_contains_question_and_passage_ids():
    passages = [{"id": "siem", "title": "SIEM", "text": "Un SIEM centralise les journaux."}]
    prompt = build_prompt("Que fait un SIEM ?", passages)
    assert "Que fait un SIEM ?" in prompt and "[siem]" in prompt


def test_extractive_answer_uses_first_sentence():
    text = extractive_answer([{"id": "x", "title": "T", "text": "Première phrase. Deuxième phrase."}])
    assert text == "Première phrase. [x]"


def test_evaluation_reports_near_out_of_corpus_and_score_summary():
    assert 0.0 <= evaluate()["near_out_of_corpus_refused"] <= 1.0
    s = score_summary(LocalRetriever())
    assert s["in_corpus_min"] > s["out_of_corpus_max"]
