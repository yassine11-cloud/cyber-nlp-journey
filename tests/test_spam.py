import os
import tempfile

from phase2_nlp_classique.spam_baseline import SAMPLE, evaluate, load_dataset


def test_load_dataset_parses_label_and_text():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "SMSSpamCollection")
        with open(path, "w", encoding="utf-8") as f:
            f.write("ham\tSalut, tu viens ce soir ?\nspam\tGagnez 1000 euros maintenant\n")
        texts, labels = load_dataset(path)
    assert labels == ["ham", "spam"]
    assert texts[0].startswith("Salut")


def test_evaluate_returns_valid_metrics():
    texts = [t for _, t in SAMPLE]
    labels = [l for l, _ in SAMPLE]
    for ngram_range in [(1, 1), (1, 2)]:
        result = evaluate(texts, labels, ngram_range=ngram_range)
        assert result["n_train"] + result["n_test"] == len(SAMPLE)
        for key in ["accuracy", "precision_spam", "recall_spam", "f1_spam"]:
            assert 0.0 <= result[key] <= 1.0
        assert sum(sum(row) for row in result["confusion_matrix"]) == result["n_test"]
