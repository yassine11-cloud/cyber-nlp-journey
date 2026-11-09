import pytest

torch = pytest.importorskip("torch")

from phase2_nlp_classique.lstm_classifier import LSTMClassifier, run  # noqa: E402


def test_forward_returns_one_logit_per_message():
    model = LSTMClassifier(vocab_size=20, emb_dim=8, hidden=8)
    x = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
    lengths = torch.tensor([3, 2])
    assert model(x, lengths).shape == (2,)


def test_one_epoch_run_returns_valid_metrics():
    metrics = run(seed=0, epochs=1, verbose=False)
    assert 0.0 <= metrics["accuracy"] <= 1.0 and metrics["n_test"] > 0


def test_word_dropout_variant_runs_and_is_named():
    metrics = run(seed=0, epochs=1, word_dropout=0.3, verbose=False)
    assert metrics["model"] == "lstm_wd30" and 0.0 <= metrics["accuracy"] <= 1.0
