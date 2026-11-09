from phase2_nlp_classique.data_fr import generate_dataset
from phase2_nlp_classique.embeddings_numpy import build_pairs, most_similar, sentences_to_ids, train_skipgram
from phase2_nlp_classique.text_utils import build_vocab


def _setup():
    texts = [r["text"] for r in generate_dataset(n_per_class=40, seed=0)]
    vocab = build_vocab(texts, min_freq=2)
    return vocab, sentences_to_ids(texts, vocab)


def test_build_pairs_respects_window():
    pairs = build_pairs([[5, 6, 7]], window=1)
    assert sorted(map(tuple, pairs.tolist())) == [(5, 6), (6, 5), (6, 7), (7, 6)]


def test_training_reduces_loss_and_gives_embedding_matrix():
    vocab, sentences = _setup()
    embeddings, history = train_skipgram(sentences, len(vocab), dim=8, epochs=3, seed=0)
    assert embeddings.shape == (len(vocab), 8)
    assert history[-1] < history[0]


def test_most_similar_excludes_the_word_itself():
    vocab, sentences = _setup()
    embeddings, _ = train_skipgram(sentences, len(vocab), dim=8, epochs=1, seed=0)
    word = "compte" if "compte" in vocab else next(w for w in vocab if w not in ("<pad>", "<unk>"))
    neighbours = most_similar(word, embeddings, vocab, k=4)
    assert len(neighbours) == 4 and word not in [w for w, _ in neighbours]
