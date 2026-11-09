from phase2_nlp_classique.data_fr import generate_dataset, split_dataset, to_xy
from phase2_nlp_classique.text_utils import build_vocab, encode, tokenize


def test_generation_is_deterministic_and_balanced():
    a = generate_dataset(n_per_class=50, seed=1)
    b = generate_dataset(n_per_class=50, seed=1)
    assert a == b
    assert sum(r["label"] == "spam" for r in a) == 50
    assert sum(r["label"] == "ham" for r in a) == 50


def test_test_set_uses_unseen_templates():
    train, val, test = split_dataset(generate_dataset(n_per_class=200, seed=0))
    seen = {r["template"] for r in train} | {r["template"] for r in val}
    assert test and not ({r["template"] for r in test} & seen)


def test_to_xy_encodes_spam_as_one():
    texts, y = to_xy([{"text": "a", "label": "spam", "template": "spam-0"},
                      {"text": "b", "label": "ham", "template": "ham-0"}])
    assert texts == ["a", "b"] and y == [1, 0]


def test_tokenize_strips_accents_and_case():
    assert tokenize("Félicitations, VOUS avez gagné !") == ["felicitations", "vous", "avez", "gagne"]


def test_vocab_and_encode_pad_and_unknown():
    vocab = build_vocab(["bonjour le monde", "bonjour tout le monde"])
    ids, length = encode("bonjour inconnu", vocab, max_len=5)
    assert len(ids) == 5 and length == 2
    assert ids[0] == vocab["bonjour"] and ids[1] == 1 and ids[2:] == [0, 0, 0]
