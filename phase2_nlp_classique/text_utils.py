"""Outils de texte partagés : tokenisation, vocabulaire, encodage à longueur fixe."""
import re
import unicodedata
from collections import Counter

PAD, UNK = "<pad>", "<unk>"


def strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def tokenize(text):
    return re.findall(r"[a-z0-9]+", strip_accents(text.lower()))


def build_vocab(texts, min_freq=1, max_size=None):
    """Associe chaque mot fréquent à un identifiant ; 0 = padding, 1 = mot inconnu."""
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    words = [w for w, c in counts.most_common() if c >= min_freq]
    if max_size is not None:
        words = words[:max_size]
    vocab = {PAD: 0, UNK: 1}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def encode(text, vocab, max_len=32):
    """Retourne (identifiants complétés par des 0, longueur réelle)."""
    ids = [vocab.get(w, 1) for w in tokenize(text)][:max_len]
    length = max(1, len(ids))
    ids = ids + [0] * (max_len - len(ids))
    return ids, length
