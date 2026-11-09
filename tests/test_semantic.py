import zlib

import numpy as np
import pytest

pytest.importorskip("faiss")

from phase2_nlp_classique.text_utils import tokenize  # noqa: E402
from phase4_rag.rag import LocalRetriever  # noqa: E402
from phase4_rag.semantic import SemanticRetriever  # noqa: E402

STOPWORDS = set(LocalRetriever._stopwords())


def fake_encoder(texts, dim=512):
    """Encodeur de test déterministe (sac de mots haché, sans mots outils) : évite de télécharger un vrai modèle."""
    out = np.zeros((len(texts), dim), dtype="float32")
    for i, t in enumerate(texts):
        for w in tokenize(t):
            if w not in STOPWORDS:
                out[i, zlib.crc32(w.encode()) % dim] += 1.0
    return out


def test_faiss_search_returns_best_document_first():
    retriever = SemanticRetriever(encoder=fake_encoder, threshold=0.05)
    top = retriever.search("Qu'est-ce qu'un rançongiciel et comment s'en défendre ?", k=3)
    assert top[0][0] == "ransomware" and len(top) == 3


def test_semantic_retriever_refuses_below_threshold():
    retriever = SemanticRetriever(encoder=fake_encoder, threshold=0.9)
    assert retriever.answer("Donne-moi une recette de tarte aux pommes.")["refused"]
