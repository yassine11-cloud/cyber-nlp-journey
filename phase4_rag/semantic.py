"""Recherche sémantique : embeddings de phrases (sentence-transformers) + index FAISS.

Contrairement à TF-IDF, qui compare des mots exacts, un embedding de phrase place des textes de sens
proche dans des zones proches de l'espace vectoriel, même sans mot en commun. On normalise les
vecteurs : le produit scalaire (IndexFlatIP) devient alors la similarité cosinus.

Même interface que LocalRetriever (search / answer), donc phase4_rag.evaluate marche pour les deux.
"""
import numpy as np

from phase4_rag.corpus import DOCUMENTS

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # modèle multilingue, gère le français


def transformers_encoder(model_name, max_length=128):
    """Encodeur de repli : le même calcul que sentence-transformers (moyenne des vecteurs de tokens)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()

    def encode(texts):
        with torch.no_grad():
            batch = tokenizer(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            return ((hidden * mask).sum(dim=1) / mask.sum(dim=1)).numpy()

    return encode


class SemanticRetriever:
    def __init__(self, documents=DOCUMENTS, model_name=DEFAULT_MODEL, threshold=0.35, encoder=None):
        """`encoder` : fonction facultative liste[str] -> tableau (n, d), pour tester sans télécharger de modèle."""
        import faiss

        self.documents = documents
        self.threshold = threshold
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name)
                encoder = lambda texts: model.encode(texts, convert_to_numpy=True)
            except Exception as error:               # incompatibilité de versions, téléchargement interrompu...
                print(f"sentence-transformers indisponible ({type(error).__name__}) : repli sur transformers.")
                encoder = transformers_encoder(model_name)
        self._encode = lambda texts: self._normalise(np.asarray(encoder(texts), dtype="float32"))
        vectors = self._encode([f"{d['title']}. {d['text']}" for d in documents])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    @staticmethod
    def _normalise(vectors):
        return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)

    def search(self, question, k=3):
        scores, ids = self.index.search(self._encode([question]), k)
        return [(self.documents[i]["id"], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def answer(self, question, k=3):
        results = self.search(question, k)
        if not results or results[0][1] < self.threshold:
            return {"refused": True, "sources": []}
        return {"refused": False, "sources": results}
