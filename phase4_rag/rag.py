"""Recherche d'information locale (TF-IDF) avec refus des questions hors corpus.

C'est la brique « retrieval » d'un RAG : on retrouve les passages pertinents et on refuse de
répondre quand aucun document n'est assez proche de la question. La génération par un LLM
est volontairement laissée de côté pour garder un projet reproductible sans clé API.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from phase4_rag.corpus import DOCUMENTS

# Mots outils français les plus fréquents : ils ne portent pas de sens et gonflent la similarité.
FRENCH_STOPWORDS = [
    "a", "à", "au", "aux", "avec", "ce", "ces", "cette", "comment", "dans", "de", "des", "du", "donne", "elle", "en",
    "est", "et", "faire", "il", "je", "l", "la", "le", "les", "leur", "lui", "ma", "me", "mon", "ne", "ni", "nous",
    "on", "ou", "par", "pas", "pour", "pourquoi", "qu", "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "sa",
    "se", "ses", "si", "son", "sur", "ta", "te", "ton", "tu", "un", "une", "vous", "y", "entre", "faut", "fait", "sont",
    "peut", "moi", "explique", "selon", "d", "c", "s", "n", "t", "m", "j",
]


class LocalRetriever:
    def __init__(self, documents=DOCUMENTS, threshold=0.10):
        self.documents = documents
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1, 2),
                                          sublinear_tf=True, stop_words=self._stopwords())
        self.matrix = self.vectorizer.fit_transform([f"{d['title']}. {d['text']}" for d in documents])

    @staticmethod
    def _stopwords():
        from sklearn.feature_extraction.text import strip_accents_unicode
        return sorted({strip_accents_unicode(w.lower()) for w in FRENCH_STOPWORDS})

    def search(self, question, k=3):
        """Retourne les k meilleurs documents avec leur score de similarité cosinus."""
        scores = cosine_similarity(self.vectorizer.transform([question]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [(self.documents[i]["id"], float(scores[i])) for i in order]

    def answer(self, question, k=3):
        """Retourne les sources ou un refus si le meilleur score est sous le seuil."""
        results = self.search(question, k)
        if not results or results[0][1] < self.threshold:
            return {"refused": True, "sources": []}
        return {"refused": False, "sources": results}
