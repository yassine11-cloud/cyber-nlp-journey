"""NLP classique : détection de spam / smishing par sacs de mots et régression logistique.

On compare deux représentations (unigrammes seuls, puis unigrammes + bigrammes)
avec le même classifieur, sur le jeu public « SMS Spam Collection » (UCI).

Lancer (télécharge le jeu au premier lancement, connexion Internet requise) :
    python phase2_nlp_classique/spam_baseline.py
    python phase2_nlp_classique/spam_baseline.py --output reports/spam_baseline.json

Sans connexion, place toi-même le fichier SMSSpamCollection dans data/ (format : label<TAB>texte).
"""
import argparse
import io
import json
import os
import urllib.request
import zipfile

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_PATH = os.path.join("data", "SMSSpamCollection")

# Petit échantillon écrit à la main, utilisé uniquement par les tests hors ligne.
SAMPLE = [
    ("spam", "URGENT : votre compte est bloqué, cliquez sur ce lien pour gagner 500 euros"),
    ("spam", "Félicitations, vous avez gagné un iPhone gratuit, répondez GAGNE maintenant"),
    ("spam", "Votre colis est en attente, payez 1,99 euro de frais sur ce site"),
    ("spam", "Alerte sécurité : confirmez votre mot de passe immédiatement via ce lien"),
    ("spam", "Offre exclusive : crédit gratuit sans justificatif, appelez maintenant"),
    ("spam", "Gagnez un bon d'achat de 1000 euros, cliquez ici pour réclamer votre prix"),
    ("spam", "Votre banque vous informe d'un débit suspect, connectez-vous sur ce lien"),
    ("spam", "Dernière chance : abonnement gratuit, tapez STOP pour annuler les frais"),
    ("ham", "On se retrouve à 18h devant la gare, tu peux prendre le pain ?"),
    ("ham", "Merci pour le document, je te réponds demain après la réunion"),
    ("ham", "Peux-tu m'envoyer le compte rendu du cours de ce matin ?"),
    ("ham", "Je suis en retard, commence sans moi, j'arrive dans dix minutes"),
    ("ham", "Bon anniversaire ! On fête ça samedi chez moi, tu viens ?"),
    ("ham", "N'oublie pas d'apporter ton ordinateur pour la séance de demain"),
    ("ham", "J'ai fini le rapport, tu peux le relire avant ce soir ?"),
    ("ham", "Le dîner est prêt, on t'attend pour commencer"),
]


def download_dataset(dest=DATA_PATH):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    with archive.open("SMSSpamCollection") as src, open(dest, "wb") as out:
        out.write(src.read())
    return dest


def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        download_dataset(path)
    labels, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            label, _, text = line.rstrip("\n").partition("\t")
            if text:
                labels.append(label)
                texts.append(text)
    return texts, labels


def evaluate(texts, labels, ngram_range=(1, 1), test_size=0.25, seed=42):
    """Entraîne et évalue un pipeline TF-IDF + régression logistique."""
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    model = make_pipeline(
        TfidfVectorizer(ngram_range=ngram_range, lowercase=True, strip_accents="unicode"),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "ngram_range": list(ngram_range),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_spam": float(precision_score(y_test, y_pred, pos_label="spam", zero_division=0)),
        "recall_spam": float(recall_score(y_test, y_pred, pos_label="spam", zero_division=0)),
        "f1_spam": float(f1_score(y_test, y_pred, pos_label="spam", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["ham", "spam"]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    texts, labels = load_dataset()
    results = {
        "unigrammes": evaluate(texts, labels, (1, 1)),
        "unigrammes+bigrammes": evaluate(texts, labels, (1, 2)),
    }
    for name, r in results.items():
        print(f"{name:>22}  acc={r['accuracy']:.3f}  precision={r['precision_spam']:.3f}  "
              f"recall={r['recall_spam']:.3f}  f1={r['f1_spam']:.3f}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
