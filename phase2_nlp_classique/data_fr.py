"""Jeu de données français synthétique : smishing (spam) contre messages légitimes.

Les messages sont générés à partir de gabarits écrits pour ce projet, avec du bruit (fautes de frappe,
accents supprimés, majuscules). Les messages légitimes contiennent volontairement des mots « piégeux »
(compte, lien, colis, virement) pour éviter une séparation trop triviale.

La séparation entraînement / validation / test se fait par gabarit : le test contient des gabarits
jamais vus à l'entraînement, ce qui mesure la généralisation et pas la mémorisation.

Limite : données synthétiques, donc les scores restent optimistes par rapport à de vrais messages.
"""
import random
import unicodedata

NAMES = ["Karim", "Salma", "Youssef", "Nadia", "Omar", "Imane", "Hamza", "Sara", "Mehdi", "Lina"]
PLACES = ["la gare", "la bibliothèque", "le campus", "le café", "la salle B12", "l'amphi 3"]
TIMES = ["18h", "19h30", "14h", "9h", "20h"]
AMOUNTS = ["49", "120", "350", "1 000", "2 500", "75"]
BRANDS = ["Banque Atlas", "PostePay", "Chronopost", "Amazone", "TelcoMobile", "ImpôtsInfo"]
DOMAINS = ["secure-verif.top", "maj-compte.xyz", "colis-suivi.info", "gain-promo.click", "bit.ly/3xYz"]

SPAM_TEMPLATES = [
    "{brand} : activité suspecte sur votre compte. Confirmez vos identifiants sous 24h sur {domain}",
    "Félicitations {name}, vous avez gagné {amount} euros ! Réclamez votre prix ici : {domain}",
    "Votre colis est bloqué, réglez {amount} centimes de frais de douane sur {domain}",
    "URGENT {brand} : votre carte sera désactivée aujourd'hui, mettez-la à jour sur {domain}",
    "Remboursement de {amount} euros en attente, saisissez votre IBAN sur {domain}",
    "Dernier rappel : votre abonnement sera facturé {amount} euros, annulez sur {domain}",
    "Alerte sécurité : un appareil inconnu s'est connecté, vérifiez votre mot de passe sur {domain}",
    "{name}, vous êtes présélectionné pour un crédit gratuit de {amount} euros, répondez OUI",
]

HAM_TEMPLATES = [
    "On se retrouve à {time} devant {place}, tu peux prendre le pain ?",
    "{name}, peux-tu m'envoyer le compte rendu avant {time} ?",
    "Ton colis est arrivé chez le voisin, passe le chercher ce soir",
    "J'ai bien reçu ton virement de {amount} euros, merci",
    "Le lien du cours est dans le mail du prof, la séance est à {time}",
    "Je suis en retard, commence sans moi, j'arrive à {place} dans dix minutes",
    "Ma banque m'a appelé pour mettre à jour mon dossier, je passe à l'agence demain",
    "N'oublie pas le mot de passe du wifi de {place}, il est écrit au tableau",
    "{name} a partagé un document avec toi sur le drive de la promo",
    "Rappel : réunion de projet à {time}, apporte ton ordinateur",
]


def _strip_accents(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def _noise(rng, text):
    if rng.random() < 0.10:
        text = _strip_accents(text)
    if rng.random() < 0.10 and len(text) > 6:
        i = rng.randrange(1, len(text) - 2)
        text = text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if rng.random() < 0.10:
        text = rng.choice(["Salut, ", "Hey, ", "Bonjour, "]) + text[0].lower() + text[1:]
    if rng.random() < 0.05:
        text = text.upper()
    return text


def generate_dataset(n_per_class=800, seed=0):
    """Retourne une liste de dictionnaires {text, label, template}."""
    rng = random.Random(seed)
    rows = []
    for label, templates in (("spam", SPAM_TEMPLATES), ("ham", HAM_TEMPLATES)):
        for _ in range(n_per_class):
            tid = rng.randrange(len(templates))
            text = templates[tid].format(
                name=rng.choice(NAMES), place=rng.choice(PLACES), time=rng.choice(TIMES),
                amount=rng.choice(AMOUNTS), brand=rng.choice(BRANDS), domain=rng.choice(DOMAINS),
            )
            rows.append({"text": _noise(rng, text), "label": label, "template": f"{label}-{tid}"})
    return rows


def split_dataset(rows, heldout_per_class=2, val_fraction=0.15, seed=0):
    """Test = derniers gabarits de chaque classe (jamais vus). Entraînement / validation = le reste."""
    heldout = {f"spam-{len(SPAM_TEMPLATES) - 1 - i}" for i in range(heldout_per_class)}
    heldout |= {f"ham-{len(HAM_TEMPLATES) - 1 - i}" for i in range(heldout_per_class)}
    test = [r for r in rows if r["template"] in heldout]
    pool = [r for r in rows if r["template"] not in heldout]
    random.Random(seed).shuffle(pool)
    n_val = int(len(pool) * val_fraction)
    return pool[n_val:], pool[:n_val], test


def to_xy(rows):
    """Textes et étiquettes entières (spam = 1, ham = 0)."""
    return [r["text"] for r in rows], [1 if r["label"] == "spam" else 0 for r in rows]
