"""Questions annotées à la main : document attendu pour chaque question, et questions hors corpus."""

IN_CORPUS = [
    ("Comment reconnaître un email d'hameçonnage ?", "phishing"),
    ("Pourquoi un deuxième facteur protège-t-il même si le mot de passe est volé ?", "mfa"),
    ("Comment se défendre contre un rançongiciel qui chiffre les fichiers ?", "ransomware"),
    ("Que signifie ne faire confiance à aucun appareil par défaut dans le réseau ?", "zero_trust"),
    ("Quelle différence y a-t-il entre chiffrement symétrique et asymétrique ?", "chiffrement"),
    ("Que fait un pare-feu avec les ports et les protocoles ?", "pare_feu"),
    ("À quoi sert la corrélation des journaux dans un SIEM ?", "siem"),
    ("Comment prioriser les failles avec le score CVSS ?", "vulnerabilites"),
    ("Quelle est la différence entre un plan de continuité et un plan de reprise d'activité ?", "pca_pra"),
    ("Pourquoi ne donner que les droits strictement nécessaires aux utilisateurs ?", "moindre_privilege"),
    ("Combien de copies de données faut-il conserver selon la règle 3-2-1 ?", "sauvegarde"),
    ("Que fixe la norme ISO 27001 pour un système de management de la sécurité ?", "iso27001"),
    ("Comment traiter un risque identifié : réduire, transférer, accepter ou éviter ?", "analyse_risque"),
    ("Comment se protéger d'un appel qui joue sur l'urgence pour obtenir une information ?", "ingenierie_sociale"),
]

OUT_OF_CORPUS = [
    "Quelle est la capitale de l'Australie ?",
    "Donne-moi une recette de tarte aux pommes.",
    "Qui a gagné la coupe du monde de football en 2018 ?",
    "Comment apprendre à jouer de la guitare ?",
    "Quelle est la distance entre la Terre et la Lune ?",
    "Explique-moi la photosynthèse des plantes.",
]

# Questions proches de la cybersécurité mais dont la réponse n'est PAS dans le corpus : test de refus plus dur.
NEAR_OUT_OF_CORPUS = [
    "Comment configurer un VPN sous Linux ?",
    "Qu'est-ce que le RGPD et que impose-t-il aux entreprises ?",
    "Comment fonctionne la blockchain ?",
    "Comment installer un antivirus sur Windows ?",
    "Quels langages de programmation utilise-t-on en cybersécurité ?",
    "Comment devenir pentester ?",
]
