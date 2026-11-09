# Cyber-NLP Journey

Parcours progressif en deep learning appliqué au NLP, avec un fil rouge cybersécurité : détecter du smishing (SMS d'hameçonnage) et retrouver des réponses dans des notes de sécurité.

Du réseau écrit à la main en NumPy jusqu'au Transformer fine-tuné et au pipeline RAG avec FAISS, chaque brique est testée et comparée aux autres sur le même jeu de données, avec des références triviales pour ne pas se tromper sur ce que signifie un bon score.

> Projet réalisé avec l'assistance de Claude (Anthropic). L'auteur exécute, teste et documente chaque brique.

## Contenu

| Phase | Contenu | Fichier |
|---|---|---|
| 1. Fondamentaux | Réseau XOR from scratch (NumPy), puis le même en PyTorch avec autograd | `phase1_fondamentaux/xor_numpy.py`, `xor_pytorch.py` |
| 2. NLP classique et RNN | Baseline TF-IDF + régression logistique, word embeddings skip-gram from scratch, classifieur LSTM | `phase2_nlp_classique/` |
| 3. Transformers | Attention et attention multi-têtes from scratch, fine-tuning de CamemBERT | `phase3_transformers/` |
| 4. RAG | Retrieval TF-IDF, retrieval sémantique (embeddings + FAISS), refus hors corpus, pipeline retrieval, prompt, génération | `phase4_rag/` |

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    macOS / Linux : source .venv/bin/activate
pip install -r requirements.txt                     # phases NumPy et scikit-learn
pip install -r requirements-deep.txt                # PyTorch, Transformers, FAISS (phases deep learning)
```

Le fine-tuning de CamemBERT est beaucoup plus rapide avec un GPU : utilise Google Colab (Exécution, Modifier le type d'exécution, GPU).

## Commandes

```bash
pytest -q                                                          # les tests PyTorch et FAISS sont sautés si les paquets manquent

# Phase 1
python phase1_fondamentaux/xor_numpy.py --save-figure reports/figures/xor_loss.png
python -m phase1_fondamentaux.xor_pytorch

# Phase 2 (jeu français, gabarits de test jamais vus)
python -m phase2_nlp_classique.benchmark_fr                        # TF-IDF + régression logistique
python -m phase2_nlp_classique.embeddings_numpy                    # skip-gram from scratch
python -m phase2_nlp_classique.lstm_classifier                     # LSTM (PyTorch)
python -m phase2_nlp_classique.lstm_classifier --word-dropout 0.3  # variante : mots remplacés par <unk> à l'entraînement
python phase2_nlp_classique/spam_baseline.py                       # SMS Spam Collection UCI (Internet requis)

# Phase 3
python -m phase3_transformers.attention_numpy                      # attention multi-têtes
python -m phase3_transformers.finetune_transformer --epochs 3      # CamemBERT (GPU conseillé)

# Phase 4
python -m phase4_rag.evaluate --output reports/rag_metrics.json    # TF-IDF
python -m phase4_rag.evaluate_semantic                             # TF-IDF contre embeddings + FAISS
python -m phase4_rag.rag_pipeline "À quoi sert un SIEM ?" --semantic --show-scores   # --threshold 0.25 pour régler le refus

# Tableau comparatif (modèles + références triviales)
python scripts/compare_models.py
```

## Dépannage (Windows)

- **Chemin trop long pendant l'installation de PyTorch** : crée l'environnement à la racine (`python -m venv C:\venv`) et utilise `C:\venv\Scripts\python.exe` à la place de `python`.
- **Le modèle d'embeddings ne se charge pas** (erreur `Unrecognized processing class`) : désactive le téléchargement Xet avec `$env:HF_HUB_DISABLE_XET = "1"` avant de relancer.
- **Lancer un script du projet** : utilise `python -m dossier.script` (par exemple `python -m phase1_fondamentaux.xor_pytorch`) depuis la racine du projet.

## Données

Le jeu principal est **synthétique** : des messages français de smishing et de messages légitimes, générés à partir de gabarits avec du bruit (fautes, accents, majuscules). Les messages légitimes contiennent des mots piégeux (compte, lien, colis, virement). La séparation se fait **par gabarit** : le test (372 messages, 218 spams et 154 légitimes) contient 4 gabarits jamais vus à l'entraînement, ce qui mesure la généralisation et pas la mémorisation. L'entraînement utilise 6 gabarits de spam et 8 gabarits légitimes. Un second jeu public, SMS Spam Collection (UCI), est utilisable par `spam_baseline.py`.

## Résultats

### Classification de smishing (jeu français, gabarits de test inconnus, graine 0)

| Modèle | Accuracy | Précision (spam) | Rappel (spam) | F1 (spam) |
|---|---|---|---|---|
| TF-IDF + régression logistique | 0,968 | 0,990 | 0,954 | 0,972 |
| CamemBERT fine-tuné (3 époques, Colab sur CPU) | 0,758 | 1,000 | 0,587 | 0,740 |
| LSTM avec word dropout 0,3 | 0,659 | 0,633 | 0,991 | 0,773 |
| LSTM | 0,570 | 0,651 | 0,573 | 0,610 |
| Référence : toujours « spam » | 0,586 | 0,586 | 1,000 | 0,739 |
| Référence : toujours « légitime » | 0,414 | 0,000 | 0,000 | 0,000 |

Un premier entraînement de CamemBERT avec la même graine avait donné 0,753 d'accuracy et 0,733 de F1 : les résultats varient légèrement d'un entraînement à l'autre (matériel différent), et un seul tirage ne permet pas de distinguer de si petits écarts.

### Précision par gabarit (test)

| Gabarit | Message | TF-IDF | LSTM | CamemBERT |
|---|---|---|---|---|
| spam-6 | alerte de sécurité, « vérifiez votre mot de passe sur [domaine] » (avec lien) | 124/125 | 125/125 | 125/125 |
| spam-7 | « crédit gratuit, répondez OUI » (**sans lien**) | 84/93 | 0/93 | 3/93 |
| ham-8 | un document partagé sur le drive de la promo | 69/71 | 5/71 | 71/71 |
| ham-9 | rappel de réunion de projet | 83/83 | 82/83 | 83/83 |

### Autres briques

| Brique | Résultat |
|---|---|
| Réseau XOR NumPy (5 graines : 1, 7, 21, 42, 99) | 100 % d'accuracy à chaque graine, perte finale moyenne ≈ 0,0003 |
| Réseau XOR PyTorch (mêmes poids initiaux que NumPy) | mêmes pertes finales que la version NumPy pour les 5 graines (par exemple 0,000323 pour la graine 1), 100 % d'accuracy |
| Retrieval TF-IDF (14 questions, k = 3, seuil 0,10) | Recall@3 = 1,000, MRR = 1,000, 6/6 questions hors corpus refusées, 4/6 questions proches du domaine mais absentes refusées |
| Retrieval sémantique, MiniLM multilingue + FAISS (seuil 0,35) | Recall@3 = 1,000, MRR = 0,929, 6/6 questions hors corpus refusées, 3/6 questions proches du domaine mais absentes refusées |

### Lecture des résultats

**Un défaut de conception du jeu de données explique l'essentiel des échecs.** Dans les gabarits que j'ai écrits, tous les gabarits de spam contiennent un domaine suspect sauf un (spam-7, « crédit gratuit, répondez OUI »), et aucun gabarit légitime n'en contient. À l'entraînement, la présence d'un domaine sépare donc parfaitement les deux classes : c'est un raccourci que les modèles peuvent apprendre au lieu du sens du message. Le gabarit spam-7, le seul spam sans domaine, est justement réservé au test.

**Les modèles échouent presque exactement là.** Sur spam-7, CamemBERT trouve 3 messages sur 93, le LSTM 0 sur 93 et TF-IDF 84 sur 93. Pour CamemBERT, c'est la **seule** source d'erreurs : les trois autres gabarits sont classés sans aucune erreur (71/71, 83/83 et 125/125), donc ses 90 spams manqués sont presque tous ceux de spam-7. Ce résultat est cohérent avec l'hypothèse du raccourci, mais il n'a pas été confirmé par une expérience directe (par exemple ajouter un lien aux messages de spam-7, ou en retirer aux autres spams, et mesurer l'effet).

**TF-IDF est le meilleur modèle ici**, mais il n'est pas immunisé : il perd 9 messages sur 93 sur ce même gabarit.

**Les références triviales relativisent les scores.** L'accuracy du LSTM (0,570) est inférieure à celle de la référence « toujours spam » (0,586). Avec le word dropout, l'erreur se déplace au lieu de disparaître : 125 des 154 messages légitimes sont classés spam, ce qui donne un rappel de 0,991 mais une précision de 0,633. Le F1 de CamemBERT (0,740) est pratiquement égal à celui de « toujours spam » (0,739), même si son accuracy est meilleure : il ne se trompe jamais sur les messages légitimes, mais il rate presque tout un type de spam. Le LSTM se trompe en plus sur ham-8 (66 messages légitimes sur 71 classés spam).

**La validation ne détecte pas le problème.** Les trois modèles neuronaux atteignent 1,000 d'accuracy de validation dès les premières époques, parce que la validation partage ses gabarits avec l'entraînement. Seul le test par gabarits inconnus révèle l'écart.

**Ces résultats ne disent pas que les Transformers sont moins bons que TF-IDF en général.** Ils portent sur 3 époques, une seule graine et un petit jeu synthétique construit avec un raccourci. Un jeu sans ce raccourci (spams sans lien, messages légitimes avec lien) permettrait de comparer les modèles sur le sens des messages.

**Retrieval.** Le TF-IDF fait mieux que les embeddings sur ces 14 questions (MRR 1,000 contre 0,929) : les questions réutilisent des mots du corpus, ce qui avantage la recherche lexicale. Le retrieval sémantique n'apporte donc pas de gain ici, et un jeu de questions reformulées sans mots du corpus serait nécessaire pour mesurer son vrai avantage. Le seuil de refus ne sépare pas toutes les questions proches du domaine : avec les embeddings, le meilleur score d'une question absente du corpus atteint 0,506 alors que la plus faible question du corpus obtient 0,466, donc aucun seuil ne les sépare (3 refus sur 6). Avec TF-IDF, l'écart n'est que de 0,006 (0,209 contre 0,215) : une séparation parfaite existe mais elle est fragile et calibrée sur un très petit jeu.

## Limites

- Données synthétiques : les scores sont optimistes par rapport à de vrais messages, et le jeu contient un raccourci (présence d'un domaine) décrit ci-dessus.
- Un seul tirage (graine 0) : aucune barre d'erreur sur les comparaisons entre modèles, et deux entraînements de CamemBERT avec la même graine ont donné des résultats légèrement différents.
- La validation partage ses gabarits avec l'entraînement : elle n'est pas informative pour détecter le surapprentissage aux gabarits.
- Corpus RAG de 14 notes et 20 questions écrites à la main : les métriques de retrieval ne mesurent pas une performance générale.
- Le seuil de refus est calibré sur ce petit jeu de questions longues. Les questions proches du domaine mais absentes du corpus sont plus difficiles à refuser que les questions sans rapport, et une question courte du corpus peut être refusée à tort : avec les embeddings et le seuil 0,35, « À quoi sert un SIEM ? » est refusée alors que des questions plus longues sur le même sujet sont acceptées.
- La génération par LLM est facultative et dépend d'une clé API ; la génération extractive répond hors ligne.

## Feuille de route

- [x] Réseau XOR from scratch (NumPy) avec test de gradient
- [x] Baseline TF-IDF et jeu français avec gabarits de test inconnus
- [x] Attention multi-têtes from scratch avec tests
- [x] Skip-gram from scratch (NumPy)
- [x] Retrieval TF-IDF avec refus et évaluation annotée
- [x] XOR en PyTorch, comparaison avec la version NumPy
- [x] Classifieur LSTM, comparaison avec TF-IDF et références triviales
- [x] Fine-tuning de CamemBERT, comparaison avec TF-IDF et LSTM
- [x] Retrieval sémantique (embeddings + FAISS), comparaison avec TF-IDF
- [x] Analyse d'erreurs par gabarit (TF-IDF, LSTM, CamemBERT)
- [ ] Jeu de données v2 sans raccourci (spams sans lien, messages légitimes avec lien) et expérience d'ablation du raccourci
- [ ] Validation par gabarit, plusieurs graines et plus de gabarits d'entraînement
- [ ] Ajouter des questions courtes au jeu d'évaluation du retrieval et recalibrer le seuil
- [ ] Pipeline RAG avec génération LLM (clé API)

## Licence

MIT, voir `LICENSE`.
