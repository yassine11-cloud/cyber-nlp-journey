# Questions d'entretien à préparer

Réponds à voix haute, en deux minutes maximum, avec tes propres chiffres.

## Phase 1, réseau de neurones
1. Pourquoi un neurone seul ne peut-il pas apprendre XOR ?
2. Que se passe-t-il avec un taux d'apprentissage de 10 ? Et de 0,001 ? (essaie dans le code)
3. Pourquoi initialise-t-on les poids au hasard plutôt qu'à zéro ?
4. À quoi sert la dérivée de la sigmoïde dans la rétropropagation ?
5. Comment le test de gradient numérique prouve-t-il que ta rétropropagation est correcte ?

## Phase 2, NLP classique
1. Qu'est-ce que TF-IDF et pourquoi est-ce mieux qu'un simple comptage de mots ?
2. Pourquoi a-t-on utilisé `class_weight="balanced"` ?
3. Pourquoi regarde-t-on le rappel et la précision de la classe « spam » et pas seulement l'accuracy ?
4. Que changent les bigrammes sur tes résultats ? Donne tes chiffres.
5. Quelles erreurs fait le modèle (faux positifs, faux négatifs) et quel est le coût de chacune en sécurité ?

## Phase 4, retrieval et refus
1. Pourquoi la Precision@3 vaut-elle 0,333 alors que tout fonctionne ?
2. Comment as-tu choisi le seuil de refus ? Que se passe-t-il s'il est trop haut ou trop bas ?
3. Pourquoi les métriques de ce projet sont-elles optimistes ? Quelle serait une meilleure évaluation ?
4. Quelle est la différence entre un retrieval lexical (TF-IDF) et un retrieval sémantique (embeddings) ?
5. Que faudrait-il ajouter pour en faire un vrai RAG ?

## Questions transversales
1. Qu'as-tu fait toi-même, et qu'as-tu fait avec l'aide de Claude ?
2. Quelle a été ta principale erreur et comment l'as-tu corrigée ?
3. Comment ce projet se relie-t-il à la cybersécurité et au GRC ?

## Phase 2 (suite) : embeddings et LSTM
1. Que signifie « deux mots proches dans l'espace des embeddings » ? Donne un exemple de tes voisins les plus proches et dis s'ils ont du sens.
2. Pourquoi l'échantillonnage négatif ? Que ferait un softmax complet sur tout le vocabulaire ?
3. Pourquoi un LSTM plutôt qu'un RNN simple ? Que résout la porte d'oubli ?
4. Pourquoi emballer les séquences (`pack_padded_sequence`) ? Que se passerait-il sans cela ?
5. Ton LSTM fait-il mieux ou moins bien que TF-IDF sur les gabarits jamais vus ? Pourquoi, d'après toi ?

## Phase 3 : attention et Transformers
1. Que calcule `softmax(QKᵀ / √d_k)` et pourquoi divise-t-on par √d_k ?
2. À quoi sert le masque causal ? Montre-le sur la matrice de poids.
3. Pourquoi plusieurs têtes plutôt qu'une seule ?
4. L'attention ne connaît pas l'ordre des mots : comment le Transformer le retrouve-t-il ? (test de permutation)
5. Qu'est-ce qui change entre entraîner un modèle from scratch et le fine-tuner ? Pourquoi un petit taux d'apprentissage (2e-5) ?
6. CamemBERT contre TF-IDF contre LSTM : quel est le meilleur sur tes données et à quel coût ?

## Phase 4 (suite) : retrieval sémantique et RAG
1. Pourquoi normaliser les vecteurs avant d'utiliser le produit scalaire dans FAISS ?
2. Dans quel cas le retrieval sémantique bat-il TF-IDF, et dans quel cas TF-IDF suffit-il ?
3. Comment as-tu calibré le seuil de refus avec les scores `in_corpus_min`, `out_of_corpus_max` et `near_out_of_corpus_max` ?
4. Que contient ton prompt et pourquoi demande-t-on de citer les passages ?
5. Quels risques restent dans un RAG (hallucination malgré le contexte, injection de prompt dans un document, corpus obsolète) et comment les limiter ?
