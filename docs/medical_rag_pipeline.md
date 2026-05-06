# Architecture du Pipeline RAG Médical Strict

Ce document décrit l'implémentation du pipeline de recherche et génération augmentée (RAG) optimisé pour les rapports cliniques sur matériel local (MacBook Pro Intel).

## 1. Vue d'Ensemble du Pipeline

Le pipeline suit un flux strict en 9 étapes pour garantir la fidélité clinique et la traçabilité des sources.

1.  **User Question** : Saisie interactive ou CLI.
2.  **Query Normalization** : Nettoyage et préparation de la requête.
3.  **Clinical-aware Retrieval** : Recherche hybride (E5-base + BM25) avec pondération clinique.
4.  **Context Builder** : Expansion sémantique (résumés, statuts de validation).
5.  **Evidence Pack Builder** : Structuration des preuves avec scores de confiance.
6.  **Prompt Builder** : Injection dans un système de règles strictes.
7.  **LLM Client** : Appel à Llama 3.2 3B via Ollama (Temp: 0.1).
8.  **Answer Validator** : Détection d'hallucinations numériques et contrôle des citations.
9.  **Final Answer** : Réponse validée avec sources exhaustives.

## 2. Détail Technique des Composants

### Moteur de Recherche (SearchEngine)
Situé dans `scripts/retrieval/search.py`, il coordonne Qdrant (vectoriel) et SQLite (mots-clés).

### Orchestrateur (MedicalRagPipeline)
Situé dans `scripts/retrieval/pipeline.py`, il est le chef d'orchestre qui applique les 9 étapes. Il contient notamment le **Validator** qui scanne chaque nombre généré par l'IA pour vérifier s'il existe dans les documents sources.

### Génération (AnswerGenerator)
Situé dans `scripts/retrieval/generation.py`, il gère le **System Prompt** médical. Ce prompt interdit à l'IA d'inventer des informations ou de répondre à des questions vagues sans preuve.

## 3. Configuration Matérielle et Modèles

*   **Embedding** : `intfloat/multilingual-e5-base` (768 dimensions).
*   **LLM** : `llama3.2:latest` (3.2B parameters).
*   **Format** : GGUF (Quantification 4-bit) optimisé pour l'inférence CPU via Ollama.
*   **Fenêtre de Contexte** : 128k tokens supportés (utilisés ici pour ~4k tokens par requête afin de maintenir une vitesse d'inférence élevée).
*   **Rôle du LLM** : Synthèse sémantique, raisonnement clinique prudent et insertion de citations structurées.
*   **Infrastructure** : CPU Local (Intel), pas de dépendance cloud pour la confidentialité des données.

## 4. Utilisation

Pour lancer le pipeline en mode interactif :
```bash
python3 scripts/retrieval/rag_cli.py
```

Pour obtenir une sortie JSON exploitable par une autre application :
```bash
python3 scripts/retrieval/rag_cli.py "Ma question" --json
```
