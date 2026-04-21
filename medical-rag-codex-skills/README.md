# medical-rag-codex-skills

Dépôt de démarrage pour un projet de fin d'études intitulé : **Plateforme intelligente RAG pour l’exploitation de rapports d’analyses médicales (PDF multimodaux)**.

Ce dépôt ne contient pas de backend, pas d'API, pas d'interface, et pas de code factice. Il formalise une base de travail propre, orientée **skills**, pour préparer une future chaîne RAG multimodale et hybride appliquée à des comptes rendus d'analyses médicales au format PDF.

## Contexte

Les laboratoires et centres d'analyses produisent fréquemment des rapports PDF hétérogènes : texte brut, tableaux, zones scannées, graphiques, logos, en-têtes techniques et parfois résultats issus d'automates. Ces documents sont utiles pour la consultation clinique, mais difficiles à exploiter automatiquement dans un système de recherche et de question-réponse fiable.

L'objectif de ce projet est de préparer une architecture documentaire et méthodologique pour transformer ces PDF en informations exploitables, traçables et réutilisables dans un pipeline RAG.

## Objectifs

- Extraire le texte, les tableaux et les zones d'images ou graphiques depuis des rapports médicaux PDF.
- Structurer les résultats de laboratoire dans un format normalisé.
- Préparer une recherche hybride combinant recherche vectorielle et recherche par mots-clés.
- Produire des réponses en langage naturel avec citations de sources.
- Évaluer la pertinence de la recherche, la fidélité des réponses et la résistance aux hallucinations.

## Problématique

Un rapport d'analyses médicales n'est pas un simple document textuel. La valeur informationnelle peut être répartie entre un tableau, une ligne de référence, une annotation, une image ou un graphique. Un pipeline uniquement textuel risque de perdre du contexte, d'ignorer des indices utiles et de produire des réponses peu robustes.

Une approche **multimodale** est donc pertinente pour préserver la structure réelle du document. Une approche **RAG hybride** est également nécessaire :

- la recherche vectorielle aide à retrouver des passages sémantiquement proches d'une question ;
- la recherche par mots-clés reste très utile pour les noms d'analyses, unités, valeurs exactes, sigles et formulations cliniques stables.

## Architecture cible

Chaîne cible du projet :

`Medical PDF Reports -> Ingestion Layer -> Multimodal Extraction -> Clinical Structuring -> Chunking -> Anonymization -> Indexing Layer -> keyword store + vector store + object store + metadata store -> Query Processing Service -> Query Embedding -> Query Routing -> Vector Search + Keyword Search -> Result Fusion -> Reranker -> Context Augmentation -> LLM Generation -> Output Guardrails -> Final Answer`

Cette architecture est détaillée dans [docs/architecture.md](/home/onizuka/Bureau/PFE/medical-rag-platform/medical-rag-codex-skills/docs/architecture.md).

## Skills du dépôt

Le dépôt est organisé autour de quatre skills complémentaires :

- `pdf-lab-intake` : guider l'ingestion d'un PDF médical, l'extraction multimodale, la détection de tableaux, les références de page et l'usage raisonné de l'OCR.
- `lab-record-structurer` : convertir les résultats extraits en enregistrements normalisés et traçables.
- `hybrid-rag-answer` : cadrer la recherche hybride, la fusion de résultats, le reranking, la citation et l'abstention.
- `rag-evaluator` : évaluer la qualité de la recherche, l'ancrage des réponses et la robustesse face aux réponses non fondées.

## Version 1 scope

Le périmètre V1 de ce dépôt est volontairement limité :

- documentation d'architecture ;
- conventions de travail pour futures sessions Codex ;
- skills réutilisables ;
- schémas et règles de référence ;
- structure prête à accueillir plus tard des exemples de données et des évaluations.

Ce dépôt n'a pas pour rôle d'implémenter dès maintenant un service applicatif complet.

## Version 2 ideas

- Ajouter un dossier d'expérimentation pour comparer plusieurs extracteurs PDF.
- Introduire un jeu de requêtes d'évaluation annotées manuellement.
- Définir un format commun de chunks et de citations.
- Ajouter des guides d'anonymisation et de qualité des données.
- Préparer ensuite un prototype minimal de pipeline hors production, seulement si le cadrage documentaire est stabilisé.

## Arborescence

```text
medical-rag-codex-skills/
├── README.md
├── AGENTS.md
├── .gitignore
├── docs/
│   ├── architecture.md
│   └── roadmap.md
├── skills/
│   ├── pdf-lab-intake/
│   │   ├── SKILL.md
│   │   ├── agents/openai.yaml
│   │   └── references/extraction-rules.md
│   ├── lab-record-structurer/
│   │   ├── SKILL.md
│   │   ├── agents/openai.yaml
│   │   └── references/lab-schema.md
│   ├── hybrid-rag-answer/
│   │   ├── SKILL.md
│   │   ├── agents/openai.yaml
│   │   └── references/retrieval-policy.md
│   └── rag-evaluator/
│       ├── SKILL.md
│       ├── agents/openai.yaml
│       └── references/eval-metrics.md
└── sample-data/
    └── README.md
```

## Orientation du dépôt

Ce dépôt est conçu pour être lisible par un encadrant, exploitable par un étudiant en PFE, et suffisamment structuré pour servir de base à des sessions futures avec Codex ou d'autres assistants de développement orientés documentation et architecture.
