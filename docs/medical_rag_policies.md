# Medical RAG Policies and Configuration

Ce projet externalise désormais plusieurs règles métier dans `config/` :

- `medical_topics.yml` : topics médicaux (triggers, analytes, exclusions).
- `analyte_families.yml` : familles biologiques techniques et poids associés.
- `priority_scoring.yml` : coefficients/seuils de scoring de priorité.
- `assistant_messages.yml` : messages de conversation générale.

## Points importants

- Les topics médicaux sont configurables sans modifier le code Python.
- Les familles biologiques sont configurables sans modifier le code Python.
- Les messages conversationnels sont configurables.
- En cas de config absente ou invalide, des defaults sûrs sont appliqués automatiquement.

## Priorité technique

Le scoring de priorité sert uniquement à organiser l’affichage technique des anomalies.

**Les scores de priorité sont utilisés uniquement pour organiser l’affichage technique des anomalies. Ils ne constituent pas une évaluation clinique de gravité ni une recommandation médicale.**

## Sécurité

Les hard-gates (validation de sécurité/factualité) restent implémentés dans le code et ne sont pas externalisés, car ce sont des règles de sécurité non négociables.
