# Reference Range Lookup — Behavior Changelog

## Feature flag
- `REFERENCE_RANGE_STRICT_MODE=true`:
  Active le flow strict déterministe pour les plages physiologiques.
- `REFERENCE_RANGE_STRICT_MODE=false`:
  Désactive le flow strict sans redeploy et utilise un fallback contrôlé.

## Status possibles

### `selected`
Définition :
Une plage physiologique unique et fiable a été sélectionnée.

Exemple :
AMH femme 30–34 ans -> 2,34–3,55 ng/ml

Comportement :
Réponse directe, courte, sourcée.

### `ambiguous`
Définition :
Plusieurs plages compatibles ou plusieurs documents avec plages différentes.

Comportement :
Ne pas choisir arbitrairement.
Demander précision : date, document, profil.
Afficher éventuellement 2–3 options.

### `fallback`
Définition :
Aucune plage spécifique au profil demandé n’est trouvée, mais une plage générique existe.

Exemple :
User demande “ACTH femme”, mais la référence ACTH disponible est générale.

Comportement :
Répondre avec la plage générale seulement si explicitement signalé :
“Aucune plage spécifique femme n’est indiquée ; la plage disponible est générale.”

### `no_match`
Définition :
Aucune référence exploitable n’a été trouvée après lookup analyte/doc/type.

Comportement :
Réponse claire, pas de valeur inventée, pas de tableau global.

## Interdictions
- Ne jamais laisser le LLM choisir la plage.
- Ne jamais retourner un tableau global multi-analytes si `request_all_ranges=false`.
- Ne jamais faire un fallback silencieux.
- Ne jamais inventer une source.
