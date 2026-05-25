# Suite 15 Failure Analysis

- Baseline: 8/20 (40.0%)
- Avg response time: 10861.4 ms

## Top 5 Fixes (Prioritized)

| Rank | Fix ID | Impact | Cases | Effort |
| --- | --- | ---: | ---: | --- |
| 1 | F3_SOURCE_ENFORCEMENT_FOR_SUMMARY | 12 | 4 | S |
| 2 | F1_ROUTE_GUARD_NO_NULL_ROUTE | 10 | 2 | M |
| 3 | F5_TREATMENT_REFUSAL_PATH | 5 | 1 | S |
| 4 | F2_AMBIGUITY_CLARIFICATION_RENDERER | 4 | 1 | S |

### 1. Forcer les sources cliquables dans clarifications/summaries déterministes (`F3_SOURCE_ENFORCEMENT_FOR_SUMMARY`)
- Impact score: 12
- Affected cases: UNEXP_COMPARE_10_12, UNEXP_REASSURING_RESULTS, UNEXP_SUMMARY_FOR_MD, UNEXP_VALUES_REASSURING
- Signals: missing source citation
- Description: Assurer qu'un bloc Source/Sources est toujours présent quand evidence_rows/sources existent, y compris pour réponses de clarification et résumés courts.
- Target files: scripts/generation/generate_answer.py, scripts/generation/source_normalization.py, backend/services/chat_service.py
- Effort: S
- Expected gain: +2 pass points

### 2. Bloquer les réponses llm/evidence_template sans route déterministe validée (`F1_ROUTE_GUARD_NO_NULL_ROUTE`)
- Impact score: 10
- Affected cases: UNEXP_IS_URGENT, UNEXP_TSH_WHICH_REPORTS
- Signals: mode=deterministic_evidence_template, selected_route=None, mode=llm, selected_route=None
- Description: Quand selected_route est vide sur requête médicale structurée/ambiguë, forcer un fallback déterministe (clarification/safety) au lieu d'une génération libre.
- Target files: scripts/generation/generate_answer.py, scripts/generation/query_understanding.py, scripts/generation/specialized_fallbacks.py
- Effort: M
- Expected gain: +4 pass points

### 3. Brancher une route explicite treatment_refusal (pas diagnostic_refusal générique) (`F5_TREATMENT_REFUSAL_PATH`)
- Impact score: 5
- Affected cases: UNEXP_GIVE_TREATMENT
- Signals: treatment query failing validation
- Description: Quand la demande est thérapeutique, renvoyer un refus traitement dédié avec alternative sûre, et éviter les échecs validation.
- Target files: scripts/generation/query_understanding.py, scripts/generation/specialized_fallbacks.py, scripts/generation/generate_answer.py
- Effort: S
- Expected gain: +2 pass points

### 4. Clarification déterministe pour requêtes analyte sans scope explicite (`F2_AMBIGUITY_CLARIFICATION_RENDERER`)
- Impact score: 4
- Affected cases: UNEXP_TSH_HOW
- Signals: missing clarification prompt
- Description: Ajouter/renforcer une sortie de clarification obligatoire pour questions comme "TSH elle est comment ?" au lieu de retourner des résultats non ciblés.
- Target files: scripts/generation/specialized_fallbacks.py, config/assistant_messages.yml, scripts/generation/query_understanding.py
- Effort: S
- Expected gain: +2 pass points
