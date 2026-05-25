# Suite 15 Failure Analysis

- Baseline: 10/20 (50.0%)
- Avg response time: 16632.7 ms

## Top 5 Fixes (Prioritized)

| Rank | Fix ID | Impact | Cases | Effort |
| --- | --- | ---: | ---: | --- |
| 1 | F3_SOURCE_ENFORCEMENT_FOR_SUMMARY | 18 | 6 | S |
| 2 | F1_ROUTE_GUARD_NO_NULL_ROUTE | 10 | 2 | M |

### 1. Forcer les sources cliquables dans clarifications/summaries déterministes (`F3_SOURCE_ENFORCEMENT_FOR_SUMMARY`)
- Impact score: 18
- Affected cases: UNEXP_REASSURING_RESULTS, UNEXP_RENAL_NORMAL, UNEXP_SUMMARY_FOR_MD, UNEXP_THYROID_COHERENT, UNEXP_TOX_POSITIVE, UNEXP_VALUES_REASSURING
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
