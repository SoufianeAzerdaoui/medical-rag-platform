# LLM Writer Benchmark Summary

## Classement global par modèle

1. **llama3.2:latest** — score global Q1-Q4 : 8.5

## Résumé des scores et du temps

| Modèle | Score global Q1-Q4 | model_verified_rate (llm_expected) | llm_writer_rate (llm_expected) | llm_expected_success_rate | deterministic_preferred_success_rate | fallback_rate | hard_gate_rejection_rate (llm_expected) | repair_success_rate | accepted_llm_writer_count | deterministic_fallback_count | deterministic_preferred_count | deterministic_only_count | avg_llm_writer_ms | avg_response_time (s) | Fallbacks Q1-Q4 | Fail Q1-Q4 | Warning Q1-Q4 | Temps moyen (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:latest | 8.5 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1 | 3 | 3 | 1 | 6773.416 | 6.939 | 0 | 0 | 0 | 6.939 |

## Détails des fallbacks et warnings

### warnings_by_type
- downgraded_non_fact_error:unsupported_analyte: 2
- Some numeric values were not found in evidence.: 1
- missing_conclusion: 1
- downgraded_non_fact_error:unsupported_value: 1

## Erreurs fréquentes


### Avertissements
- downgraded_non_fact_error:unsupported_analyte: 2
- Some numeric values were not found in evidence.: 1
- missing_conclusion: 1
- downgraded_non_fact_error:unsupported_value: 1

## Détails par modèle / question

| Modèle | Question | Score | Route | Strategy | LLM expected | Writer | Validation | Qualité | Fallback | Temps | Answer preview |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:latest | Q1 | 8 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | warning |  | 0.114 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| llama3.2:latest | Q2 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.118 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| llama3.2:latest | Q3 | 9 | doc_scoped_priority_anomalies | deterministic_preferred | False | professional_fallback | pass | warning |  | 0.183 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| llama3.2:latest | Q4 | 8 | doc_scoped_medical_interpretation_guarded | llm_writer_expected | True | llm_writer | pass | pass |  | 27.34 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  - TSHus : 55,00 mUI/L (au-dessus de la r |
| llama3.2:latest | Q5 | 9 | doc_scoped_toxicology_summary | deterministic_only | False | deterministic_toxicology_renderer | pass | pass |  | 0.15 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |

## Check non-régression Q5

Q5 reste un contrôle non-régression déterministe. Le score LLM writer attendu se concentre sur les routes llm_expected.
