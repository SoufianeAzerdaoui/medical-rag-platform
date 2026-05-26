# LLM Writer Benchmark Summary

## Classement global par modèle

**Benchmark non concluant pour le choix du modèle.** Au moins un modèle n'a pas été vérifié sur Q1-Q4 (routes llm_expected).

1. **gemma3:4b** — score global Q1-Q4 : 8.5
2. **llama3.2:latest** — score global Q1-Q4 : 8.5
3. **mistral:7b-instruct-q4_0** — score global Q1-Q4 : 8.5
4. **qwen2.5:7b-instruct** — score global Q1-Q4 : 8.5

## Résumé des scores et du temps

| Modèle | Score global Q1-Q4 | model_verified_rate (llm_expected) | llm_writer_rate (llm_expected) | llm_expected_success_rate | deterministic_preferred_success_rate | fallback_rate | hard_gate_rejection_rate (llm_expected) | repair_success_rate | accepted_llm_writer_count | deterministic_fallback_count | deterministic_preferred_count | deterministic_only_count | avg_llm_writer_ms | avg_response_time (s) | Fallbacks Q1-Q4 | Fail Q1-Q4 | Warning Q1-Q4 | Temps moyen (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3:4b | 8.5 | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0 | 4 | 3 | 1 | 0.0 | 0.163 | 0 | 0 | 0 | 0.163 |
| llama3.2:latest | 8.5 | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0 | 4 | 3 | 1 | 0.0 | 0.175 | 0 | 0 | 0 | 0.175 |
| mistral:7b-instruct-q4_0 | 8.5 | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0 | 4 | 3 | 1 | 0.0 | 0.165 | 0 | 0 | 0 | 0.165 |
| qwen2.5:7b-instruct | 8.5 | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | 0 | 4 | 3 | 1 | 0.0 | 0.167 | 0 | 0 | 0 | 0.167 |

## Détails des fallbacks et warnings

### warnings_by_type
- missing_conclusion: 8
- Some numeric values were not found in evidence.: 4
- downgraded_non_fact_error:unsupported_value: 4
- downgraded_non_fact_error:analyte_overmatch: 4
- downgraded_non_fact_error:false_missing_item: 4

## Erreurs fréquentes


### Avertissements
- missing_conclusion: 8
- Some numeric values were not found in evidence.: 4
- downgraded_non_fact_error:unsupported_value: 4
- downgraded_non_fact_error:analyte_overmatch: 4
- downgraded_non_fact_error:false_missing_item: 4

## Détails par modèle / question

| Modèle | Question | Score | Route | Strategy | LLM expected | Writer | Validation | Qualité | Fallback | Temps | Answer preview |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:latest | Q1 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.144 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| llama3.2:latest | Q2 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.161 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| llama3.2:latest | Q3 | 9 | doc_scoped_priority_anomalies | deterministic_preferred | False | professional_fallback | pass | warning |  | 0.223 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| llama3.2:latest | Q4 | 7 | doc_scoped_medical_interpretation_guarded | llm_writer_expected | True | professional_fallback | pass | warning |  | 0.173 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  Non, on ne peut pas conclure à un diagno |
| llama3.2:latest | Q5 | 9 | doc_scoped_toxicology_summary | deterministic_only | False | deterministic_toxicology_renderer | pass | pass |  | 0.143 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| qwen2.5:7b-instruct | Q1 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.136 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| qwen2.5:7b-instruct | Q2 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.152 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| qwen2.5:7b-instruct | Q3 | 9 | doc_scoped_priority_anomalies | deterministic_preferred | False | professional_fallback | pass | warning |  | 0.201 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| qwen2.5:7b-instruct | Q4 | 7 | doc_scoped_medical_interpretation_guarded | llm_writer_expected | True | professional_fallback | pass | warning |  | 0.18 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  Non, on ne peut pas conclure à un diagno |
| qwen2.5:7b-instruct | Q5 | 9 | doc_scoped_toxicology_summary | deterministic_only | False | deterministic_toxicology_renderer | pass | pass |  | 0.145 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| mistral:7b-instruct-q4_0 | Q1 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.146 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| mistral:7b-instruct-q4_0 | Q2 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.148 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| mistral:7b-instruct-q4_0 | Q3 | 9 | doc_scoped_priority_anomalies | deterministic_preferred | False | professional_fallback | pass | warning |  | 0.199 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| mistral:7b-instruct-q4_0 | Q4 | 7 | doc_scoped_medical_interpretation_guarded | llm_writer_expected | True | professional_fallback | pass | warning |  | 0.166 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  Non, on ne peut pas conclure à un diagno |
| mistral:7b-instruct-q4_0 | Q5 | 9 | doc_scoped_toxicology_summary | deterministic_only | False | deterministic_toxicology_renderer | pass | pass |  | 0.148 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| gemma3:4b | Q1 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.139 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| gemma3:4b | Q2 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.143 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| gemma3:4b | Q3 | 9 | doc_scoped_priority_anomalies | deterministic_preferred | False | professional_fallback | pass | warning |  | 0.195 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| gemma3:4b | Q4 | 7 | doc_scoped_medical_interpretation_guarded | llm_writer_expected | True | professional_fallback | pass | warning |  | 0.173 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  Non, on ne peut pas conclure à un diagno |
| gemma3:4b | Q5 | 9 | doc_scoped_toxicology_summary | deterministic_only | False | deterministic_toxicology_renderer | pass | pass |  | 0.145 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |

## Check non-régression Q5

Q5 reste un contrôle non-régression déterministe. Le score LLM writer attendu se concentre sur les routes llm_expected.
