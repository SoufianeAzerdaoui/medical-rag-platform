# LLM Writer Benchmark Summary

## Classement global par modèle

1. **gemma3:4b** — score global Q1-Q4 : 9.0
2. **llama3.2:latest** — score global Q1-Q4 : 9.0
3. **mistral:7b-instruct-q4_0** — score global Q1-Q4 : 8.0
4. **qwen2.5:7b-instruct** — score global Q1-Q4 : 8.0

## Résumé des scores et du temps

| Modèle | Score global Q1-Q4 | model_verified_rate (llm_expected) | llm_writer_rate (llm_expected) | llm_expected_success_rate | deterministic_preferred_success_rate | fallback_rate | hard_gate_rejection_rate (llm_expected) | repair_success_rate | accepted_llm_writer_count | deterministic_fallback_count | deterministic_preferred_count | deterministic_only_count | avg_llm_writer_ms | avg_response_time (s) | Fallbacks Q1-Q4 | Fail Q1-Q4 | Warning Q1-Q4 | Temps moyen (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3:4b | 9.0 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1 | 3 | 3 | 1 | 16440.466 | 16.89 | 0 | 0 | 0 | 16.89 |
| llama3.2:latest | 9.0 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1 | 3 | 3 | 1 | 14564.962 | 14.764 | 0 | 0 | 0 | 14.764 |
| mistral:7b-instruct-q4_0 | 8.0 | 100.0% | 0.0% | 0.0% | 100.0% | 25.0% | 0.0% | 0.0% | 0 | 4 | 3 | 1 | 22522.447 | 22.735 | 1 | 0 | 0 | 22.735 |
| qwen2.5:7b-instruct | 8.0 | 100.0% | 0.0% | 0.0% | 100.0% | 25.0% | 0.0% | 0.0% | 0 | 4 | 3 | 1 | 22522.734 | 22.721 | 1 | 0 | 0 | 22.721 |

## Détails des fallbacks et warnings

### fallback_by_reason
- llm_timeout: 2

### warnings_by_type
- missing_conclusion: 6
- Some numeric values were not found in evidence.: 4
- downgraded_non_fact_error:unsupported_value: 4
- downgraded_non_fact_error:analyte_overmatch: 2
- downgraded_non_fact_error:false_missing_item: 2

## Erreurs fréquentes


### Avertissements
- missing_conclusion: 6
- Some numeric values were not found in evidence.: 4
- downgraded_non_fact_error:unsupported_value: 4
- downgraded_non_fact_error:analyte_overmatch: 2
- downgraded_non_fact_error:false_missing_item: 2

## Détails par modèle / question

| Modèle | Question | Score | Route | Strategy | LLM expected | Writer | Validation | Qualité | Fallback | Temps | Answer preview |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:latest | Q1 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.163 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| llama3.2:latest | Q2 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.141 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| llama3.2:latest | Q3 | 9 | doc_scoped_priority_anomalies | deterministic_preferred | False | professional_fallback | pass | warning |  | 0.2 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| llama3.2:latest | Q4 | 9 | doc_scoped_medical_interpretation_guarded | llm_writer_expected | True | llm_writer | pass | pass |  | 58.553 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  - TSHus : 55,00 mUI/L ; référence 0,35 - |
| llama3.2:latest | Q5 | 9 | doc_scoped_toxicology_summary | deterministic_only | False | deterministic_toxicology_renderer | pass | pass |  | 0.164 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| qwen2.5:7b-instruct | Q1 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.149 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| qwen2.5:7b-instruct | Q2 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.157 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| qwen2.5:7b-instruct | Q3 | 9 | doc_scoped_priority_anomalies | deterministic_preferred | False | professional_fallback | pass | warning |  | 0.213 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| qwen2.5:7b-instruct | Q4 | 5 | doc_scoped_medical_interpretation_guarded | llm_writer_expected | True | professional_fallback | pass | warning | llm_timeout | 90.364 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  Non, on ne peut pas conclure à un diagno |
| qwen2.5:7b-instruct | Q5 | 9 | doc_scoped_toxicology_summary | deterministic_only | False | deterministic_toxicology_renderer | pass | pass |  | 0.188 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| mistral:7b-instruct-q4_0 | Q1 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.145 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| mistral:7b-instruct-q4_0 | Q2 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.154 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| mistral:7b-instruct-q4_0 | Q3 | 9 | doc_scoped_priority_anomalies | deterministic_preferred | False | professional_fallback | pass | warning |  | 0.212 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| mistral:7b-instruct-q4_0 | Q4 | 5 | doc_scoped_medical_interpretation_guarded | llm_writer_expected | True | professional_fallback | pass | warning | llm_timeout | 90.429 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  Non, on ne peut pas conclure à un diagno |
| mistral:7b-instruct-q4_0 | Q5 | 9 | doc_scoped_toxicology_summary | deterministic_only | False | deterministic_toxicology_renderer | pass | pass |  | 0.404 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| gemma3:4b | Q1 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.385 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| gemma3:4b | Q2 | 9 | doc_scoped_biological_summary | deterministic_preferred | False | professional_fallback | pass | pass |  | 0.403 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| gemma3:4b | Q3 | 9 | doc_scoped_priority_anomalies | deterministic_preferred | False | professional_fallback | pass | warning |  | 0.544 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| gemma3:4b | Q4 | 9 | doc_scoped_medical_interpretation_guarded | llm_writer_expected | True | llm_writer | pass | pass |  | 66.227 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  **Faits techniques observés :** Le profi |
| gemma3:4b | Q5 | 9 | doc_scoped_toxicology_summary | deterministic_only | False | deterministic_toxicology_renderer | pass | pass |  | 0.172 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |

## Check non-régression Q5

Q5 reste un contrôle non-régression déterministe. Le score LLM writer attendu se concentre sur les routes llm_expected.
