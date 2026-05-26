# LLM Writer Benchmark Summary

## Classement global par modèle

1. **gemma3:4b** — score global Q1-Q4 : 8.75
2. **llama3.2:latest** — score global Q1-Q4 : 8.75
3. **mistral:7b-instruct-q4_0** — score global Q1-Q4 : 8.75
4. **qwen2.5:7b-instruct** — score global Q1-Q4 : 8.75

## Résumé des scores et du temps

| Modèle | Score global Q1-Q4 | model_verified_rate (llm_expected) | llm_writer_rate (llm_expected) | llm_expected_success_rate | deterministic_preferred_success_rate | fallback_rate | hard_gate_rejection_rate (llm_expected) | repair_success_rate | accepted_llm_writer_count | deterministic_fallback_count | deterministic_preferred_count | deterministic_only_count | avg_llm_writer_ms | avg_response_time (s) | Fallbacks Q1-Q4 | Fail Q1-Q4 | Warning Q1-Q4 | Temps moyen (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3:4b | 8.75 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 3 | 0 | 0 | 0.0 | 4.011 | 0 | 0 | 0 | 4.011 |
| llama3.2:latest | 8.75 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 3 | 0 | 0 | 0.0 | 5.585 | 0 | 0 | 0 | 5.585 |
| mistral:7b-instruct-q4_0 | 8.75 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 3 | 0 | 0 | 0.0 | 4.008 | 0 | 0 | 0 | 4.008 |
| qwen2.5:7b-instruct | 8.75 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 | 3 | 0 | 0 | 0.0 | 3.879 | 0 | 0 | 0 | 3.879 |

## Détails des fallbacks et warnings

### warnings_by_type
- Some numeric values were not found in evidence.: 4
- missing_conclusion: 4
- downgraded_non_fact_error:unsupported_value: 4

## Erreurs fréquentes


### Avertissements
- Some numeric values were not found in evidence.: 4
- missing_conclusion: 4
- downgraded_non_fact_error:unsupported_value: 4

## Détails par modèle / question

| Modèle | Question | Score | Route | Strategy | LLM expected | Writer | Validation | Qualité | Fallback | Temps | Answer preview |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama3.2:latest | Q1 | 9 | doc_scoped_biological_summary |  | False | professional_fallback | pass | pass |  | 0.138 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| llama3.2:latest | Q2 | 9 | doc_scoped_biological_summary |  | False | professional_fallback | pass | pass |  | 0.144 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| llama3.2:latest | Q3 | 9 |  |  | False | professional_fallback | pass | warning |  | 0.201 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| llama3.2:latest | Q4 | 8 |  |  | False | llm_writer | pass | pass |  | 21.856 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  - TSHus : 55,00 mUI/L (au-dessus de la r |
| llama3.2:latest | Q5 | 9 | doc_scoped_toxicology_summary |  | False | deterministic_toxicology_renderer | pass | pass |  | 0.169 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| qwen2.5:7b-instruct | Q1 | 9 | doc_scoped_biological_summary |  | False | professional_fallback | pass | pass |  | 0.154 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| qwen2.5:7b-instruct | Q2 | 9 | doc_scoped_biological_summary |  | False | professional_fallback | pass | pass |  | 0.163 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| qwen2.5:7b-instruct | Q3 | 9 |  |  | False | professional_fallback | pass | warning |  | 0.221 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| qwen2.5:7b-instruct | Q4 | 8 |  |  | False | llm_writer | pass | pass |  | 14.976 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  - TSHus : 55,00 mUI/L (au-dessus de la r |
| qwen2.5:7b-instruct | Q5 | 9 | doc_scoped_toxicology_summary |  | False | deterministic_toxicology_renderer | pass | pass |  | 0.159 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| mistral:7b-instruct-q4_0 | Q1 | 9 | doc_scoped_biological_summary |  | False | professional_fallback | pass | pass |  | 0.211 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| mistral:7b-instruct-q4_0 | Q2 | 9 | doc_scoped_biological_summary |  | False | professional_fallback | pass | pass |  | 0.187 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| mistral:7b-instruct-q4_0 | Q3 | 9 |  |  | False | professional_fallback | pass | warning |  | 0.22 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| mistral:7b-instruct-q4_0 | Q4 | 8 |  |  | False | llm_writer | pass | pass |  | 15.415 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  - TSHus : 55,00 mUI/L (au-dessus de la r |
| mistral:7b-instruct-q4_0 | Q5 | 9 | doc_scoped_toxicology_summary |  | False | deterministic_toxicology_renderer | pass | pass |  | 0.169 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| gemma3:4b | Q1 | 9 | doc_scoped_biological_summary |  | False | professional_fallback | pass | pass |  | 0.151 | Anormaux : Bilirubine Directe (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus), APOLIPOPROTÉINE A1 (APO A1) (au-de |
| gemma3:4b | Q2 | 9 | doc_scoped_biological_summary |  | False | professional_fallback | pass | pass |  | 0.155 | Anormaux : Réserve Alcaline (en dessous). Résultats dans la référence uniquement : Phosphore, LDH. Conclusion technique  |
| gemma3:4b | Q3 | 9 |  |  | False | professional_fallback | pass | warning |  | 0.214 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| gemma3:4b | Q4 | 8 |  |  | False | llm_writer | pass | pass |  | 15.523 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  - TSHus : 55,00 mUI/L (au-dessus de la r |
| gemma3:4b | Q5 | 9 | doc_scoped_toxicology_summary |  | False | deterministic_toxicology_renderer | pass | pass |  | 0.159 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |

## Check non-régression Q5

Q5 reste un contrôle non-régression déterministe. Le score LLM writer attendu se concentre sur les routes llm_expected.
