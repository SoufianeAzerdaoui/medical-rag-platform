# LLM Writer Benchmark Summary

## Classement global par modèle

1. **gemma3:4b** — score moyen Q1-Q4 : 6.5
2. **llama3.2:latest** — score moyen Q1-Q4 : 6.5
3. **mistral:7b-instruct-q4_0** — score moyen Q1-Q4 : 6.5
4. **qwen2.5:7b-instruct** — score moyen Q1-Q4 : 6.5

## Résumé des scores et du temps

| Modèle | Score moyen Q1-Q4 | llm_writer_rate | fallback_rate | hard_gate_rejection_rate | repair_success_rate | avg_llm_writer_ms | avg_response_time (s) | Fallbacks Q1-Q4 | Fail Q1-Q4 | Warning Q1-Q4 | Temps moyen (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma3:4b | 6.5 | 50.0% | 50.0% | 50.0% | 0.0% | 17816.637 | 40.65 | 2 | 0 | 1 | 40.65 |
| llama3.2:latest | 6.5 | 50.0% | 50.0% | 50.0% | 0.0% | 17985.646 | 39.483 | 2 | 0 | 1 | 39.483 |
| mistral:7b-instruct-q4_0 | 6.5 | 50.0% | 50.0% | 50.0% | 0.0% | 17789.933 | 40.689 | 2 | 0 | 1 | 40.689 |
| qwen2.5:7b-instruct | 6.5 | 50.0% | 50.0% | 50.0% | 0.0% | 17814.449 | 40.696 | 2 | 0 | 1 | 40.696 |

## Détails des fallbacks et warnings

### fallback_by_reason
- llm_validation_fail_hard_gate: 4
- llm_repair_failed: 4

### warnings_by_type
- over_verbose_intro: 8
- downgraded_non_fact_error:unsupported_analyte: 8
- Some numeric values were not found in evidence.: 4
- missing_conclusion: 4
- patient_inventory_long_cell:Cellule Markdown > 150 char.: 4
- downgraded_non_fact_error:unsupported_value: 4

## Erreurs fréquentes


### Avertissements
- over_verbose_intro: 8
- downgraded_non_fact_error:unsupported_analyte: 8
- Some numeric values were not found in evidence.: 4
- missing_conclusion: 4
- patient_inventory_long_cell:Cellule Markdown > 150 char.: 4
- downgraded_non_fact_error:unsupported_value: 4

## Détails par modèle / question

| Modèle | Question | Score | Route | Writer | Validation | Qualité | Fallback | Temps | Answer preview |
|---|---|---|---|---|---|---|---|---|---|
| llama3.2:latest | Q1 | 5 | doc_scoped_biological_summary | professional_fallback | pass | warning | llm_validation_fail_hard_gate | 33.755 | Anormaux : Bilirubine Directe (au-dessus), CRÉATININE (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus). Normaux /  |
| llama3.2:latest | Q2 | 6 | doc_scoped_biological_summary | llm_writer | warning | warning |  | 32.574 | Anormaux - CRP : 7 mg/l ; statut haut. - Réserve Alcaline : 20 mmol/l ; statut bas. Résultats dans la référence uniqueme |
| llama3.2:latest | Q3 | 6 | doc_scoped_priority_anomalies | professional_fallback | pass | warning | llm_repair_failed | 51.844 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| llama3.2:latest | Q4 | 9 | doc_scoped_medical_interpretation_guarded | llm_writer | pass | pass |  | 39.759 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  - TSH : 55,00 mUI/L (au-dessus de la réf |
| llama3.2:latest | Q5 | 8 | doc_scoped_toxicology_summary | deterministic_toxicology_renderer | pass | pass |  | 0.123 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| qwen2.5:7b-instruct | Q1 | 5 | doc_scoped_biological_summary | professional_fallback | pass | warning | llm_validation_fail_hard_gate | 39.202 | Anormaux : Bilirubine Directe (au-dessus), CRÉATININE (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus). Normaux /  |
| qwen2.5:7b-instruct | Q2 | 6 | doc_scoped_biological_summary | llm_writer | warning | warning |  | 32.015 | Anormaux - CRP : 7 mg/l ; statut haut. - Réserve Alcaline : 20 mmol/l ; statut bas. Résultats dans la référence uniqueme |
| qwen2.5:7b-instruct | Q3 | 6 | doc_scoped_priority_anomalies | professional_fallback | pass | warning | llm_repair_failed | 51.935 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| qwen2.5:7b-instruct | Q4 | 9 | doc_scoped_medical_interpretation_guarded | llm_writer | pass | pass |  | 39.632 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  - TSH : 55,00 mUI/L (au-dessus de la réf |
| qwen2.5:7b-instruct | Q5 | 8 | doc_scoped_toxicology_summary | deterministic_toxicology_renderer | pass | pass |  | 0.124 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| mistral:7b-instruct-q4_0 | Q1 | 5 | doc_scoped_biological_summary | professional_fallback | pass | warning | llm_validation_fail_hard_gate | 39.375 | Anormaux : Bilirubine Directe (au-dessus), CRÉATININE (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus). Normaux /  |
| mistral:7b-instruct-q4_0 | Q2 | 6 | doc_scoped_biological_summary | llm_writer | warning | warning |  | 31.848 | Anormaux - CRP : 7 mg/l ; statut haut. - Réserve Alcaline : 20 mmol/l ; statut bas. Résultats dans la référence uniqueme |
| mistral:7b-instruct-q4_0 | Q3 | 6 | doc_scoped_priority_anomalies | professional_fallback | pass | warning | llm_repair_failed | 51.842 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| mistral:7b-instruct-q4_0 | Q4 | 9 | doc_scoped_medical_interpretation_guarded | llm_writer | pass | pass |  | 39.693 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  - TSH : 55,00 mUI/L (au-dessus de la réf |
| mistral:7b-instruct-q4_0 | Q5 | 8 | doc_scoped_toxicology_summary | deterministic_toxicology_renderer | pass | pass |  | 0.124 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |
| gemma3:4b | Q1 | 5 | doc_scoped_biological_summary | professional_fallback | pass | warning | llm_validation_fail_hard_gate | 39.203 | Anormaux : Bilirubine Directe (au-dessus), CRÉATININE (au-dessus), LDH (au-dessus), CKMB (CPKMB) (au-dessus). Normaux /  |
| gemma3:4b | Q2 | 6 | doc_scoped_biological_summary | llm_writer | warning | warning |  | 32.157 | Anormaux - CRP : 7 mg/l ; statut haut. - Réserve Alcaline : 20 mmol/l ; statut bas. Résultats dans la référence uniqueme |
| gemma3:4b | Q3 | 6 | doc_scoped_priority_anomalies | professional_fallback | pass | warning | llm_repair_failed | 51.753 | Les anomalies techniques ci-dessous sont organisées par section du rapport.  4 valeurs exploitables ont été retrouvées.  |
| gemma3:4b | Q4 | 9 | doc_scoped_medical_interpretation_guarded | llm_writer | pass | pass |  | 39.488 | Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls.  - TSH : 55,00 mUI/L (au-dessus de la réf |
| gemma3:4b | Q5 | 8 | doc_scoped_toxicology_summary | deterministic_toxicology_renderer | pass | pass |  | 0.123 | Synthèse toxicologique technique  - Sous seuil: 11 - Au-dessus du seuil: 2 - Référence manquante/ambiguë: 5  Conclusion  |

## Check non-régression Q5

Les résultats de Q5 sont inclus dans le JSON et le CSV, mais le score global LLM writer est calculé uniquement sur Q1-Q4.
