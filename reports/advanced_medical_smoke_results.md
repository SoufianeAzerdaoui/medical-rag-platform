# Advanced Medical Smoke Results

- Base URL: `http://127.0.0.1:8000`
- Conversation ID: `conv_beee5560-514c-4b0c-bea8-7298b0826c06`
- Score: **10/10**
- Elapsed: `2.09s`

| # | Status | selected_route | generation_mode | validation_status | quality | response_time | retrieval_ms | llm_writer_ms | displayed | sources |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | PASS | reference_range_lookup | deterministic_reference_range_lookup | pass | pass | 0.176 | 0.000 | 0.000 | 0 | 3 |
| 2 | PASS | reference_range_lookup | deterministic_reference_range_lookup | pass | pass | 0.121 | 0.000 | 0.000 | 0 | 3 |
| 3 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | pass | 0.100 | 0.000 | 0.000 | 1 | 1 |
| 4 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | pass | 0.102 | 0.000 | 0.000 | 1 | 1 |
| 5 | PASS | global_analyte_abnormal_search | deterministic_global_analyte_abnormal_search | pass | warning | 0.149 | 0.000 | 0.000 | 5 | 5 |
| 6 | PASS | global_toxicology_search | deterministic_global_toxicology_search | pass | pass | 0.203 | 0.000 | 0.000 | 5 | 5 |
| 7 | PASS | global_toxicology_search | deterministic_global_toxicology_search | pass | pass | 0.144 | 0.000 | 0.000 | 3 | 3 |
| 8 | PASS | doc_scoped_toxicology_threshold_search | deterministic_doc_scoped_toxicology_threshold_search | pass | pass | 0.144 | 0.000 | 0.000 | 2 | 2 |
| 9 | PASS | doc_scoped_toxicology_summary | deterministic_doc_scoped_toxicology_summary | pass | pass | 0.190 | 0.000 | 0.000 | 13 | 13 |
| 10 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | pass | 0.099 | 0.000 | 0.000 | 1 | 1 |

## Fail Details
- None
