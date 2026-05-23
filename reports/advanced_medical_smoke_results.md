# Advanced Medical Smoke Results

- Base URL: `http://127.0.0.1:8000`
- Conversation ID: `conv_256bf6e2-81a6-4411-a12b-2f1f87df86fa`
- Score: **10/10**
- Elapsed: `1.237s`

| # | Status | selected_route | generation_mode | validation_status | quality | response_time | retrieval_ms | llm_writer_ms | displayed | sources |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | PASS | reference_range_lookup | deterministic_reference_range_lookup | pass | pass | 0.083 | 0.000 | 0.000 | 0 | 3 |
| 2 | PASS | reference_range_lookup | deterministic_reference_range_lookup | pass | pass | 0.062 | 0.000 | 0.000 | 0 | 3 |
| 3 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | warning | 0.052 | 0.000 | 0.000 | 1 | 1 |
| 4 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | warning | 0.055 | 0.000 | 0.000 | 1 | 1 |
| 5 | PASS | global_analyte_abnormal_search | deterministic_global_analyte_abnormal_search | pass | pass | 0.080 | 0.000 | 0.000 | 5 | 5 |
| 6 | PASS | global_toxicology_search | deterministic_global_toxicology_search | pass | pass | 0.121 | 0.000 | 0.000 | 5 | 5 |
| 7 | PASS | global_toxicology_search | deterministic_global_toxicology_search | pass | pass | 0.091 | 0.000 | 0.000 | 3 | 3 |
| 8 | PASS | doc_scoped_toxicology_threshold_search | deterministic_doc_scoped_toxicology_threshold_search | pass | pass | 0.080 | 0.000 | 0.000 | 2 | 2 |
| 9 | PASS | doc_scoped_toxicology_summary | deterministic_doc_scoped_toxicology_summary | pass | pass | 0.116 | 0.000 | 0.000 | 13 | 13 |
| 10 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | warning | 0.054 | 0.000 | 0.000 | 1 | 1 |

## Fail Details
- None
