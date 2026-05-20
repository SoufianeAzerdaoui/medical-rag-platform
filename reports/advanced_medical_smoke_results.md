# Advanced Medical Smoke Results

- Base URL: `http://127.0.0.1:8000`
- Conversation ID: `conv_ccf418e1-b32b-4175-ab5f-085795bb786d`
- Score: **10/10**
- Elapsed: `1.245s`

| # | Status | selected_route | generation_mode | validation_status | quality | response_time | retrieval_ms | llm_writer_ms | displayed | sources |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | PASS | reference_range_lookup | deterministic_reference_range_lookup | warning | warning | 0.156 | 0.000 | 0.000 | 0 | 6 |
| 2 | PASS | reference_range_lookup | deterministic_reference_range_lookup | warning | warning | 0.067 | 0.000 | 0.000 | 0 | 6 |
| 3 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | pass | 0.047 | 0.000 | 0.000 | 1 | 1 |
| 4 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | pass | 0.044 | 0.000 | 0.000 | 1 | 1 |
| 5 | PASS | global_analyte_abnormal_search | deterministic_global_analyte_abnormal_search | pass | pass | 0.070 | 0.000 | 0.000 | 5 | 5 |
| 6 | PASS | global_toxicology_search | deterministic_global_toxicology_search | pass | pass | 0.118 | 0.000 | 0.000 | 83 | 83 |
| 7 | PASS | global_toxicology_search | deterministic_global_toxicology_search | pass | pass | 0.070 | 0.000 | 0.000 | 8 | 8 |
| 8 | PASS | doc_scoped_toxicology_threshold_search | deterministic_doc_scoped_toxicology_threshold_search | pass | pass | 0.065 | 0.000 | 0.000 | 2 | 2 |
| 9 | PASS | doc_scoped_toxicology_summary | deterministic_doc_scoped_toxicology_summary | pass | pass | 0.097 | 0.000 | 0.000 | 13 | 13 |
| 10 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | warning | 0.048 | 0.000 | 0.000 | 1 | 1 |

## Fail Details
- None
