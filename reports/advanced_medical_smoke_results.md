# Advanced Medical Smoke Results

- Base URL: `http://127.0.0.1:8000`
- Conversation ID: `conv_d2e8991d-210d-4d23-835b-b68be40c5925`
- Score: **10/10**
- Elapsed: `1.258s`

| # | Status | selected_route | generation_mode | validation_status | quality | response_time | retrieval_ms | llm_writer_ms | displayed | sources |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | PASS | reference_range_lookup | deterministic_reference_range_lookup | pass | pass | 0.093 | 0.000 | 0.000 | 0 | 3 |
| 2 | PASS | reference_range_lookup | deterministic_reference_range_lookup | pass | pass | 0.060 | 0.000 | 0.000 | 0 | 3 |
| 3 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | pass | 0.054 | 0.000 | 0.000 | 1 | 1 |
| 4 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | pass | 0.051 | 0.000 | 0.000 | 1 | 1 |
| 5 | PASS | global_analyte_abnormal_search | deterministic_global_analyte_abnormal_search | pass | pass | 0.078 | 0.000 | 0.000 | 5 | 5 |
| 6 | PASS | global_toxicology_search | deterministic_global_toxicology_search | pass | pass | 0.118 | 0.000 | 0.000 | 5 | 5 |
| 7 | PASS | global_toxicology_search | deterministic_global_toxicology_search | pass | pass | 0.084 | 0.000 | 0.000 | 3 | 3 |
| 8 | PASS | doc_scoped_toxicology_threshold_search | deterministic_doc_scoped_toxicology_threshold_search | pass | pass | 0.079 | 0.000 | 0.000 | 2 | 2 |
| 9 | PASS | doc_scoped_toxicology_summary | deterministic_doc_scoped_toxicology_summary | pass | pass | 0.143 | 0.000 | 0.000 | 13 | 13 |
| 10 | PASS | doc_scoped_single_analyte_status | deterministic_single_analyte_lookup | pass | pass | 0.060 | 0.000 | 0.000 | 1 | 1 |

## Fail Details
- None
