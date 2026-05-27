# LLM Writer Today Check

- Generated at: 2026-05-26T22:04:03+01:00
- Base URL: http://127.0.0.1:8000
- Conversation: conv_a9220fda-0810-45a2-8b52-1c1a892261e0

## Summary

- Total: 7
- Passed: 3
- Failed: 4

## Results

- [PASS] doc_scoped_biological_summary_short | route=doc_scoped_biological_summary | mode=hybrid_structured_llm_writer | validation=pass | note=ok
- [PASS] doc_scoped_biological_summary_note | route=doc_scoped_biological_summary | mode=hybrid_structured_llm_writer | validation=pass | note=ok
- [PASS] doc_scoped_medical_interpretation_guarded | route=doc_scoped_medical_interpretation_guarded | mode=deterministic_guarded_medical_interpretation | validation=pass | note=ok
- [FAIL] open_grounded_medical_question#1 | route=open_grounded_medical_question | mode=deterministic_no_evidence_response | validation=pass | note=route_or_llm_not_matching
- [FAIL] open_grounded_medical_question#2 | route=unstructured | mode=llm | validation=fail | note=route_or_llm_not_matching
- [FAIL] open_grounded_medical_question#3 | route=cohort_search | mode=deterministic_no_evidence_response | validation=warning | note=route_or_llm_not_matching
- [FAIL] response_transform | route= | mode=deterministic_response_transform_professional | validation=fail | note=route_mismatch:
