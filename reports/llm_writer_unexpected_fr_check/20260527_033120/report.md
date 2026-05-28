# LLM Writer Unexpected FR Check

- Generated at: 2026-05-27T03:31:30+01:00
- Base URL: http://127.0.0.1:8000
- Conversation: conv_a9220fda-0810-45a2-8b52-1c1a892261e0

## Summary

- Total: 5
- Passed: 3
- Failed: 2

## Results

- [PASS] fr_unexpected_range_female_profile | route=reference_range_lookup | mode=deterministic_reference_range_lookup | validation=pass | note=ok
- [PASS] fr_unexpected_doc_status_in_reference | route=reference_range_lookup | mode=deterministic_reference_range_lookup | validation=pass | note=ok
- [FAIL] fr_unexpected_summary_in_reference_focus | route=doc_scoped_abnormal_results | mode=deterministic_doc_scoped_abnormal_results | validation=pass | note=answer_style_missing
- [FAIL] fr_unexpected_summary_writer_forced | route=doc_scoped_biological_summary | mode=deterministic_doc_scoped_biological_summary | validation=pass | note=llm_not_attempted
- [PASS] fr_unexpected_doctor_note_physiological_ranges | route=doc_scoped_biological_summary | mode=hybrid_structured_llm_writer | validation=warning | note=ok
