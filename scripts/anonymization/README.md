# Chunk Anonymization Phase

This module anonymizes raw chunk output before indexing.

## Pipeline Position

`data/extraction/**/document.json`
→ `scripts/chunking/build_chunks.py`
→ `data/chunks/chunks.raw.jsonl` (raw, sensitive)
→ `scripts/anonymization/anonymize_chunks.py`
→ `data/chunks/chunks.anonymized.jsonl` (indexable)
→ indexing

## Important Rules

- Never index `chunks.raw.jsonl`.
- Only index `chunks.anonymized.jsonl`.
- Mapping is private and reversible. Never commit it.

## Run

```bash
python3 scripts/anonymization/anonymize_chunks.py \
  --input data/chunks/chunks.raw.jsonl \
  --output data/chunks/chunks.anonymized.jsonl \
  --mapping-xlsx data/private/anonymization_mapping.xlsx \
  --report data/chunks/anonymization_report.json
```

## Outputs

- `data/chunks/chunks.anonymized.jsonl`
- `data/chunks/anonymization_report.json`
- preferred: `data/private/anonymization_mapping.xlsx`
- fallback if `openpyxl` missing: `data/private/anonymization_mapping.csv`

## Tokenization Logic

Stable reversible sequence tokens are used:

- patient IDs (`patient_id`, `patient_id_raw`, `ip_patient`) → `PAT_000001`
- sample IDs (`sample_id`, `sample_id_raw`) → `SAMPLE_000001`
- report IDs (`report_id`, `report_id_raw`) → `REPORT_000001`

Mapping is reused across runs:

- existing mapping is loaded if found
- known original values reuse same token
- new values receive next sequence token

## Exit Codes

- `0`: `ready_for_indexing`
- `1`: `blocked` (validation errors)

## Notes

- The script does not modify the raw input file.
- It validates parent links, duplicate IDs, forbidden raw metadata fields, and text leakage of sensitive values.
