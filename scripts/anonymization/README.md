# Anonymization - raw chunks to safe indexing chunks

## Role
This stage anonymizes raw chunks before indexing.

Pipeline position:

`data/chunks/chunks.raw.jsonl -> anonymization -> data/chunks/chunks.anonymized.jsonl -> indexing`

## Inputs
- `--input`: raw chunks JSONL
- `--mapping-xlsx`: private reversible mapping file path

## Outputs
- `data/chunks/chunks.anonymized.jsonl`
- `data/chunks/anonymization_report.json`
- `data/private/anonymization_mapping.xlsx` (or CSV fallback if `openpyxl` missing)

## Run
```bash
python scripts/anonymization/anonymize_chunks.py \
  --input data/chunks/chunks.raw.jsonl \
  --output data/chunks/chunks.anonymized.jsonl \
  --mapping-xlsx data/private/anonymization_mapping.xlsx \
  --report data/chunks/anonymization_report.json
```

## Guarantees implemented in code
- Stable tokens for sensitive IDs (`PAT_*`, `SAMPLE_*`, `REPORT_*`).
- `chunk_id` unchanged.
- `parent_chunk_id` unchanged.
- `content_hash` recomputed on final anonymized payload (deterministic).
- Validation detects missing/mismatched hash and leakage patterns.

## Readiness checks
After run, verify in `data/chunks/anonymization_report.json`:
- `validation_errors` is empty
- `readiness_status == "ready_for_indexing"`

## Security rules
- Never index `chunks.raw.jsonl`.
- Never expose or commit mapping files under `data/private/`.
