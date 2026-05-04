# Extraction Data - Medical PDF to document.json

## Role
This stage parses medical PDFs and produces structured extraction artifacts used by chunking.

Pipeline position:

`docs/*.pdf -> extraction_data -> data/extraction/<report_dir>/document.json`

## Inputs
- One PDF path (required by CLI): `scripts/extraction_data/run_extraction.py <pdf_path>`
- Optional output root: `--output-root data/extraction`

Important:
- `run_extraction.py` expects a file path, not a directory path.
- Command like `python scripts/extraction_data/run_extraction.py docs --output-root data/extraction` fails because `docs` is a folder.

## Outputs
For each PDF:
- `data/extraction/<report_dir>/document.json`
- related extraction artifacts in the same report folder (images/tables/text depending on extraction branch)

## Run - Single PDF
```bash
python scripts/extraction_data/run_extraction.py \
  "docs/report (4).pdf" \
  --output-root data/extraction
```

## Run - Full Corpus
```bash
find docs -maxdepth 1 -type f -name '*.pdf' -print0 | while IFS= read -r -d '' pdf; do
  python scripts/extraction_data/run_extraction.py "$pdf" --output-root data/extraction
 done
```

## Validate Extraction Outputs
Minimal validation:
```bash
python tests/validate_extraction_outputs.py
```

Explicit root + JSON report:
```bash
python tests/validate_extraction_outputs.py \
  --root data/extraction \
  --report-json data/extraction/qa_validation_report.json
```

Optional end-to-end checks used in this repo:
```bash
python tests/validate_docs_end_to_end.py
python tests/validate_real_report_extraction.py
```

## Checklist Before Chunking
- `data/extraction/` contains one folder per processed PDF.
- Each folder contains a valid `document.json`.
- `python tests/validate_extraction_outputs.py` completes without blocking errors.
- If warnings exist, review them before running strict chunking quality gates.
