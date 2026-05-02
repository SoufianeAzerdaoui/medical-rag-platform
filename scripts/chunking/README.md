# Chunking - document.json to raw RAG chunks

## Role
This stage transforms extracted `document.json` files into raw clinical chunks for RAG.

Pipeline position:

`data/extraction/**/document.json -> chunking -> data/chunks/chunks.raw.jsonl`

## Inputs
- Input root with `*/document.json` files (default: `data/extraction`)

## Outputs
- Global raw chunks: `data/chunks/chunks.raw.jsonl`
- Chunking report: `data/chunks/chunking_report.json`
- Optional per-document chunks: `data/extraction/<report_dir>/chunks.jsonl`

## Run (recommended)
```bash
python scripts/chunking/build_chunks.py \
  --input-root data/extraction \
  --output data/chunks/chunks.raw.jsonl \
  --report data/chunks/chunking_report.json \
  --write-local-chunks \
  --strict-quality-gates
```

## Useful flags
- `--write-local-chunks`: also writes one `chunks.jsonl` per extracted document folder.
- `--fail-on-error`: stop immediately on first document failure.
- `--strict-quality-gates`: fail if validation issues are detected.

## Post-run quick checks
- `data/chunks/chunks.raw.jsonl` exists and is non-empty.
- `data/chunks/chunking_report.json` exists.
- In report, `errors` should be empty.
- In strict mode, command should exit with code `0`.

## Security reminder
- `chunks.raw.jsonl` can contain sensitive fields.
- Do not index raw chunks.
- Run anonymization before indexing.
