# Indexing - build hybrid retrieval stores

## Role
This stage builds the retrieval indexes from anonymized chunks.

Pipeline position:

`data/chunks/chunks.anonymized.jsonl -> indexing -> data/indexes/*`

## Security gate
`build_indexes.py` blocks:
- `data/chunks/chunks.raw.jsonl`
- any `data/private/` path
- any chunk file other than `data/chunks/chunks.anonymized.jsonl`

## Stores created
- Qdrant local: vector index
- SQLite FTS5: keyword index
- SQLite tables: metadata + provenance/object references

## Dependencies
```bash
pip install qdrant-client FlagEmbedding sentence-transformers numpy torch
```

## Build indexes (full hybrid)
```bash
python scripts/indexing/build_indexes.py \
  --chunks data/chunks/chunks.anonymized.jsonl \
  --index-dir data/indexes \
  --collection medical_chunks \
  --embedding-model BAAI/bge-m3 \
  --batch-size 2 \
  --reset
```

Fallback embedding model:
```bash
python scripts/indexing/build_indexes.py \
  --chunks data/chunks/chunks.anonymized.jsonl \
  --index-dir data/indexes \
  --collection medical_chunks \
  --embedding-model intfloat/multilingual-e5-base \
  --batch-size 8 \
  --reset
```

Keyword+metadata only (skip vectors):
```bash
python scripts/indexing/build_indexes.py \
  --chunks data/chunks/chunks.anonymized.jsonl \
  --index-dir data/indexes \
  --collection medical_chunks \
  --skip-vector \
  --reset
```

## Validate built indexes
```bash
python scripts/indexing/validate_indexes.py \
  --index-dir data/indexes \
  --collection medical_chunks \
  --smoke-test
```

## Expected artifacts
- `data/indexes/medical_rag.sqlite`
- `data/indexes/qdrant/`
- `data/indexes/indexing_report.json`
- `data/indexes/index_manifest.json`
- `data/indexes/index_validation_report.json` (after validation script)
