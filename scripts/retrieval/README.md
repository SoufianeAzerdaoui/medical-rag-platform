# Retrieval - query hybrid medical indexes

## Role
This stage executes keyword/vector/hybrid retrieval on built indexes.

Pipeline position:

`indexing -> retrieval`

Main CLI:
- `scripts/retrieval/search.py`

Validation CLI:
- `scripts/retrieval/validate_retrieval.py`

## Prerequisites
- Indexes already built in `data/indexes`.
- Expected artifacts:
  - `data/indexes/medical_rag.sqlite`
  - `data/indexes/qdrant/` (required for vector/hybrid modes)

## Search examples
Hybrid search:
```bash
python scripts/retrieval/search.py \
  --query "calcitonine" \
  --mode hybrid \
  --top-k 5
```

Keyword-only search:
```bash
python scripts/retrieval/search.py \
  --query "trichuris trichiura" \
  --mode keyword \
  --top-k 5
```

Vector-only search:
```bash
python scripts/retrieval/search.py \
  --query "hormone thyroidienne" \
  --mode vector \
  --top-k 5
```

Search with filters:
```bash
python scripts/retrieval/search.py \
  --query "procalcitonine" \
  --mode hybrid \
  --doc-id report_1 \
  --chunk-type lab_result \
  --top-k 5
```

JSON output:
```bash
python scripts/retrieval/search.py \
  --query "peptide c" \
  --mode hybrid \
  --top-k 5 \
  --json
```

## Validate retrieval quality
Default test cases from `tests/retrieval_test_cases.json`:
```bash
python scripts/retrieval/validate_retrieval.py \
  --mode all \
  --top-k 5 \
  --index-dir data/indexes \
  --collection medical_chunks \
  --report data/retrieval/retrieval_validation_report.json
```

## Notes
- `--mode keyword` can run without vector store.
- `--mode vector` and `--mode hybrid` require Qdrant vectors.
- Context expansion is enabled by default in `search.py` and can be disabled with `--no-expand-context`.
