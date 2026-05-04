# Medical Multimodal RAG - Runbook PFE (End-to-End)

Ce runbook donne un enchainement simple et reproductible pour executer tout le pipeline:

`extraction -> chunking -> anonymization -> indexing -> retrieval`

## Prerequis
- Environnement virtuel actif (`venv`)
- PDFs dans `docs/`
- Python deps installees pour extraction/chunking/anonymization/indexing/retrieval

## 10 commandes (ordre recommande)

### 1) Extraction sur tout le corpus PDF
```bash
find docs -maxdepth 1 -type f -name '*.pdf' -print0 | while IFS= read -r -d '' pdf; do
  python scripts/extraction_data/run_extraction.py "$pdf" --output-root data/extraction
 done
```

### 2) Validation extraction (QA)
```bash
python tests/validate_extraction_outputs.py \
  --root data/extraction \
  --report-json data/extraction/qa_validation_report.json
```

### 3) Chunking global + strict quality gates
```bash
python scripts/chunking/build_chunks.py \
  --input-root data/extraction \
  --output data/chunks/chunks.raw.jsonl \
  --report data/chunks/chunking_report.json \
  --write-local-chunks \
  --strict-quality-gates
```

### 4) Tests unitaires anonymization
```bash
python -m unittest tests/test_anonymization.py -v
```

### 5) Anonymization des chunks
```bash
python scripts/anonymization/anonymize_chunks.py \
  --input data/chunks/chunks.raw.jsonl \
  --output data/chunks/chunks.anonymized.jsonl \
  --mapping-xlsx data/private/anonymization_mapping.xlsx \
  --report data/chunks/anonymization_report.json
```

### 6) Build indexing (hybrid complet, modele par defaut)
```bash
python scripts/indexing/build_indexes.py \
  --chunks data/chunks/chunks.anonymized.jsonl \
  --index-dir data/indexes \
  --collection medical_chunks \
  --embedding-model BAAI/bge-m3 \
  --batch-size 2 \
  --reset
```

### 7) Validation des index
```bash
python scripts/indexing/validate_indexes.py \
  --index-dir data/indexes \
  --collection medical_chunks \
  --smoke-test
```

### 8) Requete retrieval (hybrid)
```bash
python scripts/retrieval/search.py \
  --query "calcitonine" \
  --mode hybrid \
  --top-k 5
```

### 9) Requete retrieval avec filtres
```bash
python scripts/retrieval/search.py \
  --query "procalcitonine" \
  --mode hybrid \
  --doc-id report_1 \
  --chunk-type lab_result \
  --top-k 5 \
  --json
```

### 10) Validation retrieval (jeu de tests)
```bash
python scripts/retrieval/validate_retrieval.py \
  --mode all \
  --top-k 5 \
  --index-dir data/indexes \
  --collection medical_chunks \
  --report data/retrieval/retrieval_validation_report.json
```

## Notes importantes
- `run_extraction.py` prend un PDF (fichier), pas un dossier.
- Ne jamais indexer `data/chunks/chunks.raw.jsonl`.
- Toujours indexer `data/chunks/chunks.anonymized.jsonl`.
- Si `intfloat/multilingual-e5-base` est utilise, un acces internet est necessaire pour telecharger le modele si non cache localement.

## Artefacts finaux attendus
- Extraction: `data/extraction/**/document.json`
- Chunking: `data/chunks/chunks.raw.jsonl`, `data/chunks/chunking_report.json`
- Anonymization: `data/chunks/chunks.anonymized.jsonl`, `data/chunks/anonymization_report.json`
- Indexing: `data/indexes/medical_rag.sqlite`, `data/indexes/qdrant/`, `data/indexes/indexing_report.json`
- Retrieval validation: `data/retrieval/retrieval_validation_report.json`

## Lancer Tout Le Pipeline (One-shot)
```bash
./scripts/run_pipeline.sh
```

Le script execute automatiquement: extraction -> validation extraction -> chunking -> tests anonymization -> anonymization -> indexing -> validation indexing -> retrieval smoke test -> retrieval validation.
