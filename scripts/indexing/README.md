# Indexing Phase - Medical RAG Platform

## Objectif
La phase `indexing` construit les index necessaires pour la retrieval (plus tard), a partir des chunks deja anonymises.

Pipeline:

`data/extraction/**/document.json` -> chunking -> anonymization -> `data/chunks/chunks.anonymized.jsonl` -> indexing

## Regle de securite critique
- Seul fichier autorise: `data/chunks/chunks.anonymized.jsonl`
- Interdit: `data/chunks/chunks.raw.jsonl`
- Interdit: tout `data/private/` (dont mapping anonymization)

Ces regles evitent toute fuite de donnees brutes sensibles.

## Composants indexes
- Qdrant local: vector store semantique (embeddings)
- SQLite FTS5: keyword store lexical
- SQLite metadata: filtres structuraux cliniques
- SQLite object_references: provenance et references objets

Pas de reranker pour le moment.
Pas de retrieval fusion pour le moment.

## Installation

Installation complete:

```bash
pip install qdrant-client FlagEmbedding sentence-transformers numpy torch
```

Fallback plus leger:

```bash
pip install qdrant-client sentence-transformers numpy torch
```

## Build des index

Mode par defaut (BAAI/bge-m3):

```bash
python3 scripts/indexing/build_indexes.py \
  --chunks data/chunks/chunks.anonymized.jsonl \
  --index-dir data/indexes \
  --collection medical_chunks \
  --embedding-model BAAI/bge-m3 \
  --batch-size 2 \
  --reset
```

Fallback (intfloat/multilingual-e5-base):

```bash
python3 scripts/indexing/build_indexes.py \
  --chunks data/chunks/chunks.anonymized.jsonl \
  --index-dir data/indexes \
  --collection medical_chunks \
  --embedding-model intfloat/multilingual-e5-base \
  --batch-size 8 \
  --reset
```

## Validation des index

```bash
python3 scripts/indexing/validate_indexes.py \
  --index-dir data/indexes \
  --collection medical_chunks \
  --smoke-test
```

## Conseils laptop CPU
- `BAAI/bge-m3`: batch-size recommande `2`
- fallback `intfloat/multilingual-e5-base`: batch-size recommande `8`

Le script detecte automatiquement `cuda` si disponible, sinon `cpu`.

