# Pour executer l'extraction des pdf 
(venv deja active)

for pdf in docs/*.pdf; do
  ./venv/bin/python scripts/extraction_data/run_extraction.py "$pdf"
done


## pour executer les test de regession ( si tout va bien  , on verra OK) 

./venv/bin/python -m unittest tests/test_extraction_deduplication.py


## Schema de validation (EXTRACTION) ,pour executer le Script QA (validation du doc.json de chaque pdf)

./venv/bin/python tests/validate_extraction_outputs.py


## pour executer l'Anonymization

python3 scripts/anonymization/anonymize_chunks.py \
  --input data/chunks/chunks.raw.jsonl \
  --output data/chunks/chunks.anonymized.jsonl \
  --mapping-xlsx data/private/anonymization_mapping.xlsx \
  --report data/chunks/anonymization_report.json


# Status

### extraction (done)
### chunking  (soon)
### anonymisation  (soon)
### indexing (soon)
### hybrid retrieval (soon)
### multimodal RAG (soon)


# Ce que j'ai  déjà validé


### document.json → chunks → anonymization → vector index → keyword index → metadata index → hybrid retrieval → strict context


#### Offline pipeline : prêt
#### Retrieval layer  : prêt
#### Context gating   : prêt
#### LLM generation   : prochaine étape

Clinical Structuring
        ↓
Chunk Builder
        ↓
1. Atomic result chunks
2. Section chunks
3. Document summary chunks
4. Visual/table reference chunks
        ↓
Anonymization
        ↓
Embedding + Keyword indexing
        ↓
Hybrid retrieval


