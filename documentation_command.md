# Pour executer l'extraction des pdf 
(venv deja active)

for pdf in docs/*.pdf; do
  ./venv/bin/python scripts/extraction_data/run_extraction.py "$pdf"
done


## pour executer les test de regession ( si tout va bien  , on verra OK) 

./venv/bin/python -m unittest tests/test_extraction_deduplication.py


## Schema de validation (EXTRACTION) ,pour executer le Script QA (validation du doc.json de chaque pdf)

./venv/bin/python tests/validate_extraction_outputs.py


# Status

### extraction (done)
### chunking  (soon)
### anonymisation  (soon)
### indexing (soon)
### hybrid retrieval (soon)
### multimodal RAG (soon)



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