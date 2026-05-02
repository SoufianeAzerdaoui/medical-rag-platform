#!/usr/bin/env bash
set -euo pipefail

PIPELINE_START_TS=$(date +%s)

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '\n[%s] WARNING: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

format_duration() {
  local total_secs="$1"
  local h=$((total_secs / 3600))
  local m=$(((total_secs % 3600) / 60))
  local s=$((total_secs % 60))
  printf '%02dh:%02dm:%02ds' "${h}" "${m}" "${s}"
}

run_timed() {
  local step_label="$1"
  shift

  local step_start step_end step_elapsed
  step_start=$(date +%s)
  "$@"
  step_end=$(date +%s)
  step_elapsed=$((step_end - step_start))
  log "${step_label} termine en $(format_duration "${step_elapsed}")"
}

# 1) Verify project root
if [[ ! -d scripts ]]; then
  echo "ERROR: dossier scripts/ introuvable. Lancez ce script depuis la racine du projet." >&2
  exit 1
fi

if [[ ! -d docs ]]; then
  echo "ERROR: dossier docs/ introuvable. Lancez ce script depuis la racine du projet." >&2
  exit 1
fi

pdf_count=$(find docs -maxdepth 1 -type f -name '*.pdf' | wc -l | tr -d ' ')
if [[ "${pdf_count}" == "0" ]]; then
  echo "ERROR: aucun PDF trouve dans docs/." >&2
  exit 1
fi

log "PDF detectes dans docs/: ${pdf_count}"

# 2) Create required directories
log "Creation des dossiers necessaires"
mkdir -p data/extraction data/chunks data/private data/indexes data/retrieval

# 3) Extraction
log "Etape 1/9 - Extraction de tous les PDFs"
find docs -maxdepth 1 -type f -name '*.pdf' -print0 | while IFS= read -r -d '' pdf; do
  log "Extraction: ${pdf}"
  run_timed "Extraction ${pdf}" python scripts/extraction_data/run_extraction.py "$pdf" --output-root data/extraction
done

# 4) Validate extraction
log "Etape 2/9 - Validation extraction"
run_timed "Validation extraction" python tests/validate_extraction_outputs.py \
  --root data/extraction \
  --report-json data/extraction/qa_validation_report.json

# 5) Chunking
log "Etape 3/9 - Chunking"
run_timed "Chunking" python scripts/chunking/build_chunks.py \
  --input-root data/extraction \
  --output data/chunks/chunks.raw.jsonl \
  --report data/chunks/chunking_report.json \
  --write-local-chunks \
  --strict-quality-gates

# 6) Anonymization tests (optional)
log "Etape 4/9 - Tests anonymization (optionnel)"
if [[ -f tests/test_anonymization.py ]]; then
  run_timed "Tests anonymization" python -m unittest tests/test_anonymization.py -v
else
  warn "tests/test_anonymization.py introuvable. Etape ignoree."
fi

# 7) Anonymization
log "Etape 5/9 - Anonymization"
run_timed "Anonymization" python scripts/anonymization/anonymize_chunks.py \
  --input data/chunks/chunks.raw.jsonl \
  --output data/chunks/chunks.anonymized.jsonl \
  --mapping-xlsx data/private/anonymization_mapping.xlsx \
  --report data/chunks/anonymization_report.json

# 8) Indexing (always anonymized chunks)
log "Etape 6/9 - Indexing (chunks anonymises uniquement)"
run_timed "Indexing" python scripts/indexing/build_indexes.py \
  --chunks data/chunks/chunks.anonymized.jsonl \
  --index-dir data/indexes \
  --collection medical_chunks \
  --embedding-model BAAI/bge-m3 \
  --batch-size 2 \
  --reset

# 9) Validate indexing
log "Etape 7/9 - Validation indexing"
run_timed "Validation indexing" python scripts/indexing/validate_indexes.py \
  --index-dir data/indexes \
  --collection medical_chunks \
  --smoke-test

# Detect retrieval dir name: retrieval or retrival
retrieval_dir=""
if [[ -d scripts/retrieval ]]; then
  retrieval_dir="scripts/retrieval"
elif [[ -d scripts/retrival ]]; then
  retrieval_dir="scripts/retrival"
fi

# 10) Retrieval smoke + filtered search (optional if search.py exists)
log "Etape 8/9 - Retrieval smoke tests"
if [[ -n "${retrieval_dir}" && -f "${retrieval_dir}/search.py" ]]; then
  run_timed "Retrieval smoke query calcitonine" python "${retrieval_dir}/search.py" \
    --query "calcitonine" \
    --mode hybrid \
    --top-k 5

  run_timed "Retrieval filtered query procalcitonine" python "${retrieval_dir}/search.py" \
    --query "procalcitonine" \
    --mode hybrid \
    --doc-id report_1 \
    --chunk-type lab_result \
    --top-k 5 \
    --json
else
  warn "search.py introuvable dans scripts/retrieval ou scripts/retrival. Etape smoke retrieval ignoree."
fi

# 11) Retrieval validation (optional)
log "Etape 9/9 - Validation retrieval (optionnel)"
if [[ -n "${retrieval_dir}" && -f "${retrieval_dir}/validate_retrieval.py" ]]; then
  run_timed "Validation retrieval" python "${retrieval_dir}/validate_retrieval.py" \
    --mode all \
    --top-k 5 \
    --index-dir data/indexes \
    --collection medical_chunks \
    --report data/retrieval/retrieval_validation_report.json
else
  warn "validate_retrieval.py introuvable dans scripts/retrieval ou scripts/retrival. Etape ignoree."
fi

# Final summary
log "Pipeline termine. Artefacts principaux:"
echo "- extraction output        : data/extraction/"
echo "- chunking output          : data/chunks/chunks.raw.jsonl"
echo "- anonymized chunks        : data/chunks/chunks.anonymized.jsonl"
echo "- anonymization report     : data/chunks/anonymization_report.json"
echo "- sqlite index             : data/indexes/medical_rag.sqlite"
echo "- qdrant dir               : data/indexes/qdrant/"
echo "- indexing report          : data/indexes/indexing_report.json"
echo "- validation report        : data/indexes/index_validation_report.json"

if [[ -f data/retrieval/retrieval_validation_report.json ]]; then
  echo "- retrieval report         : data/retrieval/retrieval_validation_report.json"
else
  echo "- retrieval report         : (absent - validation retrieval non executee ou non disponible)"
fi

PIPELINE_END_TS=$(date +%s)
PIPELINE_ELAPSED=$((PIPELINE_END_TS - PIPELINE_START_TS))
log "Duree totale pipeline: $(format_duration "${PIPELINE_ELAPSED}")"
