from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "data" / "indexes"
SQLITE_PATH = INDEX_DIR / "medical_rag.sqlite"
QDRANT_DIR = INDEX_DIR / "qdrant"
QDRANT_COLLECTION = "medical_chunks"

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_FALLBACK_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_BATCH_SIZE = 2
DEFAULT_DEVICE = "cpu"
DEFAULT_EMBEDDING_DIMENSION = 1024

DEFAULT_TOP_K = 5
DEFAULT_KEYWORD_TOP_K = 20
DEFAULT_VECTOR_TOP_K = 20
DEFAULT_RRF_K = 60
DEFAULT_CONTEXT_MAX_CHUNKS = 8

VALIDATION_REPORT_PATH = REPO_ROOT / "data" / "retrieval" / "retrieval_validation_report.json"
DEFAULT_TEST_CASES_PATH = REPO_ROOT / "tests" / "retrieval_test_cases.json"

MAPPING_XLSX = REPO_ROOT / "data" / "private" / "anonymization_mapping.xlsx"
MAPPING_CSV = REPO_ROOT / "data" / "private" / "anonymization_mapping.csv"

TECHNICAL_LABELS = [
    "results_table_block",
    "patient_info_block",
    "validation_block",
    "facility_block",
    "document_header",
    "footer_block",
]

PARASITOLOGY_CONTEXT_HINTS = {
    "resultat_final",
    "examen_apres_coloration",
    "examen_apres_enrichissement",
    "examen_microscopique",
}
