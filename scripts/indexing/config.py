from __future__ import annotations

from pathlib import Path

ALLOWED_SCHEMA_VERSION = "clinical_chunk_anonymized_v1"
FORBIDDEN_SCHEMA_VERSIONS = {
    "clinical_chunk_raw_v2",
    "clinical_chunk_raw_v1",
    "clinical_chunk_v2",
    "clinical_chunk_v1",
}

REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "doc_id",
    "chunk_type",
    "text_for_embedding",
    "text_for_keyword",
    "metadata",
    "provenance",
    "quality",
    "routing",
}

FORBIDDEN_METADATA_FIELDS = {
    "patient_id",
    "patient_id_raw",
    "ip_patient",
    "patient_birth_date",
    "patient_birth_date_raw",
    "patient_address",
    "prescriber",
    "validated_by",
    "edited_by",
    "printed_by",
    "phone",
    "fax",
}

FORBIDDEN_TEXT_PATTERNS = [
    "PYXIS TEST",
    "PATIENT TEST1",
    "Dr.",
    "Prescripteur",
    "Validé(e) par",
    "Edité(e) par",
    "Imprimé par",
]

DEFAULT_BGE_MODEL = "BAAI/bge-m3"
DEFAULT_E5_MODEL = "intfloat/multilingual-e5-base"

DEFAULT_BATCH_BGE = 2
DEFAULT_BATCH_E5 = 8

DEFAULT_COLLECTION = "medical_chunks"

QDRANT_DIRNAME = "qdrant"
SQLITE_FILENAME = "medical_rag.sqlite"
INDEXING_REPORT_FILENAME = "indexing_report.json"
INDEX_MANIFEST_FILENAME = "index_manifest.json"
INDEX_VALIDATION_REPORT_FILENAME = "index_validation_report.json"

SMOKE_TERMS = ["CRP", "ACTH", "Trichuris", "Lithium", "Ferritine"]


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

