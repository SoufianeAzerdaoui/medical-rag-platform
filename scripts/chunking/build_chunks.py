#!/usr/bin/env python3
"""
Clinical raw chunk builder for multimodal / hybrid RAG over medical PDF extraction JSON files.

Input:
    data/extraction/**/document.json

Output:
    data/chunks/chunks.raw.jsonl
    data/chunks/chunking_report.json

Optional:
    data/extraction/<report_dir>/chunks.jsonl

Design:
    - Document summary chunks: one macro chunk per report.
    - Section chunks: one macro chunk per clinical section.
    - Atomic result chunks: one micro chunk per lab/parasitology result.
    - Visual reference chunks: page/table/image references for multimodal provenance.
    - Validation chunks: validation/editing status.

Important:
    This script does NOT anonymize and does NOT pseudonymize.
    It keeps raw extracted values.
    The anonymization step must run AFTER this chunking step and BEFORE indexing.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "clinical_chunk_raw_v2"
MIN_EMBEDDING_CHARS = 80
MAX_SECTION_EMBEDDING_CHARS = 1200
MAX_CHUNK_ID_LENGTH = 220

NON_INDEXABLE_BLOCK_TYPES = {
    "facility_block",
    "patient_info_block",
    "report_metadata_block",
}

LOW_VALUE_SECTION_TITLES = {
    "facility",
    "patient information",
    "report metadata",
}

TECHNICAL_BLOCK_LABELS = {
    "results_table_block",
    "patient_info_block",
    "validation_block",
    "facility_block",
    "document_header",
    "footer_block",
}

TECHNICAL_SUFFIX_PATTERNS = [
    re.compile(r":\s*Laboratory results\s+results_table_block\b", re.IGNORECASE),
    re.compile(r"\bLaboratory results\s+results_table_block\b", re.IGNORECASE),
    re.compile(r"\bresults_table_block\b", re.IGNORECASE),
    re.compile(r"\bpatient_info_block\b", re.IGNORECASE),
    re.compile(r"\bvalidation_block\b", re.IGNORECASE),
    re.compile(r"\bfacility_block\b", re.IGNORECASE),
    re.compile(r"\bdocument_header\b", re.IGNORECASE),
    re.compile(r"\bfooter_block\b", re.IGNORECASE),
    re.compile(r"\bLaboratory results\s+Laboratory results\b", re.IGNORECASE),
]

TRUNCATED_REFERENCE_ENDING_PATTERN = re.compile(
    r"(?:\b(de|du|des|d'|avec|sans|pour|chez|car|si|ou|et|à|a|le|la|les|un|une|pas de)\s*)$",
    re.IGNORECASE,
)

COMPLEX_REFERENCE_PATTERN = re.compile(
    r"\b(homme|femme|enfant|adulte|nouveau|nourrisson|cordon|ans|mois|jour|jours|risque|sexe|age|âge)\b",
    re.IGNORECASE,
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)

    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_accents(text: str) -> str:
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def slugify(text: Any) -> str:
    text = strip_accents(clean(text)).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def make_chunk_id(doc_id: str, chunk_type: str, *parts: Any) -> str:
    """
    Human-readable deterministic chunk id.
    No hashing is used.
    """
    values = [slugify(doc_id), slugify(chunk_type)]

    for part in parts:
        part_slug = slugify(part)
        if part_slug and part_slug != "unknown":
            values.append(part_slug)

    chunk_id = "chk_" + "_".join(values)
    return chunk_id[:MAX_CHUNK_ID_LENGTH]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(chunks: Iterable[Dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1

    return count


def write_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def iter_document_paths(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("document.json"))


def get_facility(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc.get("facility") or {}


def get_patient(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc.get("patient") or {}


def get_report(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc.get("report") or {}


def get_validation(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc.get("validation") or {}


def get_validation_report(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc.get("validation_report") or {}


def get_sample_type(doc: Dict[str, Any]) -> str:
    report = get_report(doc)
    return clean(report.get("sample_type") or report.get("sample_label"))


def get_request_or_report_date(doc: Dict[str, Any]) -> str:
    report = get_report(doc)
    return clean(
        report.get("report_date")
        or report.get("request_date")
        or report.get("received_date")
    )


def reference_range_text(reference_range: Any) -> str:
    if not reference_range:
        return ""

    if isinstance(reference_range, str):
        return clean(reference_range)

    if not isinstance(reference_range, dict):
        return clean(reference_range)

    text = clean(reference_range.get("text"))
    if text:
        return text

    low = reference_range.get("low")
    high = reference_range.get("high")
    unit = clean(reference_range.get("unit"))

    if low is not None and high is not None:
        value = f"{low} - {high}"
        if unit:
            value += f" {unit}"
        return value

    if low is not None:
        value = f">= {low}"
        if unit:
            value += f" {unit}"
        return value

    if high is not None:
        value = f"<= {high}"
        if unit:
            value += f" {unit}"
        return value

    return ""


def reference_quality_status(reference_range: Any) -> str:
    """
    Detect possibly incomplete reference ranges.
    This does not fix the reference. It only marks it as suspicious.
    """
    text = reference_range_text(reference_range)

    if not text:
        return "missing"

    stripped = text.strip()
    stripped_norm = strip_accents(stripped).lower()

    if stripped.count("(") > stripped.count(")"):
        return "possibly_truncated"

    if stripped.endswith((",", ";", ":", "-", "–", "(")):
        return "possibly_truncated"

    if TRUNCATED_REFERENCE_ENDING_PATTERN.search(stripped_norm):
        return "possibly_truncated"

    return "complete"


def reference_complexity(reference_range: Any) -> str:
    if not reference_range:
        return "missing"

    text = strip_accents(reference_range_text(reference_range)).lower()

    if not text:
        return "missing"

    if COMPLEX_REFERENCE_PATTERN.search(text):
        return "complex_clinical_context"

    if isinstance(reference_range, dict):
        if reference_range.get("low") is not None or reference_range.get("high") is not None:
            return "simple_numeric_range"

    return "raw_text_only"


def numeric_interpretation(value_numeric: Any, reference_range: Any) -> str:
    """
    Technical comparison only.
    This is not a medical diagnosis.
    """
    if value_numeric is None or not isinstance(reference_range, dict):
        return "unknown"

    if reference_quality_status(reference_range) == "possibly_truncated":
        return "needs_review_reference_truncated"

    complexity = reference_complexity(reference_range)

    if complexity == "complex_clinical_context":
        return "needs_clinical_context"

    low = reference_range.get("low")
    high = reference_range.get("high")

    try:
        value = float(value_numeric)
    except (TypeError, ValueError):
        return "unknown"

    if low is not None:
        try:
            if value < float(low):
                return "below_reference"
        except (TypeError, ValueError):
            pass

    if high is not None:
        try:
            if value > float(high):
                return "above_reference"
        except (TypeError, ValueError):
            pass

    if low is not None or high is not None:
        return "within_reference"

    if clean(reference_range.get("text")):
        return "needs_clinical_context"

    return "unknown"


def base_metadata(doc: Dict[str, Any], extraction_json_path: Path) -> Dict[str, Any]:
    facility = get_facility(doc)
    report = get_report(doc)
    patient = get_patient(doc)
    validation = get_validation(doc)
    validation_report = get_validation_report(doc)

    validation_status_value = clean(report.get("status") or validation.get("status"))
    if not validation_status_value:
        validation_status_value = "unknown"

    return {
        "schema_version": SCHEMA_VERSION,

        # Source
        "doc_id": clean(doc.get("doc_id")),
        "source_pdf": clean(doc.get("source_pdf")),
        "extraction_json": str(extraction_json_path),
        "document_type": clean(doc.get("document_type")),
        "pdf_type": clean(doc.get("pdf_type")),
        "page_count": doc.get("page_count"),

        # Facility
        "country": clean(facility.get("country")),
        "ministry": clean(facility.get("ministry")),
        "organization": clean(facility.get("organization")),
        "department": clean(facility.get("department")),
        "laboratory": clean(facility.get("laboratory")),
        "website": clean(facility.get("website")),
        "phone": clean(facility.get("phone")),
        "fax": clean(facility.get("fax")),

        # Patient raw data - intentionally kept for pre-anonymization output
        "patient_name": clean(patient.get("name") or patient.get("patient_name")),
        "patient_id": clean(patient.get("patient_id")),
        "patient_id_raw": clean(patient.get("patient_id_raw")),
        "ip_patient": clean(patient.get("ip_patient")),
        "patient_birth_date": clean(patient.get("birth_date")),
        "patient_birth_date_raw": clean(patient.get("birth_date_raw")),
        "patient_reported_age": patient.get("reported_age"),
        "patient_age": patient.get("age_final") or patient.get("age"),
        "patient_age_source": clean(patient.get("age_source_of_truth")),
        "patient_sex": clean(patient.get("sex")),
        "patient_sex_raw": clean(patient.get("sex_raw")),
        "patient_address": clean(patient.get("address")),
        "age_consistency_status": clean(patient.get("age_consistency_status")),

        # Report raw data
        "report_id": clean(report.get("report_id")),
        "report_id_raw": clean(report.get("report_id_raw")),
        "report_date": clean(report.get("report_date")),
        "report_date_raw": clean(report.get("report_date_raw")),
        "request_date": clean(report.get("request_date")),
        "request_date_raw": clean(report.get("request_date_raw")),
        "received_date": clean(report.get("received_date")),
        "received_date_raw": clean(report.get("received_date_raw")),
        "sample_id": clean(report.get("sample_id")),
        "sample_id_raw": clean(report.get("sample_id_raw")),
        "sample_type": clean(report.get("sample_type")),
        "sample_label": clean(report.get("sample_label")),
        "encounter_type": clean(report.get("encounter_type")),
        "encounter_type_raw": clean(report.get("encounter_type_raw")),
        "origin": clean(report.get("origin")),
        "service": clean(report.get("service")),
        "prescriber": clean(report.get("prescriber")),
        "specialty": clean(report.get("specialty")),
        "status": clean(report.get("status")),

        # Validation raw data
        "validation_status": validation_status_value,
        "validation_title": clean(validation.get("validation_title")),
        "validated_by": clean(report.get("validated_by") or validation.get("validated_by")),
        "validation_date": clean(report.get("validation_date") or validation.get("validation_date")),
        "edited_by": clean(report.get("edited_by") or validation.get("edited_by")),
        "edited_date": clean(report.get("edited_date") or validation.get("edited_date")),
        "printed_by": clean(report.get("printed_by") or validation.get("printed_by")),
        "print_date": clean(report.get("print_date") or validation.get("print_date")),
        "is_signed": validation.get("is_signed"),
        "is_stamped": validation.get("is_stamped"),

        # Quality
        "result_quality_status": clean(validation_report.get("result_quality_status")),
    }


def build_keyword_text(text: str, extra_terms: Iterable[Any]) -> str:
    """
    Keyword text is intentionally richer and more redundant than embedding text.
    It helps BM25/keyword retrieval with exact terms, units, acronyms, and accentless variants.
    """
    terms = [text]

    for term in extra_terms:
        term = clean(term)
        if term:
            terms.append(term)

    joined = clean(" ".join(terms))
    accentless = strip_accents(joined)

    return clean(f"{joined} {accentless}")


def clean_index_text(text: str) -> str:
    """
    Remove technical chunking/block artifacts from searchable text while preserving clinical content.
    """
    cleaned = clean(text)
    if not cleaned:
        return ""

    for pattern in TECHNICAL_SUFFIX_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)

    cleaned = re.sub(r"\s+([:;,\.\)])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def enrich_short_embedding_text(text: str, metadata: Dict[str, Any]) -> str:
    """
    Enrich short chunks for vector search.
    Keep keyword text unchanged.
    """
    text = clean(text)

    if len(text) >= MIN_EMBEDDING_CHARS:
        return text

    context_parts = []

    document_type = clean(metadata.get("document_type"))
    laboratory = clean(metadata.get("laboratory"))
    section = clean(metadata.get("section"))
    sample_type = clean(metadata.get("sample_type"))
    patient_sex = clean(metadata.get("patient_sex"))
    patient_age = metadata.get("patient_age")

    if document_type:
        context_parts.append(f"Type de document: {document_type}")

    if laboratory:
        context_parts.append(f"Laboratoire: {laboratory}")

    if section:
        context_parts.append(f"Section: {section}")

    if sample_type:
        context_parts.append(f"Type de prélèvement: {sample_type}")

    if patient_sex:
        context_parts.append(f"Sexe: {patient_sex}")

    if patient_age is not None and clean(patient_age):
        context_parts.append(f"Âge calculé: {patient_age} ans")

    context = ". ".join(context_parts)

    if context:
        return clean(f"Rapport médical. {context}. Contenu: {text}")

    return text


def split_long_section_text(text: str, max_chars: int = MAX_SECTION_EMBEDDING_CHARS) -> List[str]:
    """
    Deterministic soft splitter for long section text.
    Split primarily on sentence-like punctuation, fallback to hard cut.
    """
    text = clean(text)
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    remaining = text

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        split_idx = max(
            window.rfind(". "),
            window.rfind("; "),
            window.rfind(", "),
            window.rfind(" "),
        )

        if split_idx < int(max_chars * 0.6):
            split_idx = max_chars

        part = clean(remaining[:split_idx])
        if part:
            parts.append(part)
        remaining = clean(remaining[split_idx:])

    if remaining:
        parts.append(remaining)

    return parts


def section_key_for_result(result: Dict[str, Any], doc: Dict[str, Any]) -> Tuple[str, str]:
    section_name = clean(
        result.get("section_name")
        or result.get("section")
        or result.get("source_table_id")
    )

    if not section_name:
        doc_type = clean(doc.get("document_type"))
        if doc_type == "parasitology_stool_report":
            section_name = "Résultats parasitologie"
        else:
            section_name = "Résultats biologiques"

    return slugify(section_name), section_name


def make_chunk(
    *,
    doc_id: str,
    chunk_type: str,
    text_for_embedding: str,
    text_for_keyword: str,
    metadata: Dict[str, Any],
    provenance: Dict[str, Any],
    quality: Dict[str, Any],
    routing: Dict[str, Any],
    parent_chunk_id: Optional[str] = None,
    chunk_id_parts: Iterable[Any] = (),
    modality: str = "text",
) -> Dict[str, Any]:
    clean_embedding = clean_index_text(text_for_embedding)
    clean_keyword = clean_index_text(text_for_keyword)
    chunk_id = make_chunk_id(doc_id, chunk_type, *chunk_id_parts)

    normalized_quality = dict(quality or {})
    if normalized_quality.get("confidence_score") is None:
        confidence_label = clean(normalized_quality.get("confidence")).lower()
        score_map = {
            "high": 0.9,
            "medium": 0.7,
            "low": 0.5,
        }
        normalized_quality["confidence_score"] = score_map.get(confidence_label, 0.6)

    chunk = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "parent_chunk_id": parent_chunk_id,
        "doc_id": doc_id,
        "chunk_type": chunk_type,
        "modality": modality,
        "text_for_embedding": clean_embedding,
        "text_for_keyword": clean_keyword,
        "metadata": metadata,
        "provenance": provenance,
        "quality": normalized_quality,
        "routing": routing,
    }

    # Stable hash used by downstream hybrid retrieval/index pipelines.
    # It supports deterministic upserts and quick diffing across runs.
    hash_payload = {
        "doc_id": chunk["doc_id"],
        "chunk_type": chunk["chunk_type"],
        "modality": chunk["modality"],
        "text_for_embedding": chunk["text_for_embedding"],
        "text_for_keyword": chunk["text_for_keyword"],
        "metadata": chunk["metadata"],
        "provenance": chunk["provenance"],
        "routing": chunk["routing"],
    }
    chunk["content_hash"] = hashlib.md5(
        json.dumps(hash_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    return chunk


def build_document_summary_chunk(
    doc: Dict[str, Any],
    extraction_json_path: Path,
) -> Dict[str, Any]:
    doc_id = clean(doc.get("doc_id"))
    base = base_metadata(doc, extraction_json_path)
    report = get_report(doc)
    facility = get_facility(doc)
    results = doc.get("results") or []

    labels: List[str] = []

    for result in results:
        label = clean(result.get("analyte") or result.get("parameter"))
        if label:
            labels.append(label)

    unique_labels = []
    seen = set()

    for label in labels:
        key = slugify(label)
        if key not in seen:
            seen.add(key)
            unique_labels.append(label)

    label_preview = ", ".join(unique_labels[:40])

    if len(unique_labels) > 40:
        label_preview += f", ... (+{len(unique_labels) - 40} autres)"

    text = (
        f"Résumé du rapport médical. "
        f"Type de document: {clean(doc.get('document_type'))}. "
        f"Laboratoire: {clean(facility.get('laboratory'))}. "
        f"Type de prélèvement: {clean(report.get('sample_type'))}. "
        f"Date du rapport ou de la demande: {get_request_or_report_date(doc)}. "
        f"Nombre de résultats extraits: {len(results)}. "
        f"Paramètres présents: {label_preview}."
    )

    metadata = {
        **base,
        "chunk_level": "document",
        "result_count": len(results),
        "analyte_count": len(unique_labels),
        "analyte_names": unique_labels[:100],
    }

    return make_chunk(
        doc_id=doc_id,
        chunk_type="document_summary",
        text_for_embedding=text,
        text_for_keyword=build_keyword_text(text, unique_labels),
        metadata=metadata,
        provenance={
            "doc_id": doc_id,
            "source_pdf": clean(doc.get("source_pdf")),
            "extraction_json": str(extraction_json_path),
        },
        quality={
            "confidence": "medium",
            "confidence_score": None,
        },
        routing={
            "vector_index": True,
            "keyword_index": True,
            "metadata_index": True,
            "priority": "medium",
        },
        parent_chunk_id=None,
        chunk_id_parts=["document_summary"],
    )


def build_result_chunk(
    doc: Dict[str, Any],
    result: Dict[str, Any],
    extraction_json_path: Path,
) -> Dict[str, Any]:
    doc_id = clean(doc.get("doc_id"))
    base = base_metadata(doc, extraction_json_path)
    report = get_report(doc)

    analyte = clean(result.get("analyte"))
    parameter = clean(result.get("parameter"))
    result_label = analyte or parameter or "Résultat"

    value_raw = clean(result.get("value_raw") or result.get("result"))
    value_numeric = result.get("value_numeric")

    unit = clean(result.get("unit"))
    if unit.lower() == "unknown":
        unit = ""

    result_kind = clean(result.get("result_kind"))
    is_numeric_without_unit = result_kind == "numeric" and not unit
    if is_numeric_without_unit:
        unit = "non spécifiée"
    section_key, section_name = section_key_for_result(result, doc)

    reference = reference_range_text(result.get("reference_range"))
    reference_complexity_status = reference_complexity(result.get("reference_range"))
    reference_quality = reference_quality_status(result.get("reference_range"))

    observation_date = clean(
        result.get("observation_date")
        or result.get("observation_date_raw")
        or report.get("report_date")
        or report.get("request_date")
    )

    sample_type = get_sample_type(doc)
    page_number = result.get("page_number") or result.get("source_page_number")

    patient_age = base.get("patient_age")
    patient_sex = base.get("patient_sex")

    if analyte:
        text = f"Résultat de laboratoire: {result_label}"
        if value_raw:
            text += f" = {value_raw}"
            if unit:
                text += f" {unit}"
        text += "."
    else:
        text = f"{section_name}: {result_label}"
        if value_raw:
            text += f" = {value_raw}"
        text += "."

    if reference:
        if reference_quality == "possibly_truncated":
            text += f" Référence brute possiblement incomplète: {reference}."
        else:
            text += f" Valeurs de référence brutes: {reference}."

    if patient_sex:
        text += f" Sexe: {patient_sex}."

    if patient_age is not None and clean(patient_age):
        text += f" Âge calculé: {patient_age} ans."

    if sample_type:
        text += f" Type de prélèvement: {sample_type}."

    if observation_date:
        text += f" Date d'observation: {observation_date}."

    if section_name:
        text += f" Section: {section_name}."

    confidence = clean(result.get("confidence"))
    confidence_score = result.get("confidence_score")

    if reference_quality == "possibly_truncated":
        interpretation_status = "needs_review_reference_truncated"
    else:
        interpretation_status = numeric_interpretation(
            value_numeric=value_numeric,
            reference_range=result.get("reference_range"),
        )

    dedup_label = slugify(analyte or parameter or result_label) or "unknown"
    dedup_value = clean(value_raw) or "na"
    dedup_unit = clean(unit) or "na"

    metadata = {
        **base,
        "chunk_level": "atomic",
        "section": section_name,
        "section_norm": section_key,
        "analyte": analyte,
        "analyte_norm": slugify(analyte) if analyte else "",
        "parameter": parameter,
        "parameter_norm": slugify(parameter) if parameter else "",
        "value_raw": value_raw,
        "value_numeric": value_numeric,
        "unit": unit,
        "reference_range": reference,
        "reference_complexity": reference_complexity_status,
        "reference_quality_status": reference_quality,
        "result_kind": result_kind,
        "interpretation_status": interpretation_status,
        "sample_type": sample_type,
        "observation_date": observation_date,
        "page_number": page_number,
        "row_index": result.get("row_index"),
        "dedup_key": clean(
            f"{dedup_label}|{dedup_value}|{dedup_unit}|{doc_id}"
        ),
        "is_canonical": result.get("is_canonical"),
    }

    quality_flags: List[str] = []
    if is_numeric_without_unit:
        metadata["result_quality_status"] = "unit_missing"
        quality_flags.append("numeric_without_unit")
    elif not clean(metadata.get("result_quality_status")):
        metadata["result_quality_status"] = "passed"

    provenance = {
        "doc_id": doc_id,
        "source_pdf": clean(doc.get("source_pdf")),
        "extraction_json": str(extraction_json_path),
        "page_number": page_number,
        "source_page_number": result.get("source_page_number"),
        "source_table_id": clean(result.get("source_table_id")),
        "source_kind": clean(result.get("source_kind")),
        "row_index": result.get("row_index"),
        "source_line_start": result.get("source_line_start"),
        "source_line_end": result.get("source_line_end"),
    }

    parent_chunk_id = make_chunk_id(doc_id, "exam_section", section_key, "part_01")

    keyword_terms = [
        result_label,
        analyte,
        parameter,
        value_raw,
        unit,
        reference,
        reference_quality,
        result_kind,
        section_name,
        sample_type,
        observation_date,
        patient_sex,
        patient_age,
        interpretation_status,
    ]

    return make_chunk(
        doc_id=doc_id,
        chunk_type="lab_result" if analyte else "clinical_result",
        text_for_embedding=text,
        text_for_keyword=build_keyword_text(text, keyword_terms),
        metadata=metadata,
        provenance=provenance,
        quality={
            "confidence": confidence,
            "confidence_score": confidence_score,
            "ocr_correction_applied": result.get("ocr_correction_applied"),
            "reference_quality_status": reference_quality,
            "quality_flags": quality_flags,
        },
        routing={
            "vector_index": True,
            "keyword_index": True,
            "metadata_index": True,
            "priority": "high",
        },
        parent_chunk_id=parent_chunk_id,
        chunk_id_parts=[
            "result",
            section_key,
            result.get("row_index"),
            result.get("dedup_key"),
            result_label,
            value_raw,
            unit,
        ],
    )


def build_result_chunks(
    doc: Dict[str, Any],
    extraction_json_path: Path,
) -> List[Dict[str, Any]]:
    chunks = []

    for result in doc.get("results") or []:
        chunks.append(build_result_chunk(doc, result, extraction_json_path))

    return chunks


def build_section_chunks_from_blocks(
    doc: Dict[str, Any],
    extraction_json_path: Path,
) -> List[Dict[str, Any]]:
    doc_id = clean(doc.get("doc_id"))
    base = base_metadata(doc, extraction_json_path)
    chunks: List[Dict[str, Any]] = []

    for page in doc.get("pages") or []:
        page_number = page.get("page_number")

        for block in page.get("blocks") or []:
            block_type = clean(block.get("block_type"))
            section_title = clean(block.get("section_title") or block_type)
            section_key = slugify(section_title or block_type)

            if block_type in NON_INDEXABLE_BLOCK_TYPES:
                continue

            if section_title.lower() in LOW_VALUE_SECTION_TITLES:
                continue

            if block.get("is_indexable") is False:
                continue

            raw_text = clean(
                block.get("index_text")
                or block.get("normalized_text")
                or block.get("text")
            )

            if not raw_text:
                continue

            metadata = {
                **base,
                "chunk_level": "section",
                "section": section_title,
                "section_norm": section_key,
                "block_id": clean(block.get("block_id")),
                "block_type": block_type,
                "page_number": page_number,
                "source_table_ids": block.get("source_table_ids") or [],
                "source_image_ids": block.get("source_image_ids") or [],
            }

            split_parts = split_long_section_text(raw_text)
            total_parts = len(split_parts)

            for idx, part_text in enumerate(split_parts, start=1):
                part_suffix = f"part_{idx:02d}" if total_parts > 1 else "part_01"
                part_metadata = {
                    **metadata,
                    "section_part_index": idx,
                    "section_part_total": total_parts,
                }
                part_embedding = enrich_short_embedding_text(part_text, part_metadata)

                chunk = make_chunk(
                    doc_id=doc_id,
                    chunk_type="exam_section",
                    text_for_embedding=part_embedding,
                    text_for_keyword=build_keyword_text(part_text, [section_title]),
                    metadata=part_metadata,
                    provenance={
                        "doc_id": doc_id,
                        "source_pdf": clean(doc.get("source_pdf")),
                        "extraction_json": str(extraction_json_path),
                        "page_number": page_number,
                        "block_id": clean(block.get("block_id")),
                        "source_table_ids": block.get("source_table_ids") or [],
                        "source_image_ids": block.get("source_image_ids") or [],
                        "section_part_index": idx,
                        "section_part_total": total_parts,
                    },
                    quality={
                        "confidence": clean(block.get("confidence")),
                        "confidence_score": block.get("confidence_score"),
                        "embedding_enriched": len(part_text) < MIN_EMBEDDING_CHARS,
                    },
                    routing={
                        "vector_index": True,
                        "keyword_index": True,
                        "metadata_index": True,
                        "priority": "medium",
                    },
                    parent_chunk_id=make_chunk_id(doc_id, "document_summary", "document_summary"),
                    chunk_id_parts=[section_key, part_suffix],
                )

                chunks.append(chunk)

    return deduplicate_chunks(chunks)


def build_section_chunks_from_results(
    doc: Dict[str, Any],
    extraction_json_path: Path,
) -> List[Dict[str, Any]]:
    doc_id = clean(doc.get("doc_id"))
    base = base_metadata(doc, extraction_json_path)

    grouped: Dict[str, Dict[str, Any]] = {}

    for result in doc.get("results") or []:
        section_key, section_name = section_key_for_result(result, doc)

        if section_key not in grouped:
            grouped[section_key] = {
                "section_name": section_name,
                "results": [],
            }

        grouped[section_key]["results"].append(result)

    chunks: List[Dict[str, Any]] = []

    for section_key, payload in grouped.items():
        section_name = payload["section_name"]
        results = payload["results"]

        labels = []
        value_fragments = []

        for result in results:
            label = clean(result.get("analyte") or result.get("parameter"))
            value = clean(result.get("value_raw") or result.get("result"))
            unit = clean(result.get("unit"))

            if unit.lower() == "unknown":
                unit = ""

            if label:
                labels.append(label)

            if label and value:
                fragment = f"{label}: {value}"
                if unit:
                    fragment += f" {unit}"
                value_fragments.append(fragment)

        raw_text = (
            f"Section médicale: {section_name}. "
            f"Nombre de résultats: {len(results)}. "
            f"Résultats principaux: {', '.join(value_fragments[:60])}."
        )

        metadata = {
            **base,
            "chunk_level": "section",
            "section": section_name,
            "section_norm": section_key,
            "result_count": len(results),
            "analyte_names": labels[:100],
        }

        split_parts = split_long_section_text(raw_text)
        total_parts = len(split_parts)

        for idx, part_text in enumerate(split_parts, start=1):
            part_suffix = f"part_{idx:02d}" if total_parts > 1 else "part_01"
            part_metadata = {
                **metadata,
                "section_part_index": idx,
                "section_part_total": total_parts,
            }
            part_embedding = enrich_short_embedding_text(part_text, part_metadata)

            chunk = make_chunk(
                doc_id=doc_id,
                chunk_type="exam_section",
                text_for_embedding=part_embedding,
                text_for_keyword=build_keyword_text(part_text, labels),
                metadata=part_metadata,
                provenance={
                    "doc_id": doc_id,
                    "source_pdf": clean(doc.get("source_pdf")),
                    "extraction_json": str(extraction_json_path),
                    "section": section_name,
                    "section_part_index": idx,
                    "section_part_total": total_parts,
                },
                quality={
                    "confidence": "medium",
                    "confidence_score": None,
                    "embedding_enriched": len(part_text) < MIN_EMBEDDING_CHARS,
                },
                routing={
                    "vector_index": True,
                    "keyword_index": True,
                    "metadata_index": True,
                    "priority": "medium",
                },
                parent_chunk_id=make_chunk_id(doc_id, "document_summary", "document_summary"),
                chunk_id_parts=[section_key, part_suffix],
            )

            chunks.append(chunk)

    return chunks


def build_section_chunks(
    doc: Dict[str, Any],
    extraction_json_path: Path,
) -> List[Dict[str, Any]]:
    block_chunks = build_section_chunks_from_blocks(doc, extraction_json_path)

    if block_chunks:
        existing_keys = {chunk["metadata"].get("section_norm") for chunk in block_chunks}

        fallback_chunks = [
            chunk
            for chunk in build_section_chunks_from_results(doc, extraction_json_path)
            if chunk["metadata"].get("section_norm") not in existing_keys
        ]

        return deduplicate_chunks(block_chunks + fallback_chunks)

    return build_section_chunks_from_results(doc, extraction_json_path)


def build_validation_chunk(
    doc: Dict[str, Any],
    extraction_json_path: Path,
) -> Optional[Dict[str, Any]]:
    doc_id = clean(doc.get("doc_id"))
    base = base_metadata(doc, extraction_json_path)
    report = get_report(doc)
    validation = get_validation(doc)

    status = clean(report.get("status") or validation.get("status"))
    if not status:
        status = "unknown"
    validation_title = clean(validation.get("validation_title"))
    validation_date = clean(report.get("validation_date") or validation.get("validation_date"))
    edited_date = clean(report.get("edited_date") or validation.get("edited_date"))

    if not any([status, validation_title, validation_date, edited_date]):
        return None

    text = (
        f"Statut administratif du rapport. "
        f"Validation: {status or 'non précisée'}. "
        f"Titre: {validation_title or 'non précisé'}. "
        f"Date de validation: {validation_date or 'non précisée'}. "
        f"Date d'édition: {edited_date or 'non précisée'}."
    )

    metadata = {
        **base,
        "chunk_level": "document",
        "validation_status": status,
        "validation_title": validation_title,
        "validation_date": validation_date,
        "edited_date": edited_date,
        "is_signed": validation.get("is_signed"),
        "is_stamped": validation.get("is_stamped"),
    }

    return make_chunk(
        doc_id=doc_id,
        chunk_type="validation_status",
        text_for_embedding=text,
        text_for_keyword=build_keyword_text(
            text,
            [status, validation_title, validation_date, edited_date],
        ),
        metadata=metadata,
        provenance={
            "doc_id": doc_id,
            "source_pdf": clean(doc.get("source_pdf")),
            "extraction_json": str(extraction_json_path),
        },
        quality={
            "confidence": clean(validation.get("confidence")),
            "confidence_score": validation.get("confidence_score"),
        },
        routing={
            "vector_index": False,
            "keyword_index": True,
            "metadata_index": True,
            "priority": "low",
        },
        parent_chunk_id=make_chunk_id(doc_id, "document_summary", "document_summary"),
        chunk_id_parts=["validation_status"],
    )


def build_visual_reference_chunks(
    doc: Dict[str, Any],
    extraction_json_path: Path,
) -> List[Dict[str, Any]]:
    doc_id = clean(doc.get("doc_id"))
    base = base_metadata(doc, extraction_json_path)
    chunks: List[Dict[str, Any]] = []

    for page in doc.get("pages") or []:
        page_number = page.get("page_number")
        image_ids = page.get("image_ids") or []
        table_ids = page.get("table_ids") or []
        ocr_visual_ids = page.get("ocr_visual_ids") or []

        if not image_ids and not table_ids and not ocr_visual_ids:
            continue

        text = (
            f"Référence visuelle du rapport médical, page {page_number}. "
            f"Images associées: {', '.join(image_ids) if image_ids else 'aucune'}. "
            f"Tables associées: {', '.join(table_ids) if table_ids else 'aucune'}. "
            f"Objets OCR visuels: {', '.join(ocr_visual_ids) if ocr_visual_ids else 'aucun'}."
        )

        metadata = {
            **base,
            "chunk_level": "visual",
            "page_number": page_number,
            "image_ids": image_ids,
            "table_ids": table_ids,
            "ocr_visual_ids": ocr_visual_ids,
        }

        chunk = make_chunk(
            doc_id=doc_id,
            chunk_type="visual_reference",
            modality="visual_reference",
            text_for_embedding=text,
            text_for_keyword=build_keyword_text(text, image_ids + table_ids + ocr_visual_ids),
            metadata=metadata,
            provenance={
                "doc_id": doc_id,
                "source_pdf": clean(doc.get("source_pdf")),
                "extraction_json": str(extraction_json_path),
                "page_number": page_number,
                "image_ids": image_ids,
                "table_ids": table_ids,
                "ocr_visual_ids": ocr_visual_ids,
            },
            quality={
                "confidence": "medium",
                "confidence_score": None,
            },
            routing={
                "vector_index": False,
                "keyword_index": True,
                "metadata_index": True,
                "object_store_reference": True,
                "priority": "low",
            },
            parent_chunk_id=make_chunk_id(doc_id, "document_summary", "document_summary"),
            chunk_id_parts=["visual", page_number, ",".join(image_ids), ",".join(table_ids)],
        )

        chunks.append(chunk)

    return chunks


def deduplicate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if chunk_id in seen:
            continue

        seen.add(chunk_id)
        unique.append(chunk)

    return unique


def build_chunks_for_document(
    doc: Dict[str, Any],
    extraction_json_path: Path,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    chunks.append(build_document_summary_chunk(doc, extraction_json_path))
    chunks.extend(build_section_chunks(doc, extraction_json_path))
    chunks.extend(build_result_chunks(doc, extraction_json_path))

    validation_chunk = build_validation_chunk(doc, extraction_json_path)
    if validation_chunk:
        chunks.append(validation_chunk)

    chunks.extend(build_visual_reference_chunks(doc, extraction_json_path))

    return deduplicate_chunks(chunks)


def validate_chunk(chunk: Dict[str, Any]) -> List[str]:
    issues = []

    required = [
        "chunk_id",
        "doc_id",
        "chunk_type",
        "text_for_embedding",
        "metadata",
        "provenance",
        "routing",
    ]

    for field in required:
        if field not in chunk:
            issues.append(f"missing_field:{field}")

    if not clean(chunk.get("text_for_embedding")):
        issues.append("empty_text_for_embedding")

    if not clean(chunk.get("doc_id")):
        issues.append("empty_doc_id")

    chunk_type = clean(chunk.get("chunk_type"))
    modality = clean(chunk.get("modality"))
    if modality not in {"text", "visual_reference"}:
        issues.append(f"unsupported_modality:{modality or 'empty'}")

    if chunk_type == "visual_reference" and modality != "visual_reference":
        issues.append("visual_reference_chunk_with_non_visual_modality")

    if chunk_type != "visual_reference" and modality == "visual_reference":
        issues.append("non_visual_chunk_with_visual_modality")

    routing = chunk.get("routing") or {}

    if routing.get("vector_index") is True:
        text_len = len(clean(chunk.get("text_for_embedding")))
        if text_len < MIN_EMBEDDING_CHARS:
            issues.append(f"short_vector_text:{text_len}")

    metadata = chunk.get("metadata") or {}

    if chunk.get("chunk_type") == "lab_result":
        if not clean(metadata.get("analyte")) and not clean(metadata.get("parameter")):
            issues.append("lab_result_missing_analyte_or_parameter")

        if not clean(metadata.get("value_raw")):
            issues.append("lab_result_missing_value_raw")

    if chunk_type == "visual_reference":
        provenance = chunk.get("provenance") or {}
        image_ids = provenance.get("image_ids") or []
        table_ids = provenance.get("table_ids") or []
        ocr_visual_ids = provenance.get("ocr_visual_ids") or []
        if not image_ids and not table_ids and not ocr_visual_ids:
            issues.append("visual_reference_missing_image_table_ocr_ids")

    if routing.get("vector_index") is not True and chunk_type in {
        "document_summary",
        "exam_section",
        "lab_result",
        "clinical_result",
    }:
        allow_vector_off = (
            chunk_type in {"lab_result", "clinical_result"}
            and clean(metadata.get("result_quality_status")) == "unit_missing"
        )
        if not allow_vector_off:
            issues.append("unexpected_vector_index_false_for_retrieval_chunk")

    if routing.get("vector_index") is True and chunk_type in {
        "validation_status",
        "visual_reference",
    }:
        issues.append("unexpected_vector_index_true_for_non_vector_chunk")

    return issues


def validate_document_coverage(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Minimal production safety gate:
    each document should have one summary and at least one retrieval chunk.
    """
    issues: List[Dict[str, Any]] = []
    by_doc: Dict[str, List[Dict[str, Any]]] = {}

    for chunk in chunks:
        by_doc.setdefault(clean(chunk.get("doc_id")), []).append(chunk)

    for doc_id, doc_chunks in by_doc.items():
        chunk_types = Counter(clean(c.get("chunk_type")) for c in doc_chunks)
        retrieval_count = sum(
            1
            for c in doc_chunks
            if (c.get("routing") or {}).get("vector_index") is True
            and clean(c.get("chunk_type")) in {"document_summary", "exam_section", "lab_result", "clinical_result"}
        )

        doc_issues = []
        if chunk_types.get("document_summary", 0) != 1:
            doc_issues.append(f"document_summary_count:{chunk_types.get('document_summary', 0)}")
        if retrieval_count == 0:
            doc_issues.append("no_vector_retrieval_chunk")

        if doc_issues:
            issues.append(
                {
                    "chunk_id": "",
                    "doc_id": doc_id,
                    "issues": doc_issues,
                }
            )

    return issues


def validate_parent_links(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues = []
    chunk_ids = {chunk.get("chunk_id") for chunk in chunks}

    for chunk in chunks:
        parent_id = chunk.get("parent_chunk_id")
        if parent_id and parent_id not in chunk_ids:
            issues.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "issues": [f"missing_parent_chunk:{parent_id}"],
                }
            )

    return issues


def validate_duplicate_ids(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counter = Counter(chunk.get("chunk_id") for chunk in chunks)
    issues = []

    for chunk_id, count in counter.items():
        if count > 1:
            issues.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": "",
                    "issues": [f"duplicate_chunk_id:{count}"],
                }
            )

    return issues


def build_all_chunks(
    input_root: Path,
    fail_on_error: bool = False,
    strict_quality_gates: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    document_paths = iter_document_paths(input_root)

    all_chunks: List[Dict[str, Any]] = []
    chunks_by_document_path: Dict[str, List[Dict[str, Any]]] = {}
    errors: List[Dict[str, Any]] = []
    validation_issues: List[Dict[str, Any]] = []

    stats = {
        "documents_found": len(document_paths),
        "documents_processed": 0,
        "chunks_total": 0,
        "chunks_by_type": Counter(),
        "chunks_by_document_type": Counter(),
        "results_by_document": {},
        "reference_quality_status": Counter(),
        "interpretation_status": Counter(),
        "embedding_enriched_chunks": 0,
    }

    for path in document_paths:
        try:
            doc = load_json(path)
            doc_chunks = build_chunks_for_document(doc, path)

            for chunk in doc_chunks:
                issues = validate_chunk(chunk)
                if issues:
                    validation_issues.append(
                        {
                            "chunk_id": chunk.get("chunk_id"),
                            "doc_id": chunk.get("doc_id"),
                            "issues": issues,
                        }
                    )

            all_chunks.extend(doc_chunks)
            chunks_by_document_path[str(path)] = doc_chunks

            doc_id = clean(doc.get("doc_id")) or str(path.parent.name)
            doc_type = clean(doc.get("document_type")) or "unknown"

            stats["documents_processed"] += 1
            stats["results_by_document"][doc_id] = len(doc.get("results") or [])

            for chunk in doc_chunks:
                stats["chunks_by_type"][chunk["chunk_type"]] += 1
                stats["chunks_by_document_type"][doc_type] += 1

                quality = chunk.get("quality") or {}
                metadata = chunk.get("metadata") or {}

                if quality.get("embedding_enriched") is True:
                    stats["embedding_enriched_chunks"] += 1

                ref_status = metadata.get("reference_quality_status")
                if ref_status:
                    stats["reference_quality_status"][ref_status] += 1

                interp_status = metadata.get("interpretation_status")
                if interp_status:
                    stats["interpretation_status"][interp_status] += 1

        except Exception as exc:
            error = {
                "path": str(path),
                "error": repr(exc),
            }
            errors.append(error)

            if fail_on_error:
                raise

    validation_issues.extend(validate_parent_links(all_chunks))
    validation_issues.extend(validate_duplicate_ids(all_chunks))
    validation_issues.extend(validate_document_coverage(all_chunks))

    stats["chunks_total"] = len(all_chunks)
    stats["chunks_by_type"] = dict(stats["chunks_by_type"])
    stats["chunks_by_document_type"] = dict(stats["chunks_by_document_type"])
    stats["reference_quality_status"] = dict(stats["reference_quality_status"])
    stats["interpretation_status"] = dict(stats["interpretation_status"])

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "stats": stats,
        "validation_issues": validation_issues,
        "errors": errors,
        "notes": [
            "JSONL is the canonical raw chunk artifact.",
            "No anonymization is applied in this script.",
            "No pseudonymization is applied in this script.",
            "Raw patient/report/sample/personnel fields are kept in metadata.",
            "Run anonymization after this step and before indexing.",
            "Vector DB / keyword DB / metadata DB should consume the anonymized chunks, not the raw chunks.",
            "Short vectorized section chunks are enriched with contextual metadata.",
            "Possibly truncated reference ranges are flagged and interpretation is set to needs_review_reference_truncated.",
        ],
    }

    if strict_quality_gates and (errors or validation_issues):
        raise RuntimeError(
            f"Strict quality gate failed: errors={len(errors)}, validation_issues={len(validation_issues)}"
        )

    return all_chunks, report, chunks_by_document_path


def write_local_chunks(chunks_by_document_path: Dict[str, List[Dict[str, Any]]]) -> int:
    written = 0

    for document_json_path, chunks in chunks_by_document_path.items():
        output_path = Path(document_json_path).parent / "chunks.jsonl"
        write_jsonl(chunks, output_path)
        written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw clinical RAG chunks from extracted medical report JSON files."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/extraction"),
        help="Root directory containing */document.json files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chunks/chunks.raw.jsonl"),
        help="Global raw output JSONL path.",
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/chunks/chunking_report.json"),
        help="Output chunking report path.",
    )

    parser.add_argument(
        "--write-local-chunks",
        action="store_true",
        help="Also write data/extraction/<report_dir>/chunks.jsonl for each document.",
    )

    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Stop immediately if one document fails.",
    )

    parser.add_argument(
        "--strict-quality-gates",
        action="store_true",
        help="Fail if any validation issue is detected (recommended for CI production pipeline).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks, report, chunks_by_document_path = build_all_chunks(
        input_root=args.input_root,
        fail_on_error=args.fail_on_error,
        strict_quality_gates=args.strict_quality_gates,
    )

    count = write_jsonl(chunks, args.output)
    write_json(report, args.report)

    print(f"OK - documents processed: {report['stats']['documents_processed']}")
    print(f"OK - chunks written: {count}")
    print(f"OK - output: {args.output}")
    print(f"OK - report: {args.report}")

    if args.write_local_chunks:
        local_count = write_local_chunks(chunks_by_document_path)
        print(f"OK - local chunk files written: {local_count}")

    if report["errors"]:
        print(f"WARNING - document errors: {len(report['errors'])}")

    if report["validation_issues"]:
        print(f"WARNING - validation issues: {len(report['validation_issues'])}")


if __name__ == "__main__":
    main()
