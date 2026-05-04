from __future__ import annotations

import re
import unicodedata
from typing import Any

try:
    from retrieval.models import RetrievalResult, SearchResponse
except Exception:
    from scripts.retrieval.models import RetrievalResult, SearchResponse
from query_understanding import contains_exact_term


_ADMIN_BLOCKED_TYPES = {"validation_status", "visual_reference"}
_EXPLICIT_ADMIN_HINTS = {
    "validation",
    "valide",
    "validité",
    "qualite",
    "qualité",
    "coherence",
    "cohérence",
    "visuel",
    "image",
    "figure",
    "source visuelle",
}


def _norm(text: str) -> str:
    value = (text or "").strip().lower().replace("µ", "u")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"\s+", " ", value)
    return value


def _explicit_admin_intent(query: str) -> bool:
    qn = _norm(query)
    return any(h in qn for h in _EXPLICIT_ADMIN_HINTS)


def _chunk_priority(chunk_type: str) -> int:
    c = (chunk_type or "").lower()
    if c == "lab_result":
        return 0
    if c == "clinical_result":
        return 1
    if c == "exam_section":
        return 2
    if c == "document_summary":
        return 3
    if c == "validation_status":
        return 4
    if c == "visual_reference":
        return 5
    return 6


def _text_excerpt(text: str, max_len: int = 560) -> str:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _int_flag(value: Any) -> int:
    try:
        return 1 if int(value or 0) == 1 else 0
    except Exception:
        return 0


def _dedupe_key(item: RetrievalResult) -> tuple[str, str, str]:
    md = item.metadata or {}
    analyte = str(md.get("analyte_norm") or md.get("analyte") or "").strip().lower()
    value = str(md.get("value_raw") or md.get("value_numeric") or "").strip().lower()
    return (
        str(item.doc_id or "").strip().lower(),
        str(item.chunk_type or "").strip().lower(),
        f"{analyte}|{value}",
    )


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _row_to_retrieval_result(row: dict[str, Any]) -> RetrievalResult:
    text = str(row.get("text_for_embedding") or row.get("text_for_keyword") or "").strip()
    text_preview = text if len(text) <= 260 else text[:257] + "..."
    md = dict(row)
    return RetrievalResult(
        chunk_id=str(row.get("chunk_id") or ""),
        doc_id=str(row.get("doc_id") or ""),
        chunk_type=str(row.get("chunk_type") or ""),
        document_type=row.get("document_type"),
        source_pdf=row.get("source_pdf"),
        page_number=_int_or_none(row.get("page_number")),
        text=text,
        text_preview=text_preview,
        metadata=md,
        score_keyword=None,
        score_vector=None,
        score_hybrid=None,
        rrf_score=None,
        clinical_rerank_score=None,
        final_score=None,
        retrieval_mode="hybrid",
        match_reason=["exact_analyte_sqlite_enrichment"],
    )


def _exact_analyte_sort_key(item: RetrievalResult) -> tuple[str, int, int, float]:
    md = item.metadata or {}
    doc_id = str(item.doc_id or "").strip().lower()
    page = item.page_number
    if page is None:
        page = _int_or_none(md.get("page_number"))
    row_index = _int_or_none(md.get("row_index"))
    if page is None:
        page = 999999
    if row_index is None:
        row_index = 999999
    final_score = item.final_score if item.final_score is not None else (item.score_hybrid or 0.0)
    return (doc_id, int(page), int(row_index), -float(final_score or 0.0))


def build_evidence_pack(
    response: SearchResponse,
    *,
    query: str,
    max_evidence: int = 6,
    exact_analyte: str | None = None,
    exact_analyte_rows: list[dict[str, Any]] | None = None,
    max_exact_analyte_results: int = 10,
) -> list[dict[str, Any]]:
    candidates = response.context_chunks if response.context_chunks else response.top_results
    if not candidates:
        return []

    allow_admin = _explicit_admin_intent(query)

    ordered = sorted(
        candidates,
        key=lambda r: (
            _chunk_priority(r.chunk_type),
            -(r.final_score if r.final_score is not None else (r.score_hybrid or 0.0)),
            -(r.score_vector or 0.0),
        ),
    )

    if exact_analyte:
        enriched_by_chunk: dict[str, RetrievalResult] = {}

        for item in ordered:
            md = item.metadata or {}
            analyte_norm = str(md.get("analyte_norm") or "")
            analyte_text = str(md.get("analyte") or "")
            if contains_exact_term(analyte_norm, exact_analyte) or contains_exact_term(analyte_text, exact_analyte):
                enriched_by_chunk[item.chunk_id] = item

        for row in (exact_analyte_rows or []):
            row_result = _row_to_retrieval_result(row)
            if row_result.chunk_id and row_result.chunk_id not in enriched_by_chunk:
                enriched_by_chunk[row_result.chunk_id] = row_result

        exact_candidates = list(enriched_by_chunk.values())
        exact_candidates.sort(key=_exact_analyte_sort_key)
        if exact_candidates:
            ordered = exact_candidates[: max(1, max_exact_analyte_results)]

    seen_chunk_ids: set[str] = set()
    seen_similarity_keys: set[tuple[str, str, str]] = set()
    kept: list[RetrievalResult] = []

    for item in ordered:
        ctype = (item.chunk_type or "").lower()
        if (ctype in _ADMIN_BLOCKED_TYPES) and not allow_admin:
            continue
        if item.chunk_id in seen_chunk_ids:
            continue
        s_key = _dedupe_key(item)
        if s_key in seen_similarity_keys:
            continue
        seen_chunk_ids.add(item.chunk_id)
        seen_similarity_keys.add(s_key)
        kept.append(item)
        if len(kept) >= max(1, max_evidence):
            break

    evidence_pack: list[dict[str, Any]] = []
    for idx, r in enumerate(kept, start=1):
        md = r.metadata or {}
        previous_result = md.get("previous_result")
        if previous_result in (None, ""):
            previous_result = md.get("previous_result_value_raw")

        evidence_pack.append(
            {
                "evidence_id": idx,
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "chunk_type": r.chunk_type,
                "analyte": md.get("analyte"),
                "analyte_norm": md.get("analyte_norm"),
                "parameter": md.get("parameter"),
                "value_raw": md.get("value_raw"),
                "value_numeric": _float_or_none(md.get("value_numeric")),
                "unit": md.get("unit"),
                "reference_range": md.get("reference_range"),
                "reference_low": md.get("reference_low"),
                "reference_high": md.get("reference_high"),
                "interpretation_status": md.get("interpretation_status"),
                "previous_result": previous_result,
                "previous_result_present": _int_flag(md.get("previous_result_present")),
                "section": md.get("section"),
                "source_kind": md.get("source_kind"),
                "source_table_id": md.get("source_table_id"),
                "page_number": r.page_number if r.page_number is not None else md.get("page_number"),
                "row_index": md.get("row_index"),
                "final_score": r.final_score if r.final_score is not None else r.score_hybrid,
                "clinical_rerank_score": r.clinical_rerank_score,
                "text_excerpt": _text_excerpt(r.text or r.text_preview),
            }
        )

    if exact_analyte:
        total_exact = len(evidence_pack)
        for ev in evidence_pack:
            ev["multiple_results_found"] = total_exact > 1
            ev["result_count_for_analyte"] = total_exact

    return evidence_pack
