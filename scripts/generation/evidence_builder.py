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

_UNIT_PATTERN = re.compile(
    r"(?:ug/ml|mg/l|ng/ml|ui/l|iu/l|mmol/l|pg/ml|uui/ml|uu/ml|uiu/ml|mui/l|mui/ml)",
    re.IGNORECASE,
)


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


def _clean_analyte_display(analyte: Any, parameter: Any) -> str:
    raw = str(analyte or parameter or "").strip()
    if not raw:
        return "non précisé"
    cleaned = re.sub(r"\([^)]*\d[^)]*\)", "", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -;,:")
    return cleaned or raw


def _display_quality(analyte_display: str, unit: Any) -> tuple[str, list[str]]:
    reasons: list[str] = []
    analyte_norm = _norm(analyte_display)
    unit_norm = _norm(str(unit or ""))
    analyte_compact = analyte_norm.replace(" ", "")
    unit_compact = unit_norm.replace(" ", "")

    if _UNIT_PATTERN.match(analyte_norm):
        reasons.append("analyte_starts_with_unit")
    if len(_UNIT_PATTERN.findall(analyte_norm)) >= 2:
        reasons.append("analyte_contains_repeated_units")

    if unit_compact:
        doubled = unit_compact + unit_compact
        if doubled in analyte_compact or doubled in unit_compact:
            reasons.append("unit_concatenated")

    return ("low", reasons) if reasons else ("high", [])


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


def _clean_reference_range(reference_range: Any, value_raw: Any) -> str | None:
    ref = str(reference_range or "").strip()
    if not ref:
        return None
    val = str(value_raw or "").strip()
    if not val:
        return ref
    # Fix known polluted patterns where measured value is appended to reference range (typically at end).
    val_num = re.sub(r"^[<>]=?\s*", "", val).strip()
    if val_num:
        num_re = re.escape(val_num).replace(r"\,", "[,.]").replace(r"\.", "[,.]")
        tail_pattern = re.compile(
            rf"\s*[<>]=?\s*{num_re}(?:\s*[a-zA-Zµ/%]+(?:/[a-zA-Zµ%]+)?)?\s*$",
            flags=re.IGNORECASE,
        )
        if tail_pattern.search(ref):
            ref = tail_pattern.sub("", ref).strip(" ;,|-")
    ref = re.sub(r"\s+", " ", ref).strip()
    return ref or None


def _extract_reference_from_analyte(analyte: Any) -> str | None:
    text = str(analyte or "")
    m = re.search(r"\(([^()]*\d[^()]*)\)", text)
    if not m:
        return None
    ref = re.sub(r"\s+", " ", m.group(1)).strip()
    if "-" in ref and any(ch.isdigit() for ch in ref):
        return ref
    return None


def _derive_interpretation_status(value_numeric: Any, reference_range: Any, fallback: Any) -> str | None:
    status = str(fallback or "").strip().lower()
    if status and status not in {"unknown", "n/a", "none", "null"}:
        return status
    ref = str(reference_range or "").strip()
    if not ref:
        return status or None

    nums = re.findall(r"\d+(?:[.,]\d+)?", ref)
    if not nums:
        return status or None
    try:
        val = float(str(value_numeric).replace(",", "."))
    except Exception:
        return status or None
    try:
        if "<" in ref and nums:
            hi = float(nums[0].replace(",", "."))
            return "below_reference" if val < hi else "above_reference"
        if ">" in ref and nums:
            lo = float(nums[0].replace(",", "."))
            return "above_reference" if val > lo else "below_reference"
        if len(nums) >= 2:
            lo = float(nums[0].replace(",", "."))
            hi = float(nums[1].replace(",", "."))
            if val < lo:
                return "below_reference"
            if val > hi:
                return "above_reference"
            return "within_reference"
    except Exception:
        return status or None
    return status or None


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


def build_evidence_pack(
    response: SearchResponse,
    *,
    query: str,
    max_evidence: int = 6,
    exact_analyte: str | None = None,
    exact_analyte_rows: list[dict[str, Any]] | None = None,
    supplemental_rows: list[dict[str, Any]] | None = None,
    max_exact_analyte_results: int = 10,
) -> list[dict[str, Any]]:
    candidates = response.context_chunks if response.context_chunks else response.top_results
    ordered = list(candidates) if candidates else []

    if not ordered and exact_analyte_rows:
        ordered = [_row_to_retrieval_result(r) for r in exact_analyte_rows]
    if not ordered and supplemental_rows:
        ordered = [_row_to_retrieval_result(r) for r in supplemental_rows]
    if not ordered:
        return []

    allow_admin = _explicit_admin_intent(query)

    if supplemental_rows:
        existing_ids = {x.chunk_id for x in ordered}
        for row in supplemental_rows:
            row_result = _row_to_retrieval_result(row)
            if row_result.chunk_id and row_result.chunk_id not in existing_ids:
                ordered.append(row_result)
                existing_ids.add(row_result.chunk_id)

    if exact_analyte:
        exact_candidates: list[RetrievalResult] = []
        exact_chunk_ids: set[str] = set()
        for item in ordered:
            md = item.metadata or {}
            analyte_norm = str(md.get("analyte_norm") or "")
            analyte_text = str(md.get("analyte") or "")
            if contains_exact_term(analyte_norm, exact_analyte) or contains_exact_term(analyte_text, exact_analyte):
                if item.chunk_id not in exact_chunk_ids:
                    exact_candidates.append(item)
                    exact_chunk_ids.add(item.chunk_id)

        for row in (exact_analyte_rows or []):
            row_result = _row_to_retrieval_result(row)
            if row_result.chunk_id and row_result.chunk_id not in exact_chunk_ids:
                exact_candidates.append(row_result)
                exact_chunk_ids.add(row_result.chunk_id)
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
        reference_range_raw = md.get("reference_range")
        reference_range = _clean_reference_range(reference_range_raw, md.get("value_raw"))
        if not reference_range:
            reference_range = _extract_reference_from_analyte(md.get("analyte"))
        interpretation_status = _derive_interpretation_status(
            md.get("value_numeric"),
            reference_range,
            md.get("interpretation_status"),
        )
        analyte_display = _clean_analyte_display(md.get("analyte"), md.get("parameter"))
        display_quality, quality_reasons = _display_quality(analyte_display, md.get("unit"))
        source = "sqlite_exact_match" if "exact_analyte_sqlite_enrichment" in (r.match_reason or []) else "retrieval"

        evidence_pack.append(
            {
                "evidence_id": idx,
                "rank": idx,
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "chunk_type": r.chunk_type,
                "analyte": md.get("analyte"),
                "analyte_display": analyte_display,
                "analyte_norm": md.get("analyte_norm"),
                "parameter": md.get("parameter"),
                "value_raw": md.get("value_raw"),
                "value_numeric": _float_or_none(md.get("value_numeric")),
                "unit": md.get("unit"),
                "reference_range": reference_range,
                "reference_range_raw": reference_range_raw,
                "reference_low": md.get("reference_low"),
                "reference_high": md.get("reference_high"),
                "interpretation_status": interpretation_status,
                "previous_result": previous_result,
                "previous_result_present": _int_flag(md.get("previous_result_present")),
                "section": md.get("section"),
                "source_kind": md.get("source_kind"),
                "source_table_id": md.get("source_table_id"),
                "source_pdf": r.source_pdf or md.get("source_pdf"),
                "page_number": r.page_number if r.page_number is not None else md.get("page_number"),
                "row_index": md.get("row_index"),
                "final_score": r.final_score if r.final_score is not None else r.score_hybrid,
                "clinical_rerank_score": r.clinical_rerank_score,
                "source": source,
                "evidence_display_quality": display_quality,
                "evidence_display_quality_reasons": quality_reasons,
                "text_excerpt": _text_excerpt(r.text or r.text_preview),
            }
        )

    if exact_analyte:
        total_exact = len(evidence_pack)
        for ev in evidence_pack:
            ev["multiple_results_found"] = total_exact > 1
            ev["result_count_for_analyte"] = total_exact

    return evidence_pack
