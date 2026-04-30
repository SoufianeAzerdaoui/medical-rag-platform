from __future__ import annotations

from typing import Any

try:
    from .filters import MappingResolver, resolve_filters
    from .models import RetrievalFilters, RetrievalResult
    from .sqlite_store import SQLiteStore
except ImportError:
    from filters import MappingResolver, resolve_filters
    from models import RetrievalFilters, RetrievalResult
    from sqlite_store import SQLiteStore


def _preview(text: str, max_len: int = 220) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _row_to_result(row: dict[str, Any], retrieval_mode: str) -> RetrievalResult:
    text = (row.get("text_for_keyword") or row.get("text_for_embedding") or "").strip()
    metadata = {
        k: row.get(k)
        for k in [
            "document_type",
            "sample_type",
            "request_date",
            "patient_token",
            "sample_token",
            "report_token",
            "validation_status",
            "reference_quality_status",
            "age_consistency_status",
            "section",
            "section_norm",
            "analyte",
            "value_raw",
            "unit",
            "source_pdf",
            "page_number",
            "parent_chunk_id",
        ]
        if k in row
    }

    bm25 = row.get("bm25_score")
    score_keyword = -float(bm25) if bm25 is not None else None

    return RetrievalResult(
        chunk_id=str(row.get("chunk_id") or ""),
        doc_id=str(row.get("doc_id") or ""),
        chunk_type=str(row.get("chunk_type") or ""),
        document_type=row.get("document_type"),
        source_pdf=row.get("source_pdf"),
        page_number=row.get("page_number"),
        text=text,
        text_preview=_preview(text),
        metadata=metadata,
        score_keyword=score_keyword,
        retrieval_mode=retrieval_mode,
        match_reason=["keyword_match"],
    )


class KeywordSearcher:
    def __init__(self, sqlite_store: SQLiteStore, mapping_resolver: MappingResolver | None = None) -> None:
        self.sqlite_store = sqlite_store
        self.mapping_resolver = mapping_resolver or MappingResolver()

    def search(self, query: str, *, top_k: int, filters: RetrievalFilters) -> list[RetrievalResult]:
        q = (query or "").strip()
        if not q:
            raise ValueError("query is empty")

        resolved_filters = resolve_filters(filters, self.mapping_resolver)
        rows = self.sqlite_store.keyword_search_rows(
            q,
            top_k=top_k,
            filters=resolved_filters,
        )

        results: list[RetrievalResult] = []
        for i, row in enumerate(rows, start=1):
            item = _row_to_result(row, retrieval_mode="keyword")
            item.rank_keyword = i
            item.keyword_rank = i
            if not resolved_filters.is_empty():
                item.match_reason.append("metadata_filter_applied")
            results.append(item)

        return results
