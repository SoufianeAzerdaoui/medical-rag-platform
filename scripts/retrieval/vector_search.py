from __future__ import annotations

from typing import Any

try:
    from .filters import MappingResolver, resolve_filters, row_matches_filters
    from .models import RetrievalFilters, RetrievalResult
    from .qdrant_store import QdrantStore
    from .sqlite_store import SQLiteStore
except ImportError:
    from filters import MappingResolver, resolve_filters, row_matches_filters
    from models import RetrievalFilters, RetrievalResult
    from qdrant_store import QdrantStore
    from sqlite_store import SQLiteStore


def _preview(text: str, max_len: int = 220) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


class VectorSearcher:
    def __init__(
        self,
        qdrant_store: QdrantStore,
        sqlite_store: SQLiteStore,
        mapping_resolver: MappingResolver | None = None,
    ) -> None:
        self.qdrant_store = qdrant_store
        self.sqlite_store = sqlite_store
        self.mapping_resolver = mapping_resolver or MappingResolver()

    def _result_from_rows(self, point: dict[str, Any], row: dict[str, Any]) -> RetrievalResult:
        text = (row.get("text_for_embedding") or row.get("text_for_keyword") or "").strip()
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

        return RetrievalResult(
            chunk_id=str(row.get("chunk_id") or point.get("chunk_id") or ""),
            doc_id=str(row.get("doc_id") or point.get("doc_id") or ""),
            chunk_type=str(row.get("chunk_type") or point.get("chunk_type") or ""),
            document_type=row.get("document_type"),
            source_pdf=row.get("source_pdf") or (point.get("payload") or {}).get("source_pdf"),
            page_number=row.get("page_number") or (point.get("payload") or {}).get("page_number"),
            text=text,
            text_preview=_preview(text),
            metadata=metadata,
            score_vector=float(point.get("score_vector") or 0.0),
            retrieval_mode="vector",
            match_reason=["vector_match"],
        )

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters,
        candidate_multiplier: int = 4,
    ) -> list[RetrievalResult]:
        q = (query or "").strip()
        if not q:
            raise ValueError("query is empty")

        resolved_filters = resolve_filters(filters, self.mapping_resolver)

        requested = max(top_k, 1)
        qdrant_limit = requested
        if not resolved_filters.is_empty():
            qdrant_limit = max(requested * candidate_multiplier, requested + 10)

        points = self.qdrant_store.search(
            q,
            top_k=qdrant_limit,
            filters=resolved_filters,
        )
        if not points:
            return []

        chunk_ids = [str(p.get("chunk_id") or "") for p in points if p.get("chunk_id")]
        rows_by_id = self.sqlite_store.get_chunk_rows_by_ids(chunk_ids)

        results: list[RetrievalResult] = []
        for point in points:
            chunk_id = str(point.get("chunk_id") or "")
            row = rows_by_id.get(chunk_id)
            if not row:
                continue
            if not row_matches_filters(row, resolved_filters):
                continue
            result = self._result_from_rows(point, row)
            if not resolved_filters.is_empty():
                result.match_reason.append("metadata_filter_applied")
            results.append(result)

        results.sort(key=lambda r: r.score_vector or 0.0, reverse=True)
        final = results[:requested]
        for i, item in enumerate(final, start=1):
            item.rank_vector = i
            item.vector_rank = i
        return final
