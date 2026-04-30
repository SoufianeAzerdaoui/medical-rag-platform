from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RetrievalFilters:
    document_type: str | None = None
    doc_id: str | None = None
    patient_id: str | None = None
    sample_id: str | None = None
    request_date: str | None = None
    request_date_from: str | None = None
    request_date_to: str | None = None
    sample_type: str | None = None
    chunk_type: str | None = None
    source_pdf: str | None = None

    # Resolved token values (after optional mapping lookup)
    patient_token: str | None = None
    sample_token: str | None = None
    report_token: str | None = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.document_type,
                self.doc_id,
                self.patient_id,
                self.sample_id,
                self.request_date,
                self.request_date_from,
                self.request_date_to,
                self.sample_type,
                self.chunk_type,
                self.source_pdf,
                self.patient_token,
                self.sample_token,
                self.report_token,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    chunk_type: str
    document_type: str | None
    source_pdf: str | None
    page_number: int | None
    text: str
    text_preview: str
    metadata: dict[str, Any] = field(default_factory=dict)

    score_keyword: float | None = None
    score_vector: float | None = None
    score_hybrid: float | None = None

    rank_keyword: int | None = None
    rank_vector: int | None = None
    rank_hybrid: int | None = None

    # Explicit aliases required in specification
    keyword_rank: int | None = None
    vector_rank: int | None = None
    hybrid_rank: int | None = None

    retrieval_mode: str = "hybrid"
    match_reason: list[str] = field(default_factory=list)

    def finalize_rank_aliases(self) -> None:
        self.keyword_rank = self.rank_keyword
        self.vector_rank = self.rank_vector
        self.hybrid_rank = self.rank_hybrid

    def to_dict(self) -> dict[str, Any]:
        self.finalize_rank_aliases()
        return asdict(self)


@dataclass
class SearchResponse:
    query: str
    mode: str
    filters: dict[str, Any]
    top_results: list[RetrievalResult]
    context_chunks: list[RetrievalResult] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    excluded_context_candidates: list[dict[str, Any]] = field(default_factory=list)
    answerability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "filters": self.filters,
            "top_results": [r.to_dict() for r in self.top_results],
            "context_chunks": [r.to_dict() for r in self.context_chunks],
            "sources": self.sources,
            "excluded_context_candidates": self.excluded_context_candidates,
            "answerability": self.answerability,
        }
