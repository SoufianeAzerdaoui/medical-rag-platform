from __future__ import annotations

from typing import Any

try:
    from .models import RetrievalFilters, RetrievalResult
    from .keyword_search import KeywordSearcher
    from .vector_search import VectorSearcher
except ImportError:
    from models import RetrievalFilters, RetrievalResult
    from keyword_search import KeywordSearcher
    from vector_search import VectorSearcher


class HybridSearcher:
    def __init__(self, keyword_searcher: KeywordSearcher, vector_searcher: VectorSearcher) -> None:
        self.keyword_searcher = keyword_searcher
        self.vector_searcher = vector_searcher

    def search(
        self,
        query: str,
        *,
        top_k: int,
        keyword_top_k: int,
        vector_top_k: int,
        rrf_k: int,
        filters: RetrievalFilters,
    ) -> list[RetrievalResult]:
        if not (query or "").strip():
            raise ValueError("query is empty")

        keyword_results = self.keyword_searcher.search(
            query,
            top_k=max(keyword_top_k, top_k),
            filters=filters,
        )
        vector_results = self.vector_searcher.search(
            query,
            top_k=max(vector_top_k, top_k),
            filters=filters,
        )

        by_chunk_id: dict[str, RetrievalResult] = {}

        for r in keyword_results:
            clone = RetrievalResult(**r.to_dict())
            by_chunk_id[r.chunk_id] = clone

        for r in vector_results:
            if r.chunk_id in by_chunk_id:
                cur = by_chunk_id[r.chunk_id]
                cur.score_vector = r.score_vector
                cur.rank_vector = r.rank_vector
                cur.vector_rank = r.rank_vector
                if "vector_match" not in cur.match_reason:
                    cur.match_reason.append("vector_match")
            else:
                clone = RetrievalResult(**r.to_dict())
                by_chunk_id[r.chunk_id] = clone

        merged: list[RetrievalResult] = []
        for item in by_chunk_id.values():
            rk = item.rank_keyword
            rv = item.rank_vector
            score = 0.0
            if rk is not None:
                score += 1.0 / (rrf_k + rk)
            if rv is not None:
                score += 1.0 / (rrf_k + rv)
            item.score_hybrid = score
            if item.rank_keyword is not None and "keyword_match" not in item.match_reason:
                item.match_reason.append("keyword_match")
            if item.rank_vector is not None and "vector_match" not in item.match_reason:
                item.match_reason.append("vector_match")
            if item.rank_keyword is None and item.rank_vector is None:
                continue
            item.retrieval_mode = "hybrid"
            merged.append(item)

        merged.sort(key=lambda x: x.score_hybrid or 0.0, reverse=True)
        top = merged[:top_k]
        for i, item in enumerate(top, start=1):
            item.rank_hybrid = i
            item.hybrid_rank = i
            if item.rank_keyword is not None:
                item.keyword_rank = item.rank_keyword
            if item.rank_vector is not None:
                item.vector_rank = item.rank_vector
        return top
