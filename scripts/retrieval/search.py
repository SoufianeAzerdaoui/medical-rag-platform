#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .config import (
        DEFAULT_CONTEXT_MAX_CHUNKS,
        DEFAULT_KEYWORD_TOP_K,
        DEFAULT_RRF_K,
        DEFAULT_TOP_K,
        DEFAULT_VECTOR_TOP_K,
        DEFAULT_EMBEDDING_MODEL,
        INDEX_DIR,
        QDRANT_COLLECTION,
        QDRANT_DIR,
        SQLITE_PATH,
    )
    from .context_builder import ContextBuilder
    from .filters import MappingResolver
    from .hybrid_search import HybridSearcher
    from .keyword_search import KeywordSearcher
    from .models import RetrievalFilters, SearchResponse
    from .qdrant_store import QdrantStore
    from .sqlite_store import SQLiteStore
    from .vector_search import VectorSearcher
except ImportError:
    from config import (
        DEFAULT_CONTEXT_MAX_CHUNKS,
        DEFAULT_KEYWORD_TOP_K,
        DEFAULT_RRF_K,
        DEFAULT_TOP_K,
        DEFAULT_VECTOR_TOP_K,
        DEFAULT_EMBEDDING_MODEL,
        INDEX_DIR,
        QDRANT_COLLECTION,
        QDRANT_DIR,
        SQLITE_PATH,
    )
    from context_builder import ContextBuilder
    from filters import MappingResolver
    from hybrid_search import HybridSearcher
    from keyword_search import KeywordSearcher
    from models import RetrievalFilters, SearchResponse
    from qdrant_store import QdrantStore
    from sqlite_store import SQLiteStore
    from vector_search import VectorSearcher


class SearchEngine:
    def __init__(
        self,
        *,
        sqlite_path: Path | str = SQLITE_PATH,
        qdrant_dir: Path | str = QDRANT_DIR,
        collection: str = QDRANT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.mapping_resolver = MappingResolver()
        self.sqlite_store = SQLiteStore(sqlite_path)
        self.qdrant_store = QdrantStore(qdrant_dir, collection_name=collection, embedding_model=embedding_model)
        self.keyword_searcher = KeywordSearcher(self.sqlite_store, self.mapping_resolver)
        self.vector_searcher = VectorSearcher(self.qdrant_store, self.sqlite_store, self.mapping_resolver)
        self.hybrid_searcher = HybridSearcher(self.keyword_searcher, self.vector_searcher)
        self.context_builder = ContextBuilder(self.sqlite_store)

    def close(self) -> None:
        self.sqlite_store.close()

    def search(
        self,
        *,
        query: str,
        mode: str = "hybrid",
        top_k: int = DEFAULT_TOP_K,
        keyword_top_k: int = DEFAULT_KEYWORD_TOP_K,
        vector_top_k: int = DEFAULT_VECTOR_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
        filters: RetrievalFilters | None = None,
        expand_context: bool = True,
        max_context_chunks: int = DEFAULT_CONTEXT_MAX_CHUNKS,
        strict_context: bool = True,
        debug_context: bool = False,
    ) -> SearchResponse:
        q = (query or "").strip()
        if not q:
            raise ValueError("query is empty")

        filters = filters or RetrievalFilters()
        if mode not in {"keyword", "vector", "hybrid"}:
            raise ValueError(f"Unsupported mode: {mode}")

        if mode == "keyword":
            results = self.keyword_searcher.search(q, top_k=top_k, filters=filters)
        elif mode == "vector":
            results = self.vector_searcher.search(q, top_k=top_k, filters=filters)
        else:
            results = self.hybrid_searcher.search(
                q,
                top_k=top_k,
                keyword_top_k=keyword_top_k,
                vector_top_k=vector_top_k,
                rrf_k=rrf_k,
                filters=filters,
            )

        if not results:
            return SearchResponse(
                query=q,
                mode=mode,
                filters=filters.to_dict(),
                top_results=[],
                context_chunks=[],
                sources=[],
                excluded_context_candidates=[],
                answerability={"status": "insufficient_context", "reason": "no_results"},
            )

        if expand_context:
            response = self.context_builder.build(
                query=q,
                mode=mode,
                top_results=results,
                filters=filters,
                max_context_chunks=max_context_chunks,
                strict_context=strict_context,
                debug_context=debug_context,
            )
            response.filters = filters.to_dict()
            return response

        sources = [
            {
                "doc_id": r.doc_id,
                "source_pdf": r.source_pdf,
                "page_number": r.page_number,
                "chunk_id": r.chunk_id,
                "chunk_type": r.chunk_type,
                "text_preview": r.text_preview,
            }
            for r in results
        ]
        return SearchResponse(
            query=q,
            mode=mode,
            filters=filters.to_dict(),
            top_results=results,
            context_chunks=[],
            sources=sources,
            excluded_context_candidates=[],
            answerability={"status": "answerable", "reason": "raw_retrieval_only"},
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local medical RAG retrieval CLI")
    parser.add_argument("--query", required=True)
    parser.add_argument("--mode", choices=["keyword", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--keyword-top-k", type=int, default=DEFAULT_KEYWORD_TOP_K)
    parser.add_argument("--vector-top-k", type=int, default=DEFAULT_VECTOR_TOP_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)

    parser.add_argument("--document-type")
    parser.add_argument("--doc-id")
    parser.add_argument("--patient-id")
    parser.add_argument("--sample-id")
    parser.add_argument("--sample-type")
    parser.add_argument("--request-date")
    parser.add_argument("--chunk-type")
    parser.add_argument("--source-pdf")

    parser.add_argument("--expand-context", dest="expand_context", action="store_true", default=True)
    parser.add_argument("--no-expand-context", dest="expand_context", action="store_false")
    parser.add_argument("--max-context-chunks", type=int, default=DEFAULT_CONTEXT_MAX_CHUNKS)
    parser.add_argument("--strict-context", dest="strict_context", action="store_true", default=True)
    parser.add_argument("--no-strict-context", dest="strict_context", action="store_false")
    parser.add_argument("--debug-context", action="store_true")

    parser.add_argument("--index-dir", default=str(INDEX_DIR))
    parser.add_argument("--collection", default=QDRANT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _print_human(response: SearchResponse) -> None:
    print(f"QUERY: {response.query}")
    print(f"MODE: {response.mode}")
    print(f"FILTERS: {response.filters or {}}")
    print(f"TOP K RETURNED: {len(response.top_results)}")

    for i, r in enumerate(response.top_results, start=1):
        print(f"\n#{i}")
        print(f"chunk_id: {r.chunk_id}")
        print(f"doc_id: {r.doc_id}")
        print(f"chunk_type: {r.chunk_type}")
        print(f"source_pdf: {r.source_pdf}")
        print(f"page_number: {r.page_number}")
        print(f"score_hybrid: {r.score_hybrid}")
        print(f"score_keyword: {r.score_keyword}")
        print(f"score_vector: {r.score_vector}")
        print(f"keyword_rank: {r.keyword_rank}")
        print(f"vector_rank: {r.vector_rank}")
        print(f"hybrid_rank: {r.hybrid_rank}")
        print(f"match_reason: {', '.join(r.match_reason)}")
        print(f"preview: {r.text_preview}")

    if response.context_chunks:
        print("\nContext expansion:")
        for c in response.context_chunks:
            print(f"- {c.chunk_id} | {c.chunk_type} | {c.doc_id} | {', '.join(c.match_reason)}")
    else:
        print("\nContext expansion: (empty)")

    if response.excluded_context_candidates:
        included_dbg = [e for e in response.excluded_context_candidates if e.get("decision") == "include"]
        excluded_dbg = [e for e in response.excluded_context_candidates if e.get("decision") == "exclude"]
        if included_dbg:
            print("\nIncluded context candidates:")
            for e in included_dbg:
                print(
                    f"- {e.get('chunk_id')} | {e.get('chunk_type')} | {e.get('doc_id')} | "
                    f"score={e.get('evidence_score')} threshold={e.get('min_evidence_score')} | "
                    f"match_reason={e.get('match_reason')} | evidence_reasons={e.get('evidence_reasons')}"
                )
        if excluded_dbg:
            print("\nExcluded context candidates:")
            for e in excluded_dbg:
                print(
                    f"- {e.get('chunk_id')} | {e.get('chunk_type')} | {e.get('doc_id')} | "
                    f"score={e.get('evidence_score')} threshold={e.get('min_evidence_score')} | "
                    f"reason={e.get('reason')} | match_reason={e.get('match_reason')} | "
                    f"evidence_reasons={e.get('evidence_reasons')}"
                )

    if response.answerability:
        print("\nAnswerability:")
        print(json.dumps(response.answerability, ensure_ascii=False))


def _build_filters(args: argparse.Namespace) -> RetrievalFilters:
    return RetrievalFilters(
        document_type=args.document_type,
        doc_id=args.doc_id,
        patient_id=args.patient_id,
        sample_id=args.sample_id,
        sample_type=args.sample_type,
        request_date=args.request_date,
        chunk_type=args.chunk_type,
        source_pdf=args.source_pdf,
    )


def main() -> int:
    args = _parse_args()

    index_dir = Path(args.index_dir)
    sqlite_path = index_dir / "medical_rag.sqlite"
    qdrant_dir = index_dir / "qdrant"

    engine = SearchEngine(
        sqlite_path=sqlite_path,
        qdrant_dir=qdrant_dir,
        collection=args.collection,
        embedding_model=args.embedding_model,
    )
    try:
        response = engine.search(
            query=args.query,
            mode=args.mode,
            top_k=args.top_k,
            keyword_top_k=args.keyword_top_k,
            vector_top_k=args.vector_top_k,
            rrf_k=args.rrf_k,
            filters=_build_filters(args),
            expand_context=args.expand_context,
            max_context_chunks=args.max_context_chunks,
            strict_context=args.strict_context,
            debug_context=args.debug_context,
        )

        if args.json:
            print(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_human(response)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
