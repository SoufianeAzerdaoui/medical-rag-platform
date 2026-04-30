from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from .config import DEFAULT_BATCH_SIZE, DEFAULT_EMBEDDING_MODEL
    from .models import RetrievalFilters
except ImportError:
    from config import DEFAULT_BATCH_SIZE, DEFAULT_EMBEDDING_MODEL
    from models import RetrievalFilters


class EmbeddingModelWrapper:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cpu"
        self.backend = "unknown"
        self._model: Any = None
        self._cache: dict[str, np.ndarray] = {}
        self._init_backend()

    def _detect_device(self) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _init_backend(self) -> None:
        self.device = self._detect_device()
        if self.model_name == "BAAI/bge-m3":
            try:
                from FlagEmbedding import BGEM3FlagModel

                self._model = BGEM3FlagModel(self.model_name, use_fp16=self.device == "cuda")
                self.backend = "FlagEmbedding"
                return
            except Exception:
                pass

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.backend = "sentence-transformers"
        except Exception as exc:
            raise RuntimeError(
                "Unable to load embedding backend. Install FlagEmbedding or sentence-transformers."
            ) from exc

    def encode_query(self, query: str) -> np.ndarray:
        q = (query or "").strip()
        if not q:
            raise ValueError("query is empty")
        if q in self._cache:
            return self._cache[q]

        if self.backend == "FlagEmbedding":
            out = self._model.encode([q], batch_size=1, max_length=8192)
            dense = out["dense_vecs"] if isinstance(out, dict) else out
            vec = np.asarray(dense, dtype=np.float32)[0]
        else:
            vec = np.asarray(
                self._model.encode([q], batch_size=1, normalize_embeddings=True, show_progress_bar=False),
                dtype=np.float32,
            )[0]

        self._cache[q] = vec
        return vec


class QdrantStore:
    def __init__(
        self,
        qdrant_dir: Path | str,
        collection_name: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.qdrant_dir = Path(qdrant_dir)
        if not self.qdrant_dir.exists():
            raise FileNotFoundError(f"Qdrant directory missing: {self.qdrant_dir}")

        from qdrant_client import QdrantClient

        self.client = QdrantClient(path=str(self.qdrant_dir))
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self._embedder: EmbeddingModelWrapper | None = None
        self._ensure_collection_exists()

    def _get_embedder(self) -> EmbeddingModelWrapper:
        if self._embedder is None:
            self._embedder = EmbeddingModelWrapper(
                model_name=self.embedding_model,
                batch_size=self.batch_size,
            )
        return self._embedder

    def _ensure_collection_exists(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            raise RuntimeError(
                f"Qdrant collection missing: {self.collection_name}. Available: {collections}"
            )

    def _build_qdrant_filter(self, filters: RetrievalFilters):
        from qdrant_client.http import models as qm

        must = []

        if filters.doc_id:
            must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=filters.doc_id)))
        if filters.document_type:
            must.append(qm.FieldCondition(key="document_type", match=qm.MatchValue(value=filters.document_type)))
        if filters.sample_type:
            must.append(qm.FieldCondition(key="sample_type", match=qm.MatchValue(value=filters.sample_type)))
        if filters.chunk_type:
            must.append(qm.FieldCondition(key="chunk_type", match=qm.MatchValue(value=filters.chunk_type)))
        if filters.source_pdf:
            must.append(qm.FieldCondition(key="source_pdf", match=qm.MatchValue(value=filters.source_pdf)))
        if filters.patient_token:
            must.append(qm.FieldCondition(key="patient_token", match=qm.MatchValue(value=filters.patient_token)))
        if filters.sample_token:
            must.append(qm.FieldCondition(key="sample_token", match=qm.MatchValue(value=filters.sample_token)))
        if filters.report_token:
            must.append(qm.FieldCondition(key="report_token", match=qm.MatchValue(value=filters.report_token)))

        if not must:
            return None
        return qm.Filter(must=must)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters,
    ) -> list[dict[str, Any]]:
        vector = self._get_embedder().encode_query(query)
        if vector.shape[0] != 1024:
            raise RuntimeError(
                f"Unexpected query embedding dimension={vector.shape[0]}, expected 1024"
            )

        qfilter = self._build_qdrant_filter(filters)
        points = None
        if hasattr(self.client, "search"):
            points = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector.tolist(),
                limit=int(top_k),
                query_filter=qfilter,
                with_payload=True,
                with_vectors=False,
            )
        else:
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=vector.tolist(),
                limit=int(top_k),
                query_filter=qfilter,
                with_payload=True,
                with_vectors=False,
            )
            points = resp.points if hasattr(resp, "points") else resp

        out: list[dict[str, Any]] = []
        for p in points:
            payload = p.payload or {}
            out.append(
                {
                    "qdrant_id": str(p.id),
                    "chunk_id": payload.get("chunk_id"),
                    "doc_id": payload.get("doc_id"),
                    "chunk_type": payload.get("chunk_type"),
                    "score_vector": float(p.score),
                    "payload": payload,
                }
            )
        return out
