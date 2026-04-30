#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from config import (
    ALLOWED_SCHEMA_VERSION,
    DEFAULT_BATCH_BGE,
    DEFAULT_BATCH_E5,
    DEFAULT_BGE_MODEL,
    DEFAULT_COLLECTION,
    FORBIDDEN_METADATA_FIELDS,
    FORBIDDEN_SCHEMA_VERSIONS,
    FORBIDDEN_TEXT_PATTERNS,
    INDEXING_REPORT_FILENAME,
    INDEX_MANIFEST_FILENAME,
    QDRANT_DIRNAME,
    REQUIRED_CHUNK_FIELDS,
    SQLITE_FILENAME,
    resolve_repo_root,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def uuid_from_chunk_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_safe_chunks_path(chunks_path: Path) -> None:
    path_text = str(chunks_path).replace("\\", "/")
    if "/data/private/" in path_text or path_text.endswith("/data/private"):
        raise ValueError("Refus: lecture de data/private/ interdite.")
    if "chunks.raw.jsonl" in path_text:
        raise ValueError("Refus: chunks.raw.jsonl ne doit jamais etre indexe.")
    if not path_text.endswith("data/chunks/chunks.anonymized.jsonl"):
        raise ValueError(
            "Refus: seul data/chunks/chunks.anonymized.jsonl est autorise pour l'indexing."
        )


def get_nested(obj: dict[str, Any], key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else default


def get_source_table_id(chunk: dict[str, Any]) -> str | None:
    provenance = chunk.get("provenance") or {}
    metadata = chunk.get("metadata") or {}

    if provenance.get("source_table_id"):
        return provenance.get("source_table_id")

    for key in ["table_ids", "source_table_ids"]:
        values = provenance.get(key) or metadata.get(key)
        if isinstance(values, list) and values:
            return values[0]
    return None


@dataclass
class ValidationResult:
    chunks: list[dict[str, Any]]
    errors: list[str]
    warnings: list[str]
    duplicate_chunk_ids: int
    chunks_by_type: dict[str, int]


def load_and_validate_chunks(chunks_path: Path) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    chunks: list[dict[str, Any]] = []
    chunk_ids: set[str] = set()
    duplicate_count = 0
    parents_to_check: list[tuple[str, str]] = []
    by_type: Counter[str] = Counter()
    vector_true = 0
    keyword_true = 0

    if not chunks_path.exists():
        errors.append(f"Fichier introuvable: {chunks_path}")
        return ValidationResult([], errors, warnings, 0, {})

    with chunks_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Ligne {line_number}: JSON invalide ({exc})")
                continue

            missing = REQUIRED_CHUNK_FIELDS.difference(chunk.keys())
            if missing:
                errors.append(f"Ligne {line_number}: champs manquants: {sorted(missing)}")
                continue

            schema_version = chunk.get("schema_version")
            if schema_version in FORBIDDEN_SCHEMA_VERSIONS:
                errors.append(
                    f"Ligne {line_number}: schema_version interdit detecte ({schema_version})"
                )
            if schema_version != ALLOWED_SCHEMA_VERSION:
                errors.append(
                    f"Ligne {line_number}: schema_version attendu={ALLOWED_SCHEMA_VERSION}, recu={schema_version}"
                )

            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not chunk_id:
                errors.append(f"Ligne {line_number}: chunk_id vide")
                continue
            if chunk_id in chunk_ids:
                duplicate_count += 1
                errors.append(f"Ligne {line_number}: duplicate chunk_id={chunk_id}")
            chunk_ids.add(chunk_id)

            parent_chunk_id = chunk.get("parent_chunk_id")
            if parent_chunk_id is not None:
                parents_to_check.append((chunk_id, str(parent_chunk_id)))

            metadata = chunk.get("metadata") or {}
            if not isinstance(metadata, dict):
                errors.append(f"Ligne {line_number}: metadata doit etre un objet JSON")
                continue

            for forbidden_key in FORBIDDEN_METADATA_FIELDS:
                if forbidden_key in metadata:
                    errors.append(
                        f"Ligne {line_number}: metadata contient champ interdit '{forbidden_key}'"
                    )

            patient_name = metadata.get("patient_name")
            if patient_name is not None and patient_name != "PATIENT_ANON":
                errors.append(
                    f"Ligne {line_number}: patient_name non autorise ({patient_name!r})"
                )

            text_for_embedding = str(chunk.get("text_for_embedding") or "")
            text_for_keyword = str(chunk.get("text_for_keyword") or "")
            scan_text = f"{text_for_embedding}\n{text_for_keyword}".lower()
            for marker in FORBIDDEN_TEXT_PATTERNS:
                if marker.lower() in scan_text:
                    errors.append(
                        f"Ligne {line_number}: motif interdit detecte dans le texte ({marker})"
                    )

            routing = chunk.get("routing") or {}
            if not isinstance(routing, dict):
                errors.append(f"Ligne {line_number}: routing doit etre un objet JSON")
                continue

            if bool(routing.get("vector_index")):
                vector_true += 1
            if bool(routing.get("keyword_index")):
                keyword_true += 1

            by_type.update([str(chunk.get("chunk_type"))])
            chunks.append(chunk)

    chunk_ids_local = {c["chunk_id"] for c in chunks}
    for child_chunk_id, parent_chunk_id in parents_to_check:
        if parent_chunk_id not in chunk_ids_local:
            errors.append(
                f"parent_chunk_id manquant: parent={parent_chunk_id} reference par child={child_chunk_id}"
            )

    if vector_true == 0:
        errors.append("Aucun chunk avec routing.vector_index == true")
    if keyword_true == 0:
        errors.append("Aucun chunk avec routing.keyword_index == true")

    return ValidationResult(
        chunks=chunks,
        errors=errors,
        warnings=warnings,
        duplicate_chunk_ids=duplicate_count,
        chunks_by_type=dict(by_type),
    )


class EmbeddingModelWrapper:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.backend = "unknown"
        self.device = "cpu"
        self._model: Any = None
        self._init_backend()

    def _detect_device(self) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _init_backend(self) -> None:
        self.device = self._detect_device()
        if self.model_name == DEFAULT_BGE_MODEL:
            try:
                from FlagEmbedding import BGEM3FlagModel

                use_fp16 = self.device == "cuda"
                self._model = BGEM3FlagModel(self.model_name, use_fp16=use_fp16)
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
                "Impossible de charger un modele d'embedding. Installez "
                "`FlagEmbedding` (pour BAAI/bge-m3) ou `sentence-transformers`.\n"
                "Exemple: pip install FlagEmbedding sentence-transformers torch numpy"
            ) from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self.backend == "FlagEmbedding":
            outputs = self._model.encode(texts, batch_size=self.batch_size, max_length=8192)
            dense = outputs["dense_vecs"] if isinstance(outputs, dict) else outputs
            return np.asarray(dense, dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


def iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def setup_sqlite(db_path: Path, reset: bool) -> sqlite3.Connection:
    if reset and db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chunks (
          chunk_id TEXT PRIMARY KEY,
          qdrant_point_id TEXT,
          doc_id TEXT,
          parent_chunk_id TEXT,
          chunk_type TEXT,
          modality TEXT,
          schema_version TEXT,
          text_for_embedding TEXT,
          text_for_keyword TEXT,
          vector_index INTEGER,
          keyword_index INTEGER,
          metadata_index INTEGER,
          object_store_reference INTEGER,
          priority TEXT
        );

        CREATE TABLE IF NOT EXISTS metadata_chunks (
          chunk_id TEXT PRIMARY KEY,
          doc_id TEXT,
          chunk_type TEXT,
          patient_token TEXT,
          sample_token TEXT,
          report_token TEXT,
          document_type TEXT,
          pdf_type TEXT,
          sample_type TEXT,
          sample_label TEXT,
          laboratory TEXT,
          department TEXT,
          organization TEXT,
          country TEXT,
          ministry TEXT,
          section TEXT,
          section_norm TEXT,
          analyte TEXT,
          analyte_norm TEXT,
          parameter TEXT,
          parameter_norm TEXT,
          value_raw TEXT,
          value_numeric REAL,
          unit TEXT,
          unit_quality_status TEXT,
          reference_range TEXT,
          reference_quality_status TEXT,
          reference_complexity TEXT,
          result_kind TEXT,
          interpretation_status TEXT,
          result_quality_status TEXT,
          validation_status TEXT,
          age_consistency_status TEXT,
          patient_age REAL,
          patient_sex TEXT,
          observation_date TEXT,
          report_date TEXT,
          request_date TEXT,
          received_date TEXT,
          validation_date TEXT,
          page_number INTEGER,
          row_index INTEGER,
          confidence TEXT,
          confidence_score REAL,
          source_pdf TEXT,
          source_table_id TEXT,
          source_kind TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS keyword_chunks_fts USING fts5(
          chunk_id UNINDEXED,
          doc_id UNINDEXED,
          chunk_type UNINDEXED,
          text_for_keyword
        );

        CREATE TABLE IF NOT EXISTS object_references (
          chunk_id TEXT PRIMARY KEY,
          doc_id TEXT,
          source_pdf TEXT,
          page_number INTEGER,
          source_table_id TEXT,
          source_image_ids_json TEXT,
          source_table_ids_json TEXT,
          table_ids_json TEXT,
          image_ids_json TEXT,
          ocr_visual_ids_json TEXT,
          provenance_json TEXT
        );
        """
    )
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_metadata_doc_id ON metadata_chunks(doc_id);
        CREATE INDEX IF NOT EXISTS idx_metadata_chunk_type ON metadata_chunks(chunk_type);
        CREATE INDEX IF NOT EXISTS idx_metadata_patient_token ON metadata_chunks(patient_token);
        CREATE INDEX IF NOT EXISTS idx_metadata_sample_token ON metadata_chunks(sample_token);
        CREATE INDEX IF NOT EXISTS idx_metadata_report_token ON metadata_chunks(report_token);
        CREATE INDEX IF NOT EXISTS idx_metadata_document_type ON metadata_chunks(document_type);
        CREATE INDEX IF NOT EXISTS idx_metadata_sample_type ON metadata_chunks(sample_type);
        CREATE INDEX IF NOT EXISTS idx_metadata_analyte_norm ON metadata_chunks(analyte_norm);
        CREATE INDEX IF NOT EXISTS idx_metadata_interpretation_status ON metadata_chunks(interpretation_status);
        CREATE INDEX IF NOT EXISTS idx_metadata_validation_status ON metadata_chunks(validation_status);
        CREATE INDEX IF NOT EXISTS idx_metadata_reference_quality_status ON metadata_chunks(reference_quality_status);
        CREATE INDEX IF NOT EXISTS idx_metadata_unit_quality_status ON metadata_chunks(unit_quality_status);
        CREATE INDEX IF NOT EXISTS idx_metadata_page_number ON metadata_chunks(page_number);
        """
    )
    return conn


def build_qdrant_client(qdrant_path: Path):
    from qdrant_client import QdrantClient

    return QdrantClient(path=str(qdrant_path))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RAG indexes from anonymized chunks.")
    parser.add_argument("--chunks", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", default=DEFAULT_BGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_BGE)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--skip-keyword", action="store_true")
    parser.add_argument("--skip-metadata", action="store_true")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    chunks_path = (repo_root / args.chunks).resolve() if not Path(args.chunks).is_absolute() else Path(args.chunks).resolve()
    index_dir = (repo_root / args.index_dir).resolve() if not Path(args.index_dir).is_absolute() else Path(args.index_dir).resolve()
    qdrant_dir = index_dir / QDRANT_DIRNAME
    sqlite_path = index_dir / SQLITE_FILENAME
    report_path = index_dir / INDEXING_REPORT_FILENAME
    manifest_path = index_dir / INDEX_MANIFEST_FILENAME

    index_dir.mkdir(parents=True, exist_ok=True)
    qdrant_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "chunks_path": str(chunks_path),
        "index_dir": str(index_dir),
        "collection": args.collection,
        "embedding_model": args.embedding_model,
        "embedding_backend": None,
        "device": None,
        "batch_size": args.batch_size,
        "total_chunks": 0,
        "chunks_by_type": {},
        "vector_indexed_chunks": 0,
        "keyword_indexed_chunks": 0,
        "metadata_indexed_chunks": 0,
        "skipped_vector_chunks": 0,
        "skipped_keyword_chunks": 0,
        "object_reference_chunks": 0,
        "duplicate_chunk_ids": 0,
        "validation_errors": [],
        "validation_warnings": [],
        "qdrant_status": "skipped" if args.skip_vector else "pending",
        "sqlite_status": "pending",
        "embedding_dimension": None,
        "readiness_status": "blocked",
    }

    try:
        ensure_safe_chunks_path(chunks_path)
        val = load_and_validate_chunks(chunks_path)
        report["total_chunks"] = len(val.chunks)
        report["chunks_by_type"] = val.chunks_by_type
        report["duplicate_chunk_ids"] = val.duplicate_chunk_ids
        report["validation_errors"] = val.errors
        report["validation_warnings"] = val.warnings
        if val.errors:
            report["sqlite_status"] = "blocked"
            write_json(report_path, report)
            print(f"[BLOCKED] Validation errors: {len(val.errors)}")
            return 1

        conn = setup_sqlite(sqlite_path, reset=args.reset)
        try:
            cur = conn.cursor()
            core_rows = []
            meta_rows = []
            keyword_rows = []
            object_rows = []
            vector_candidates = []

            for chunk in val.chunks:
                metadata = chunk.get("metadata") or {}
                routing = chunk.get("routing") or {}
                quality = chunk.get("quality") or {}
                provenance = chunk.get("provenance") or {}
                source_table_id = get_source_table_id(chunk)
                vector_index = int(bool(routing.get("vector_index")))
                keyword_index = int(bool(routing.get("keyword_index")))
                metadata_index = int(bool(routing.get("metadata_index", True)))

                qdrant_point_id = uuid_from_chunk_id(chunk["chunk_id"]) if vector_index else None
                core_rows.append(
                    (
                        chunk["chunk_id"],
                        qdrant_point_id,
                        chunk.get("doc_id"),
                        chunk.get("parent_chunk_id"),
                        chunk.get("chunk_type"),
                        chunk.get("modality"),
                        chunk.get("schema_version"),
                        chunk.get("text_for_embedding"),
                        chunk.get("text_for_keyword"),
                        vector_index,
                        keyword_index,
                        metadata_index,
                        1,
                        routing.get("priority"),
                    )
                )

                if not args.skip_metadata:
                    meta_rows.append(
                        (
                            chunk["chunk_id"],
                            chunk.get("doc_id"),
                            chunk.get("chunk_type"),
                            metadata.get("patient_token"),
                            metadata.get("sample_token"),
                            metadata.get("report_token"),
                            metadata.get("document_type"),
                            metadata.get("pdf_type"),
                            metadata.get("sample_type"),
                            metadata.get("sample_label"),
                            metadata.get("laboratory"),
                            metadata.get("department"),
                            metadata.get("organization"),
                            metadata.get("country"),
                            metadata.get("ministry"),
                            metadata.get("section"),
                            metadata.get("section_norm"),
                            metadata.get("analyte"),
                            metadata.get("analyte_norm"),
                            metadata.get("parameter"),
                            metadata.get("parameter_norm"),
                            metadata.get("value_raw"),
                            metadata.get("value_numeric"),
                            metadata.get("unit"),
                            metadata.get("unit_quality_status"),
                            metadata.get("reference_range"),
                            metadata.get("reference_quality_status"),
                            metadata.get("reference_complexity"),
                            metadata.get("result_kind"),
                            metadata.get("interpretation_status"),
                            metadata.get("result_quality_status"),
                            metadata.get("validation_status"),
                            metadata.get("age_consistency_status"),
                            metadata.get("patient_age"),
                            metadata.get("patient_sex"),
                            metadata.get("observation_date"),
                            metadata.get("report_date"),
                            metadata.get("request_date"),
                            metadata.get("received_date"),
                            metadata.get("validation_date"),
                            metadata.get("page_number"),
                            metadata.get("row_index"),
                            quality.get("confidence"),
                            quality.get("confidence_score", 0.70),
                            provenance.get("source_pdf") or metadata.get("source_pdf"),
                            source_table_id,
                            provenance.get("source_kind") or metadata.get("source_kind"),
                        )
                    )

                if keyword_index and not args.skip_keyword:
                    keyword_rows.append(
                        (
                            chunk["chunk_id"],
                            chunk.get("doc_id"),
                            chunk.get("chunk_type"),
                            chunk.get("text_for_keyword"),
                        )
                    )

                object_rows.append(
                    (
                        chunk["chunk_id"],
                        chunk.get("doc_id"),
                        provenance.get("source_pdf") or metadata.get("source_pdf"),
                        provenance.get("page_number") or metadata.get("page_number"),
                        source_table_id,
                        json.dumps(provenance.get("source_image_ids") or metadata.get("source_image_ids") or [], ensure_ascii=False),
                        json.dumps(provenance.get("source_table_ids") or metadata.get("source_table_ids") or [], ensure_ascii=False),
                        json.dumps(provenance.get("table_ids") or metadata.get("table_ids") or [], ensure_ascii=False),
                        json.dumps(provenance.get("image_ids") or metadata.get("image_ids") or [], ensure_ascii=False),
                        json.dumps(provenance.get("ocr_visual_ids") or metadata.get("ocr_visual_ids") or [], ensure_ascii=False),
                        json.dumps(provenance, ensure_ascii=False),
                    )
                )

                if vector_index and not args.skip_vector:
                    vector_candidates.append(chunk)

            with conn:
                cur.executemany(
                    """
                    INSERT OR REPLACE INTO chunks (
                      chunk_id, qdrant_point_id, doc_id, parent_chunk_id, chunk_type, modality, schema_version,
                      text_for_embedding, text_for_keyword, vector_index, keyword_index, metadata_index, object_store_reference, priority
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    core_rows,
                )
                if not args.skip_metadata:
                    cur.executemany(
                        """
                        INSERT OR REPLACE INTO metadata_chunks (
                          chunk_id, doc_id, chunk_type, patient_token, sample_token, report_token,
                          document_type, pdf_type, sample_type, sample_label, laboratory, department,
                          organization, country, ministry, section, section_norm, analyte, analyte_norm,
                          parameter, parameter_norm, value_raw, value_numeric, unit, unit_quality_status,
                          reference_range, reference_quality_status, reference_complexity, result_kind,
                          interpretation_status, result_quality_status, validation_status, age_consistency_status,
                          patient_age, patient_sex, observation_date, report_date, request_date, received_date,
                          validation_date, page_number, row_index, confidence, confidence_score, source_pdf,
                          source_table_id, source_kind
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        meta_rows,
                    )
                if not args.skip_keyword:
                    cur.execute("DELETE FROM keyword_chunks_fts")
                    cur.executemany(
                        "INSERT INTO keyword_chunks_fts (chunk_id, doc_id, chunk_type, text_for_keyword) VALUES (?, ?, ?, ?)",
                        keyword_rows,
                    )
                cur.executemany(
                    """
                    INSERT OR REPLACE INTO object_references (
                      chunk_id, doc_id, source_pdf, page_number, source_table_id,
                      source_image_ids_json, source_table_ids_json, table_ids_json,
                      image_ids_json, ocr_visual_ids_json, provenance_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    object_rows,
                )
            report["sqlite_status"] = "ok"

            if not args.skip_vector:
                model = EmbeddingModelWrapper(args.embedding_model, args.batch_size)
                report["embedding_backend"] = model.backend
                report["device"] = model.device

                from qdrant_client.http import models as qm

                qclient = build_qdrant_client(qdrant_dir)
                if args.reset:
                    try:
                        qclient.delete_collection(collection_name=args.collection)
                    except Exception:
                        pass

                vector_dim: int | None = None
                total = len(vector_candidates)
                iterator = iter_batches(vector_candidates, args.batch_size)
                if tqdm is not None:
                    iterator = tqdm(iterator, total=(total + args.batch_size - 1) // args.batch_size, desc="Vector indexing")

                for batch in iterator:
                    texts = [str(c.get("text_for_embedding") or "") for c in batch]
                    vectors = model.encode(texts)
                    if vectors.ndim == 1:
                        vectors = vectors.reshape(1, -1)
                    if vector_dim is None:
                        vector_dim = int(vectors.shape[1])
                        qclient.recreate_collection(
                            collection_name=args.collection,
                            vectors_config=qm.VectorParams(size=vector_dim, distance=qm.Distance.COSINE),
                        )
                    points = []
                    for idx, chunk in enumerate(batch):
                        metadata = chunk.get("metadata") or {}
                        provenance = chunk.get("provenance") or {}
                        quality = chunk.get("quality") or {}
                        payload = {
                            "chunk_id": chunk.get("chunk_id"),
                            "doc_id": chunk.get("doc_id"),
                            "chunk_type": chunk.get("chunk_type"),
                            "modality": chunk.get("modality"),
                            "parent_chunk_id": chunk.get("parent_chunk_id"),
                            "document_type": metadata.get("document_type"),
                            "sample_type": metadata.get("sample_type"),
                            "patient_token": metadata.get("patient_token"),
                            "sample_token": metadata.get("sample_token"),
                            "report_token": metadata.get("report_token"),
                            "analyte": metadata.get("analyte"),
                            "analyte_norm": metadata.get("analyte_norm"),
                            "parameter": metadata.get("parameter"),
                            "parameter_norm": metadata.get("parameter_norm"),
                            "section": metadata.get("section"),
                            "section_norm": metadata.get("section_norm"),
                            "value_raw": metadata.get("value_raw"),
                            "value_numeric": metadata.get("value_numeric"),
                            "unit": metadata.get("unit"),
                            "unit_quality_status": metadata.get("unit_quality_status"),
                            "interpretation_status": metadata.get("interpretation_status"),
                            "reference_quality_status": metadata.get("reference_quality_status"),
                            "reference_complexity": metadata.get("reference_complexity"),
                            "validation_status": metadata.get("validation_status"),
                            "age_consistency_status": metadata.get("age_consistency_status"),
                            "page_number": provenance.get("page_number") or metadata.get("page_number"),
                            "confidence_score": quality.get("confidence_score", 0.70),
                            "source_pdf": provenance.get("source_pdf") or metadata.get("source_pdf"),
                            "source_table_id": get_source_table_id(chunk),
                        }
                        points.append(
                            qm.PointStruct(
                                id=uuid_from_chunk_id(str(chunk["chunk_id"])),
                                vector=vectors[idx].tolist(),
                                payload=payload,
                            )
                        )
                    qclient.upsert(collection_name=args.collection, points=points, wait=True)

                report["qdrant_status"] = "ok"
                report["embedding_dimension"] = vector_dim
            else:
                report["embedding_backend"] = "skipped"
                report["device"] = "skipped"
                report["qdrant_status"] = "skipped"

            report["vector_indexed_chunks"] = len(vector_candidates)
            report["skipped_vector_chunks"] = len(val.chunks) - len(vector_candidates)
            report["keyword_indexed_chunks"] = len(keyword_rows)
            report["skipped_keyword_chunks"] = len(val.chunks) - len(keyword_rows)
            report["metadata_indexed_chunks"] = 0 if args.skip_metadata else len(meta_rows)
            report["object_reference_chunks"] = len(object_rows)
            report["readiness_status"] = "ready_for_retrieval"

            manifest = {
                "schema_version": "index_manifest_v1",
                "created_at": utc_now_iso(),
                "chunks_path": str(chunks_path),
                "index_dir": str(index_dir),
                "collection": args.collection,
                "embedding_model": args.embedding_model,
                "embedding_backend": report["embedding_backend"],
                "device": report["device"],
                "batch_size": args.batch_size,
                "vector_store": "qdrant_local",
                "keyword_store": "sqlite_fts5",
                "metadata_store": "sqlite",
                "total_chunks": len(val.chunks),
                "vector_indexed_chunks": report["vector_indexed_chunks"],
                "keyword_indexed_chunks": report["keyword_indexed_chunks"],
                "metadata_indexed_chunks": report["metadata_indexed_chunks"],
                "object_reference_chunks": report["object_reference_chunks"],
                "skipped_vector_chunks": report["skipped_vector_chunks"],
                "embedding_dimension": report["embedding_dimension"],
                "readiness_status": "ready",
            }

            write_json(manifest_path, manifest)
            write_json(report_path, report)

            print(f"Total chunks: {len(val.chunks)}")
            print(f"Vector chunks indexed: {report['vector_indexed_chunks']}")
            print(f"Keyword chunks indexed: {report['keyword_indexed_chunks']}")
            print(f"Metadata chunks indexed: {report['metadata_indexed_chunks']}")
            print(f"Skipped vector chunks: {report['skipped_vector_chunks']}")
            print(f"Embedding model: {args.embedding_model}")
            print(f"Embedding backend: {report['embedding_backend']}")
            print(f"Device: {report['device']}")
            print(f"Batch size: {args.batch_size}")
            print(f"SQLite: {sqlite_path}")
            print(f"Qdrant dir: {qdrant_dir}")
            print(f"Manifest: {manifest_path}")
            print(f"Indexing report: {report_path}")
            return 0
        finally:
            conn.close()
    except Exception as exc:
        report["validation_errors"] = report.get("validation_errors", []) + [str(exc)]
        report["readiness_status"] = "blocked"
        if report.get("sqlite_status") == "pending":
            report["sqlite_status"] = "blocked"
        write_json(report_path, report)
        print(f"[BLOCKED] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

