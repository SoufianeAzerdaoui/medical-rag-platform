#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import (
    INDEX_VALIDATION_REPORT_FILENAME,
    QDRANT_DIRNAME,
    SMOKE_TERMS,
    SQLITE_FILENAME,
    resolve_repo_root,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def uuid_from_chunk_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate built medical RAG indexes.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    index_dir = (repo_root / args.index_dir).resolve() if not Path(args.index_dir).is_absolute() else Path(args.index_dir).resolve()
    sqlite_path = index_dir / SQLITE_FILENAME
    qdrant_dir = index_dir / QDRANT_DIRNAME
    report_path = index_dir / INDEX_VALIDATION_REPORT_FILENAME

    report: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "sqlite_status": "pending",
        "qdrant_status": "pending",
        "total_chunks": 0,
        "vector_points": 0,
        "keyword_rows": 0,
        "metadata_rows": 0,
        "validation_errors": [],
        "validation_warnings": [],
        "smoke_test_results": {},
        "readiness_status": "blocked",
    }

    try:
        if not sqlite_path.exists():
            raise RuntimeError(f"SQLite missing: {sqlite_path}")

        conn = sqlite3.connect(str(sqlite_path))
        try:
            cur = conn.cursor()
            required_tables = {"chunks", "metadata_chunks", "keyword_chunks_fts", "object_references"}
            cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
            table_names = {row[0] for row in cur.fetchall()}
            missing = required_tables.difference(table_names)
            if missing:
                raise RuntimeError(f"Tables SQLite manquantes: {sorted(missing)}")

            cur.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM metadata_chunks")
            metadata_rows = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM keyword_chunks_fts")
            keyword_rows = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM chunks WHERE vector_index = 1")
            vector_expected = int(cur.fetchone()[0])

            report["total_chunks"] = total_chunks
            report["metadata_rows"] = metadata_rows
            report["keyword_rows"] = keyword_rows

            if total_chunks <= 0:
                raise RuntimeError("Table chunks vide")
            if metadata_rows != total_chunks:
                raise RuntimeError("metadata_chunks != chunks")
            if keyword_rows <= 0:
                raise RuntimeError("keyword_chunks_fts vide")

            report["sqlite_status"] = "ok"

            if not qdrant_dir.exists():
                raise RuntimeError(f"Dossier Qdrant manquant: {qdrant_dir}")
            from qdrant_client import QdrantClient

            qclient = QdrantClient(path=str(qdrant_dir))
            collections = [c.name for c in qclient.get_collections().collections]
            if args.collection not in collections:
                raise RuntimeError(f"Collection Qdrant absente: {args.collection}")

            count_resp = qclient.count(collection_name=args.collection, exact=True)
            vector_points = int(count_resp.count)
            report["vector_points"] = vector_points

            if vector_points <= 0:
                raise RuntimeError("Aucun point vectoriel dans Qdrant")
            if vector_points != vector_expected:
                raise RuntimeError(
                    f"Incoherence Qdrant/SQLite: qdrant={vector_points} vs chunks.vector_index=1 ({vector_expected})"
                )

            cur.execute("SELECT chunk_id, keyword_index, vector_index FROM chunks LIMIT 20")
            sample_rows = cur.fetchall()
            for chunk_id, keyword_index, vector_index in sample_rows:
                cur.execute("SELECT 1 FROM metadata_chunks WHERE chunk_id = ? LIMIT 1", (chunk_id,))
                if cur.fetchone() is None:
                    raise RuntimeError(f"Chunk sample sans metadata: {chunk_id}")
                if int(keyword_index) == 1:
                    cur.execute("SELECT 1 FROM keyword_chunks_fts WHERE chunk_id = ? LIMIT 1", (chunk_id,))
                    if cur.fetchone() is None:
                        raise RuntimeError(f"Chunk keyword manquant dans FTS: {chunk_id}")
                if int(vector_index) == 1:
                    pid = uuid_from_chunk_id(str(chunk_id))
                    points = qclient.retrieve(collection_name=args.collection, ids=[pid], with_payload=False, with_vectors=False)
                    if not points:
                        raise RuntimeError(f"Point Qdrant manquant pour chunk_id={chunk_id}")

            technical_patterns = [
                r"\bresults_table_block\b",
                r"\bpatient_info_block\b",
                r"\bvalidation_block\b",
                r"\bfacility_block\b",
                r"\bdocument_header\b",
                r"\bfooter_block\b",
            ]
            cur.execute("SELECT chunk_id, doc_id, text_for_embedding, text_for_keyword FROM chunks")
            contaminated = []
            for chunk_id, doc_id, text_emb, text_kw in cur.fetchall():
                blob = f"{text_emb or ''} {text_kw or ''}"
                for pattern in technical_patterns:
                    if re.search(pattern, blob, flags=re.IGNORECASE):
                        contaminated.append(
                            {
                                "chunk_id": chunk_id,
                                "doc_id": doc_id,
                                "pattern": pattern,
                            }
                        )
                        break
            if contaminated:
                preview = contaminated[:10]
                report["validation_errors"].append(
                    f"Technical block labels found in indexed text ({len(contaminated)} chunks). Preview: {preview}"
                )
                raise RuntimeError("Indexed text contains technical block labels")

            if args.smoke_test:
                smoke_results: dict[str, list[dict[str, Any]]] = {}
                for term in SMOKE_TERMS:
                    cur.execute(
                        """
                        SELECT chunk_id, doc_id, chunk_type, substr(text_for_keyword, 1, 200)
                        FROM keyword_chunks_fts
                        WHERE keyword_chunks_fts MATCH ?
                        LIMIT 5
                        """,
                        (term,),
                    )
                    rows = cur.fetchall()
                    smoke_results[term] = [
                        {
                            "chunk_id": r[0],
                            "doc_id": r[1],
                            "chunk_type": r[2],
                            "text_excerpt": r[3],
                        }
                        for r in rows
                    ]
                report["smoke_test_results"] = smoke_results
                for term, rows in smoke_results.items():
                    print(f"[SMOKE] {term} => {len(rows)} result(s)")
                    for row in rows:
                        print(
                            f"  - {row['chunk_id']} | {row['doc_id']} | {row['chunk_type']} | {row['text_excerpt']}"
                        )

            report["qdrant_status"] = "ok"
            report["readiness_status"] = "ok"
            write_json(report_path, report)
            print(f"Validation report: {report_path}")
            return 0
        finally:
            conn.close()
    except Exception as exc:
        report["validation_errors"].append(str(exc))
        report["sqlite_status"] = report["sqlite_status"] if report["sqlite_status"] != "pending" else "blocked"
        report["qdrant_status"] = report["qdrant_status"] if report["qdrant_status"] != "pending" else "blocked"
        report["readiness_status"] = "blocked"
        write_json(report_path, report)
        print(f"[BLOCKED] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
