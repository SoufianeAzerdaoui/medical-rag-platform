from __future__ import annotations

import sqlite3
import re
from pathlib import Path
from typing import Any

try:
    from .filters import RetrievalFilters, build_sql_filter_clauses
except ImportError:
    from filters import RetrievalFilters, build_sql_filter_clauses


class SQLiteStore:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite DB missing: {self.db_path}")

        self.conn = self._connect_readonly(self.db_path)
        self.conn.row_factory = sqlite3.Row

        self.tables = self._load_tables()
        self.fts_table = self._detect_fts_table()
        self.chunks_table = self._detect_table("chunks")
        self.metadata_table = self._detect_table("metadata_chunks")
        self.object_table = self._detect_table("object_references")
        self._ensure_non_empty_chunks()

    @staticmethod
    def _connect_readonly(db_path: Path) -> sqlite3.Connection:
        uri = f"file:{db_path.resolve()}?mode=ro"
        try:
            return sqlite3.connect(uri, uri=True)
        except sqlite3.OperationalError:
            # Fallback for environments where URI read-only may be restricted
            return sqlite3.connect(str(db_path.resolve()))

    def close(self) -> None:
        self.conn.close()

    def _load_tables(self) -> dict[str, dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT name, type, sql FROM sqlite_master WHERE type IN ('table','view')")
        out: dict[str, dict[str, Any]] = {}
        for row in cur.fetchall():
            out[row["name"]] = {
                "type": row["type"],
                "sql": row["sql"] or "",
            }
        return out

    def _detect_table(self, preferred_name: str) -> str:
        if preferred_name in self.tables:
            return preferred_name
        available = sorted(self.tables.keys())
        raise RuntimeError(
            f"Table '{preferred_name}' not found. Available tables: {available}"
        )

    def _detect_fts_table(self) -> str:
        for name, meta in self.tables.items():
            sql = (meta.get("sql") or "").lower()
            if "virtual table" in sql and "fts5" in sql:
                return name
        available = sorted(self.tables.keys())
        raise RuntimeError(
            f"FTS5 table not detected in SQLite. Available tables: {available}"
        )

    def available_tables(self) -> list[str]:
        return sorted(self.tables.keys())

    def _ensure_non_empty_chunks(self) -> None:
        cur = self.conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {self.chunks_table}")
        count = int(cur.fetchone()[0])
        if count <= 0:
            raise RuntimeError("No chunks found in SQLite chunks table")

    def keyword_search_rows(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters,
    ) -> list[dict[str, Any]]:
        sql_filters, params = build_sql_filter_clauses(filters)
        where_filters = ""
        if sql_filters:
            where_filters = " AND " + " AND ".join(sql_filters)

        sql = f"""
        SELECT
          f.chunk_id AS chunk_id,
          bm25({self.fts_table}) AS bm25_score,
          c.doc_id AS doc_id,
          c.chunk_type AS chunk_type,
          c.parent_chunk_id AS parent_chunk_id,
          c.text_for_embedding AS text_for_embedding,
          c.text_for_keyword AS text_for_keyword,
          m.document_type AS document_type,
          m.sample_type AS sample_type,
          m.request_date AS request_date,
          m.patient_token AS patient_token,
          m.sample_token AS sample_token,
          m.report_token AS report_token,
          m.validation_status AS validation_status,
          m.reference_quality_status AS reference_quality_status,
          m.age_consistency_status AS age_consistency_status,
          m.section AS section,
          m.section_norm AS section_norm,
          m.analyte AS analyte,
          m.value_raw AS value_raw,
          m.unit AS unit,
          COALESCE(m.source_pdf, o.source_pdf) AS source_pdf,
          COALESCE(m.page_number, o.page_number) AS page_number
        FROM {self.fts_table} f
        JOIN {self.chunks_table} c ON c.chunk_id = f.chunk_id
        LEFT JOIN {self.metadata_table} m ON m.chunk_id = c.chunk_id
        LEFT JOIN {self.object_table} o ON o.chunk_id = c.chunk_id
        WHERE {self.fts_table} MATCH ?
        {where_filters}
        ORDER BY bm25({self.fts_table}) ASC
        LIMIT ?
        """

        cur = self.conn.cursor()
        try:
            cur.execute(sql, [query, *params, int(top_k)])
        except sqlite3.OperationalError as exc:
            if "fts5" not in str(exc).lower():
                raise
            safe_query = self._to_safe_fts_query(query)
            cur.execute(sql, [safe_query, *params, int(top_k)])
        rows = [dict(r) for r in cur.fetchall()]
        return rows

    @staticmethod
    def _to_safe_fts_query(query: str) -> str:
        # Conservative fallback for user-friendly queries that contain FTS operators/punctuation
        tokens = [t for t in re.split(r"[^\\w]+", query, flags=re.UNICODE) if t]
        if not tokens:
            return f'\"{query.strip()}\"'
        return " AND ".join(f'\"{t}\"' for t in tokens)

    def get_chunk_rows_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}

        placeholders = ",".join(["?"] * len(chunk_ids))
        sql = f"""
        SELECT
          c.chunk_id,
          c.doc_id,
          c.chunk_type,
          c.parent_chunk_id,
          c.text_for_embedding,
          c.text_for_keyword,
          c.vector_index,
          c.keyword_index,
          c.metadata_index,
          c.priority,
          m.*,
          o.source_pdf AS object_source_pdf,
          o.page_number AS object_page_number,
          o.source_table_id AS object_source_table_id,
          o.provenance_json
        FROM {self.chunks_table} c
        LEFT JOIN {self.metadata_table} m ON m.chunk_id = c.chunk_id
        LEFT JOIN {self.object_table} o ON o.chunk_id = c.chunk_id
        WHERE c.chunk_id IN ({placeholders})
        """
        cur = self.conn.cursor()
        cur.execute(sql, chunk_ids)
        rows = [dict(r) for r in cur.fetchall()]

        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            out[str(row["chunk_id"])] = row
        return out

    def get_doc_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        sql = f"""
        SELECT
          c.chunk_id,
          c.doc_id,
          c.chunk_type,
          c.parent_chunk_id,
          c.text_for_embedding,
          c.text_for_keyword,
          c.priority,
          m.*,
          o.source_pdf AS object_source_pdf,
          o.page_number AS object_page_number,
          o.provenance_json
        FROM {self.chunks_table} c
        LEFT JOIN {self.metadata_table} m ON m.chunk_id = c.chunk_id
        LEFT JOIN {self.object_table} o ON o.chunk_id = c.chunk_id
        WHERE c.doc_id = ?
        """
        cur = self.conn.cursor()
        cur.execute(sql, (doc_id,))
        return [dict(r) for r in cur.fetchall()]

    def get_all_rows_for_filtering(self, limit: int) -> list[dict[str, Any]]:
        sql = f"""
        SELECT
          c.chunk_id,
          c.doc_id,
          c.chunk_type,
          c.parent_chunk_id,
          c.text_for_embedding,
          c.text_for_keyword,
          m.*,
          COALESCE(m.source_pdf, o.source_pdf) AS source_pdf,
          COALESCE(m.page_number, o.page_number) AS page_number
        FROM {self.chunks_table} c
        LEFT JOIN {self.metadata_table} m ON m.chunk_id = c.chunk_id
        LEFT JOIN {self.object_table} o ON o.chunk_id = c.chunk_id
        LIMIT ?
        """
        cur = self.conn.cursor()
        cur.execute(sql, (int(limit),))
        return [dict(r) for r in cur.fetchall()]
