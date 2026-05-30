from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend import config
from backend.database import init_schema
from backend.services import ingestion_service


def indexed_doc_ids_from_sqlite(sqlite_path: Path) -> set[str]:
    if not sqlite_path.exists():
        return set()
    out: set[str] = set()
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        for table in ("metadata_chunks", "chunks", "object_references"):
            try:
                cur.execute(
                    f"SELECT DISTINCT lower(doc_id) AS doc_id FROM {table} "
                    "WHERE doc_id IS NOT NULL AND trim(doc_id) != ''"
                )
            except Exception:
                continue
            for row in cur.fetchall():
                doc_id = str(row["doc_id"] or "").strip().lower()
                if doc_id:
                    out.add(doc_id)
    finally:
        conn.close()
    return out


def main() -> int:
    init_schema()
    sqlite_path = config.ROOT_DIR / "data" / "indexes" / "medical_rag.sqlite"
    indexed_ids = indexed_doc_ids_from_sqlite(sqlite_path)
    result = ingestion_service.resync_docs_registry(indexed_doc_ids=indexed_ids)
    print(json.dumps({"success": True, **result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
