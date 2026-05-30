from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.ingestion_service import _replace_doc_rows


class TestIngestionDocChunkMapping(unittest.TestCase):
    def test_replace_doc_rows_preserves_other_docs_and_chunk_ids(self) -> None:
        existing = [
            {"doc_id": "report_10", "chunk_id": "chk_report_10_a", "text": "old a"},
            {"doc_id": "report_10", "chunk_id": "chk_report_10_b", "text": "old b"},
            {"doc_id": "report_12", "chunk_id": "chk_report_12_a", "text": "keep"},
        ]
        new_rows = [
            {"doc_id": "report_10", "chunk_id": "chk_report_10_a", "text": "new a"},
            {"doc_id": "report_10", "chunk_id": "chk_report_10_c", "text": "new c"},
        ]

        merged = _replace_doc_rows(existing, new_rows, "report_10")

        self.assertEqual(
            {row["chunk_id"] for row in merged},
            {"chk_report_10_a", "chk_report_10_c", "chk_report_12_a"},
        )
        kept_report12 = [row for row in merged if row.get("doc_id") == "report_12"]
        self.assertEqual(len(kept_report12), 1)
        self.assertEqual(kept_report12[0]["chunk_id"], "chk_report_12_a")

    def test_replace_doc_rows_removes_old_doc_specific_chunks(self) -> None:
        existing = [
            {"doc_id": "report_16", "chunk_id": "chk_1"},
            {"doc_id": "report_16", "chunk_id": "chk_2"},
        ]
        new_rows = [{"doc_id": "report_16", "chunk_id": "chk_3"}]
        merged = _replace_doc_rows(existing, new_rows, "report_16")
        self.assertEqual([row["chunk_id"] for row in merged], ["chk_3"])


if __name__ == "__main__":
    unittest.main()
