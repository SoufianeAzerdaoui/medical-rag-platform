from __future__ import annotations

import time
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services import ingestion_service


class TestIngestionBatchLoad(unittest.TestCase):
    def test_validate_docs_selection_handles_large_batch_fast(self) -> None:
        candidates = []
        for i in range(1, 1501):
            filename = f"report ({i}).pdf"
            candidates.append(
                ingestion_service.DocsPdfCandidate(
                    filename=filename,
                    doc_id=f"report_{i}",
                    absolute_path=f"/tmp/{filename}",
                    size_bytes=1024,
                    modified_at="2026-05-30T10:00:00Z",
                    file_hash=f"hash-{i}",
                    text_hash=f"texthash-{i}",
                    already_indexed=False,
                    is_duplicate=False,
                    duplicate_with=[],
                    duplicate_reason=None,
                    blocked=False,
                    registry_status="discovered",
                    first_seen_at=None,
                    last_seen_at=None,
                    last_ingested_at=None,
                    last_error=None,
                    duplicate_entries=[],
                    duplicate_override=False,
                    override_reason=None,
                    override_by=None,
                    override_at=None,
                )
            )

        filenames = [f"report ({i}).pdf" for i in range(1, 1501)] + [f"report ({i}).pdf" for i in range(1, 200)]
        start = time.perf_counter()
        with patch("backend.services.ingestion_service.discover_docs_pdfs", return_value=candidates):
            out = ingestion_service.validate_docs_selection(filenames)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(out), 1500)
        self.assertLess(elapsed, 1.5, f"Batch validation too slow: {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main()
