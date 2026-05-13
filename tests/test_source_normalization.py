from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from source_normalization import dedup_normalized_sources, normalize_source_for_response


class TestSourceNormalization(unittest.TestCase):
    def test_viewer_url_is_preferred_for_clickable_source(self) -> None:
        src = {
            "doc_id": "report_18",
            "source_pdf": "report (18).pdf",
            "page": 1,
            "line": 1,
            "viewer_url": "/viewer/pdf?doc_id=report_18&page=1",
            "source_url": "/api/documents/report_18/pdf?page=1",
        }
        out = normalize_source_for_response(src)
        self.assertTrue(out["is_clickable"])
        self.assertEqual(out["url"], "/viewer/pdf?doc_id=report_18&page=1")

    def test_doc_id_can_build_clickable_urls(self) -> None:
        src = {"doc_id": "report_18", "source_pdf": "report (18).pdf", "page": 1, "line": 1}
        out = normalize_source_for_response(src)
        self.assertTrue(out["is_clickable"])
        self.assertTrue(str(out.get("viewer_url") or "").startswith("/viewer/pdf?doc_id=report_18"))
        self.assertIn("report (18).pdf", str(out.get("label") or ""))

    def test_dedup_by_source_page_line_and_url(self) -> None:
        sources = [
            {"doc_id": "report_18", "source_pdf": "report (18).pdf", "page": 1, "line": 1},
            {"doc_id": "report_18", "source_pdf": "report (18).pdf", "page": 1, "line": 1},
            {"doc_id": "report_18", "source_pdf": "report (18).pdf", "page": 1, "line": 2},
        ]
        out = dedup_normalized_sources(sources)
        self.assertEqual(len(out), 2)

    def test_docs_prefix_is_normalized(self) -> None:
        src = {"doc_id": "report_18", "source_pdf": "docs/report (18).pdf", "label": "docs/report (18).pdf — page 1, ligne 1", "page": 1, "line": 1}
        out = normalize_source_for_response(src)
        self.assertEqual(out.get("source_pdf"), "report (18).pdf")
        self.assertTrue(str(out.get("label") or "").startswith("report (18).pdf"))


if __name__ == "__main__":
    unittest.main()
