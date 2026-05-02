from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKING_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "chunking"
if str(CHUNKING_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(CHUNKING_SCRIPT_ROOT))

from build_chunks import build_result_chunk


class TestChunkReferenceMetadata(unittest.TestCase):
    def _base_doc(self) -> dict:
        return {
            "doc_id": "report_test",
            "source_pdf": "docs/report_test.pdf",
            "document_type": "biology_report",
            "facility": {},
            "patient": {},
            "report": {"sample_type": "SANG", "report_date": "2024-07-19"},
            "validation": {},
            "validation_report": {},
        }

    def test_comparator_reference_fields_are_flattened(self) -> None:
        chunk = build_result_chunk(
            self._base_doc(),
            {
                "analyte": "CALCITONINE",
                "value_raw": "3,00",
                "value_numeric": 3.0,
                "unit": "pg/mL",
                "result_kind": "numeric",
                "row_index": 10,
                "page_number": 1,
                "source_kind": "chu_text_fallback",
                "reference_range": {
                    "text": "< 11,80 pg/ml",
                    "operator": "<",
                    "separator": None,
                    "unit": "pg/ml",
                    "low": None,
                    "high": 11.8,
                },
            },
            Path("data/extraction/report_test/document.json"),
        )

        metadata = chunk["metadata"]
        self.assertEqual(metadata.get("reference_range"), "< 11,80 pg/ml")
        self.assertEqual(metadata.get("reference_range_text"), "< 11,80 pg/ml")
        self.assertEqual(metadata.get("reference_operator"), "<")
        self.assertIsNone(metadata.get("reference_separator"))
        self.assertEqual(metadata.get("reference_unit"), "pg/ml")
        self.assertIsNone(metadata.get("reference_low"))
        self.assertEqual(metadata.get("reference_high"), 11.8)

    def test_between_reference_fields_are_flattened(self) -> None:
        chunk = build_result_chunk(
            self._base_doc(),
            {
                "analyte": "PEPTIDE C",
                "value_raw": "3,00",
                "value_numeric": 3.0,
                "unit": "ng/ml",
                "result_kind": "numeric",
                "row_index": 6,
                "page_number": 1,
                "source_kind": "chu_text_fallback",
                "reference_range": {
                    "text": "0,78 et 5,19 pmol/l",
                    "operator": "between",
                    "separator": "et",
                    "unit": "pmol/l",
                    "low": 0.78,
                    "high": 5.19,
                },
            },
            Path("data/extraction/report_test/document.json"),
        )

        metadata = chunk["metadata"]
        self.assertEqual(metadata.get("reference_range"), "0,78 et 5,19 pmol/l")
        self.assertEqual(metadata.get("reference_range_text"), "0,78 et 5,19 pmol/l")
        self.assertEqual(metadata.get("reference_operator"), "between")
        self.assertEqual(metadata.get("reference_separator"), "et")
        self.assertEqual(metadata.get("reference_unit"), "pmol/l")
        self.assertEqual(metadata.get("reference_low"), 0.78)
        self.assertEqual(metadata.get("reference_high"), 5.19)


if __name__ == "__main__":
    unittest.main()
