from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
GENERATION_ROOT = SCRIPTS_ROOT / "generation"
for root in (SCRIPTS_ROOT, GENERATION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from filtering_utils import dedup_evidences, row_matches_value_criterion


class TestEvidenceFiltering(unittest.TestCase):
    def test_acth_gte_uses_current_value_only(self) -> None:
        row_current_23_prev_111 = {"value_numeric": 23.0, "value_raw": "23,00", "previous_result_value_raw": "1,11"}
        row_current_111_prev_23 = {"value_numeric": 1.11, "value_raw": "1,11", "previous_result_value_raw": "23,00"}
        self.assertTrue(row_matches_value_criterion(row_current_23_prev_111, ["23"], ">="))
        self.assertFalse(row_matches_value_criterion(row_current_111_prev_23, ["23"], ">="))

    def test_acth_strict_gt_excludes_23(self) -> None:
        row_current_23 = {"value_numeric": 23.0, "value_raw": "23,00"}
        row_current_111 = {"value_numeric": 1.11, "value_raw": "1,11"}
        self.assertFalse(row_matches_value_criterion(row_current_23, ["23"], ">"))
        self.assertFalse(row_matches_value_criterion(row_current_111, ["23"], ">"))

    def test_dedup_tshus_identical_rows(self) -> None:
        rows = [
            {
                "doc_id": "report_16",
                "analyte": "TSHus",
                "analyte_norm": "tshus",
                "patient_token": "PAT_000002",
                "sample_token": "SAMPLE_1",
                "value_raw": "55,00",
                "value_numeric": 55.0,
                "unit": "mUI/L",
                "page_number": 1,
            },
            {
                "doc_id": "report_16",
                "analyte": "TSHus",
                "analyte_norm": "tshus",
                "patient_token": "PAT_000002",
                "sample_token": "SAMPLE_1",
                "value_raw": "55,00",
                "value_numeric": 55.0,
                "unit": "mUI/L",
                "page_number": 1,
            },
        ]
        evidences = dedup_evidences(rows)
        self.assertEqual(len(evidences), 1)

    def test_tshus_not_dedup_when_source_differs(self) -> None:
        rows = [
            {
                "doc_id": "report_16",
                "analyte": "TSHus",
                "analyte_norm": "tshus",
                "patient_token": "PAT_000002",
                "sample_token": "SAMPLE_1",
                "value_raw": "2,00",
                "value_numeric": 2.0,
                "unit": "mUI/L",
                "page_number": 1,
            },
            {
                "doc_id": "report_17",
                "analyte": "TSHus",
                "analyte_norm": "tshus",
                "patient_token": "PAT_000002",
                "sample_token": "SAMPLE_1",
                "value_raw": "2,00",
                "value_numeric": 2.0,
                "unit": "mUI/L",
                "page_number": 1,
            },
        ]
        evidences = dedup_evidences(rows)
        self.assertEqual(len(evidences), 2)

    def test_previous_value_never_matches_threshold(self) -> None:
        row_current_111_prev_23 = {
            "value_numeric": 1.11,
            "value_raw": "1,11",
            "previous_result_value_raw": "23,00",
            "previous_value_numeric": 23.0,
        }
        self.assertFalse(row_matches_value_criterion(row_current_111_prev_23, ["23"], ">="))


if __name__ == "__main__":
    unittest.main()
