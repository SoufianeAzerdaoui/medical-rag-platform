from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "extraction_data"
if str(EXTRACTION_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_SCRIPT_ROOT))

from utils import parse_reference_range


class TestReferenceRangeParsing(unittest.TestCase):
    def test_multi_interval_multi_unit(self) -> None:
        reference = parse_reference_range("(15,00 - 65,00) pg/ml(1.6-6.9 pmol/l)")
        self.assertEqual(reference.get("text"), "(15,00 - 65,00) pg/ml(1.6-6.9 pmol/l)")
        self.assertEqual(reference.get("operator"), "multiple")
        self.assertIsNone(reference.get("separator"))
        self.assertIsNone(reference.get("unit"))
        self.assertIsNone(reference.get("low"))
        self.assertIsNone(reference.get("high"))
        self.assertEqual(reference.get("complexity"), "multi_interval_multi_unit")
        ranges = reference.get("reference_ranges")
        self.assertIsInstance(ranges, list)
        self.assertEqual(len(ranges), 2)
        self.assertEqual((ranges[0] or {}).get("text"), "(15,00 - 65,00) pg/ml")
        self.assertEqual((ranges[0] or {}).get("operator"), "between")
        self.assertEqual((ranges[0] or {}).get("separator"), "-")
        self.assertEqual((ranges[0] or {}).get("unit"), "pg/ml")
        self.assertEqual((ranges[0] or {}).get("low"), 15.0)
        self.assertEqual((ranges[0] or {}).get("high"), 65.0)
        self.assertEqual((ranges[1] or {}).get("text"), "(1.6-6.9 pmol/l)")
        self.assertEqual((ranges[1] or {}).get("operator"), "between")
        self.assertEqual((ranges[1] or {}).get("separator"), "-")
        self.assertEqual((ranges[1] or {}).get("unit"), "pmol/l")
        self.assertEqual((ranges[1] or {}).get("low"), 1.6)
        self.assertEqual((ranges[1] or {}).get("high"), 6.9)

    def test_between_with_et_separator(self) -> None:
        reference = parse_reference_range("0,78 et 5,19 pmol/l")
        self.assertEqual(reference.get("text"), "0,78 et 5,19 pmol/l")
        self.assertEqual(reference.get("operator"), "between")
        self.assertEqual(reference.get("separator"), "et")
        self.assertEqual(reference.get("unit"), "pmol/l")
        self.assertEqual(reference.get("low"), 0.78)
        self.assertEqual(reference.get("high"), 5.19)

    def test_comparator_with_unit(self) -> None:
        reference = parse_reference_range("< 11,80 pg/ml")
        self.assertEqual(reference.get("text"), "< 11,80 pg/ml")
        self.assertEqual(reference.get("operator"), "<")
        self.assertEqual(reference.get("unit"), "pg/ml")
        self.assertIsNone(reference.get("low"))
        self.assertEqual(reference.get("high"), 11.8)

    def test_comparator_without_space(self) -> None:
        reference = parse_reference_range("<4,11 IU/ml")
        self.assertEqual(reference.get("text"), "<4,11 IU/ml")
        self.assertEqual(reference.get("operator"), "<")
        self.assertEqual(reference.get("unit"), "IU/ml")
        self.assertIsNone(reference.get("low"))
        self.assertEqual(reference.get("high"), 4.11)

    def test_between_with_a_separator_and_micro_unit(self) -> None:
        reference = parse_reference_range("4 à 20 µIU/mL")
        self.assertEqual(reference.get("text"), "4 à 20 µIU/mL")
        self.assertEqual(reference.get("operator"), "between")
        self.assertEqual(reference.get("separator"), "à")
        self.assertEqual(reference.get("unit"), "µIU/mL")
        self.assertEqual(reference.get("low"), 4.0)
        self.assertEqual(reference.get("high"), 20.0)

    def test_between_with_parenthesized_dash(self) -> None:
        reference = parse_reference_range("(14,90 - 56,90) pg/ml")
        self.assertEqual(reference.get("text"), "(14,90 - 56,90) pg/ml")
        self.assertEqual(reference.get("operator"), "between")
        self.assertEqual(reference.get("separator"), "-")
        self.assertEqual(reference.get("unit"), "pg/ml")
        self.assertEqual(reference.get("low"), 14.9)
        self.assertEqual(reference.get("high"), 56.9)
        self.assertIsNone(reference.get("complexity"))


if __name__ == "__main__":
    unittest.main()
