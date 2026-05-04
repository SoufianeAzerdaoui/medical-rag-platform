from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "extraction_data"
if str(EXTRACTION_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTRACTION_SCRIPT_ROOT))

from structure_clinical import extract_chu_lab_results


def _make_chu_pages(lines: list[str]) -> list[dict]:
    text = "\n".join(lines)
    return [
        {
            "page_number": 1,
            "width": 1000,
            "height": 1400,
            "native_text": text,
            "final_text": text,
        }
    ]


class TestChuColumnAwareResults(unittest.TestCase):
    def test_column_aware_mapping_current_reference_previous(self) -> None:
        pages = _make_chu_pages(
            [
                "LABORATOIRE CENTRAL",
                "IP Patient : 53",
                "Date Demande: 04/06/2024 12:59:47",
                "Paramétres",
                "AMPHÉTAMINE QUALITATIF",
                "20,00",
                "<500",
                "001 ng/mL",
                "AMPHÉTAMINE QUALITATIF INDICE",
                "18,00",
                "002",
                "AMPHÉTAMINE SEMI-QUANTITATIF",
                "20,00",
                "<200",
                "3,00 ng/ml",
                "BENZODIAZÉPINE QUALITATIF",
                "11,00",
                "<200",
                "4 ng/mL",
                "COCAÏNE QUALITATIF",
                "<150",
                "7,00",
                "Page 1 sur 1",
            ]
        )

        results, _raw, _stats = extract_chu_lab_results(pages)
        by_analyte = {row["analyte"]: row for row in results}

        amp_q = by_analyte["AMPHÉTAMINE QUALITATIF"]
        self.assertEqual(amp_q["value_raw"], "001")
        self.assertEqual(amp_q["unit"], "ng/mL")
        self.assertEqual((amp_q["reference_range"] or {}).get("text"), "<500")
        self.assertIsNone((amp_q["reference_range"] or {}).get("low"))
        self.assertEqual((amp_q["reference_range"] or {}).get("high"), 500.0)
        self.assertEqual((amp_q["previous_result"] or {}).get("value_raw"), "20,00")
        self.assertEqual((amp_q["previous_result"] or {}).get("value_numeric"), 20.0)
        self.assertEqual((amp_q["previous_result"] or {}).get("unit"), None)
        self.assertEqual(amp_q["row_index"], 1)
        self.assertTrue(amp_q["is_canonical"])

        amp_i = by_analyte["AMPHÉTAMINE QUALITATIF INDICE"]
        self.assertEqual(amp_i["value_raw"], "002")
        self.assertEqual(amp_i["unit"], "unknown")
        self.assertEqual((amp_i["reference_range"] or {}).get("text"), None)
        self.assertEqual((amp_i["previous_result"] or {}).get("value_raw"), "18,00")
        self.assertEqual((amp_i["previous_result"] or {}).get("value_numeric"), 18.0)

        amp_sq = by_analyte["AMPHÉTAMINE SEMI-QUANTITATIF"]
        self.assertEqual(amp_sq["value_raw"], "3,00")
        self.assertEqual(amp_sq["unit"], "ng/ml")
        self.assertEqual((amp_sq["reference_range"] or {}).get("text"), "<200")
        self.assertEqual((amp_sq["reference_range"] or {}).get("high"), 200.0)
        self.assertEqual((amp_sq["previous_result"] or {}).get("value_raw"), "20,00")

        benzo_q = by_analyte["BENZODIAZÉPINE QUALITATIF"]
        self.assertEqual(benzo_q["value_raw"], "4")
        self.assertEqual(benzo_q["unit"], "ng/mL")
        self.assertEqual((benzo_q["reference_range"] or {}).get("text"), "<200")
        self.assertEqual((benzo_q["previous_result"] or {}).get("value_raw"), "11,00")

        # Missing previous result: keep reference + current only.
        coc_q = by_analyte["COCAÏNE QUALITATIF"]
        self.assertEqual(coc_q["value_raw"], "7,00")
        self.assertEqual((coc_q["reference_range"] or {}).get("text"), "<150")
        self.assertIsNone(coc_q.get("previous_result"))

    def test_qualitative_fallback_still_works(self) -> None:
        pages = _make_chu_pages(
            [
                "LABORATOIRE CENTRAL",
                "IP Patient : 53",
                "Paramétres",
                "DÉPISTAGE QUALITATIF",
                "Positif",
                "Page 1 sur 1",
            ]
        )
        results, _raw, _stats = extract_chu_lab_results(pages)
        self.assertEqual(len(results), 1)
        row = results[0]
        self.assertEqual(row["analyte"], "DÉPISTAGE QUALITATIF")
        self.assertEqual(row["value_raw"], "Positif")
        self.assertEqual(row["unit"], "qualitative")
        self.assertEqual((row["reference_range"] or {}).get("text"), "Qualitatif")
        self.assertIsNone(row.get("previous_result"))

    def test_multi_profile_reference_patterns_and_previous_optional(self) -> None:
        cases = [
            (
                "toxicology_with_previous",
                [
                    "LABORATOIRE CENTRAL",
                    "Date Demande: 04/06/2024 12:59:47",
                    "Paramétres",
                    "AMPHÉTAMINE QUALITATIF",
                    "20,00",
                    "<500",
                    "001 ng/mL",
                    "Page 1 sur 1",
                ],
                {
                    "analyte": "AMPHÉTAMINE QUALITATIF",
                    "value_raw": "001",
                    "unit": "ng/mL",
                    "reference": "<500",
                    "previous": "20,00",
                },
            ),
            (
                "biochimie_range_30_a_40_no_previous",
                [
                    "LABORATOIRE CENTRAL",
                    "Date Demande: 01/03/2025 08:00:00",
                    "Paramétres",
                    "FERRITINE",
                    "30 à 40",
                    "35 ng/mL",
                    "Page 1 sur 1",
                ],
                {
                    "analyte": "FERRITINE",
                    "value_raw": "35",
                    "unit": "ng/mL",
                    "reference": "30 à 40",
                    "previous": None,
                },
            ),
            (
                "biochimie_range_35_72_no_previous",
                [
                    "LABORATOIRE CENTRAL",
                    "Date Demande: 01/03/2025 08:00:00",
                    "Paramétres",
                    "ALBUMINE",
                    "35 - 72",
                    "40 g/L",
                    "Page 1 sur 1",
                ],
                {
                    "analyte": "ALBUMINE",
                    "value_raw": "40",
                    "unit": "g/L",
                    "reference": "35 - 72",
                    "previous": None,
                },
            ),
            (
                "no_unit_with_previous",
                [
                    "LABORATOIRE CENTRAL",
                    "Date Demande: 02/03/2025 08:00:00",
                    "Paramétres",
                    "PARAMÈTRE SANS UNITÉ",
                    "18,00",
                    "002",
                    "Page 1 sur 1",
                ],
                {
                    "analyte": "PARAMÈTRE SANS UNITÉ",
                    "value_raw": "002",
                    "unit": "unknown",
                    "reference": None,
                    "previous": "18,00",
                },
            ),
            (
                "comparator_with_reference_unit_and_previous",
                [
                    "LABORATOIRE CENTRAL",
                    "Date Demande: 02/03/2025 08:00:00",
                    "Paramétres",
                    "CALCITONINE",
                    "5,30",
                    "< 11,80 pg/ml",
                    "3,00 pg/mL",
                    "Page 1 sur 1",
                ],
                {
                    "analyte": "CALCITONINE",
                    "value_raw": "3,00",
                    "unit": "pg/mL",
                    "reference": "< 11,80 pg/ml",
                    "previous": "5,30",
                    "reference_operator": "<",
                    "reference_unit": "pg/ml",
                    "reference_high": 11.8,
                },
            ),
            (
                "multi_interval_multi_unit_reference",
                [
                    "LABORATOIRE CENTRAL",
                    "Date Demande: 02/03/2025 08:00:00",
                    "Paramétres",
                    "PTH INTACT",
                    "(15,00 - 65,00) pg/ml(1.6-6.9 pmol/l)",
                    "7,00 pg/ml",
                    "Page 1 sur 1",
                ],
                {
                    "analyte": "PTH INTACT",
                    "value_raw": "7,00",
                    "unit": "pg/ml",
                    "reference": "(15,00 - 65,00) pg/ml(1.6-6.9 pmol/l)",
                    "previous": None,
                    "reference_operator": "multiple",
                    "reference_unit": None,
                },
            ),
        ]

        for _name, lines, expected in cases:
            with self.subTest(case=_name):
                results, _raw, _stats = extract_chu_lab_results(_make_chu_pages(lines))
                self.assertEqual(len(results), 1)
                row = results[0]
                self.assertEqual(row["analyte"], expected["analyte"])
                self.assertEqual(row["value_raw"], expected["value_raw"])
                self.assertEqual(row["unit"], expected["unit"])
                self.assertEqual((row["reference_range"] or {}).get("text"), expected["reference"])
                if "reference_operator" in expected:
                    self.assertEqual((row["reference_range"] or {}).get("operator"), expected["reference_operator"])
                if "reference_unit" in expected:
                    self.assertEqual((row["reference_range"] or {}).get("unit"), expected["reference_unit"])
                if "reference_high" in expected:
                    self.assertEqual((row["reference_range"] or {}).get("high"), expected["reference_high"])
                if expected.get("reference_operator") == "multiple":
                    ranges = (row["reference_range"] or {}).get("reference_ranges")
                    self.assertIsInstance(ranges, list)
                    self.assertEqual(len(ranges), 2)
                    self.assertEqual((ranges[0] or {}).get("unit"), "pg/ml")
                    self.assertEqual((ranges[0] or {}).get("low"), 15.0)
                    self.assertEqual((ranges[0] or {}).get("high"), 65.0)
                    self.assertEqual((ranges[1] or {}).get("unit"), "pmol/l")
                    self.assertEqual((ranges[1] or {}).get("low"), 1.6)
                    self.assertEqual((ranges[1] or {}).get("high"), 6.9)
                if expected["previous"] is None:
                    self.assertIsNone(row.get("previous_result"))
                else:
                    self.assertEqual((row["previous_result"] or {}).get("value_raw"), expected["previous"])


if __name__ == "__main__":
    unittest.main()
