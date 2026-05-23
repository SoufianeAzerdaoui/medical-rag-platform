from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from reference_range_selector import select_reference_range


class TestReferenceRangeSelector(unittest.TestCase):
    def _ranges(self) -> list[dict]:
        return [
            {"label": "2 à 12 ans", "population": "child", "sex": None, "age_min": 2.0, "age_max": 12.0, "age_unit": "years", "operator": "range", "low": 88.0, "high": 108.0, "unit": "mg/l"},
            {"label": "Adulte", "population": "adult", "sex": None, "operator": "range", "low": 84.0, "high": 102.0, "unit": "mg/l"},
            {"label": "Homme>60 ans", "population": None, "sex": "male", "age_operator": ">", "age_value": 60.0, "age_unit": "years", "operator": "range", "low": 88.0, "high": 100.0, "unit": "mg/l"},
        ]

    def test_male_over_60(self) -> None:
        out = select_reference_range(self._ranges(), requested_profile={"sex": "male", "age": 65, "age_unit": "years"})
        self.assertEqual(out.get("status"), "selected")
        self.assertEqual((out.get("selected") or {}).get("low"), 88.0)
        self.assertEqual((out.get("selected") or {}).get("high"), 100.0)

    def test_adult_population(self) -> None:
        out = select_reference_range(self._ranges(), requested_profile={"population": "adult"})
        self.assertEqual(out.get("status"), "selected")
        self.assertEqual((out.get("selected") or {}).get("low"), 84.0)
        self.assertEqual((out.get("selected") or {}).get("high"), 102.0)

    def test_without_profile_ambiguous(self) -> None:
        out = select_reference_range(self._ranges(), requested_profile=None)
        self.assertEqual(out.get("status"), "ambiguous")
        self.assertIsNone(out.get("selected"))

    def test_patient_profile_child(self) -> None:
        out = select_reference_range(
            self._ranges(),
            requested_profile=None,
            patient_profile={"sex": "female", "age": 2, "age_unit": "years"},
            use_patient_profile=True,
        )
        self.assertEqual(out.get("status"), "selected")
        self.assertEqual((out.get("selected") or {}).get("label"), "2 à 12 ans")

    def test_fallback_explicit(self) -> None:
        ranges = [{"label": "Adulte", "population": "adult", "operator": "range", "low": 84.0, "high": 102.0, "unit": "mg/l"}]
        out = select_reference_range(ranges, requested_profile={"sex": "male", "age": 80, "age_unit": "years"})
        self.assertIn(out.get("status"), {"selected", "fallback"})

    def test_no_hardcode(self) -> None:
        ranges = [
            {"label": "Alpha", "population": None, "sex": None, "operator": "range", "low": 1.0, "high": 2.0, "unit": "u"},
            {"label": "Beta>50 ans", "population": None, "sex": None, "age_operator": ">", "age_value": 50.0, "age_unit": "years", "operator": "range", "low": 3.0, "high": 4.0, "unit": "u"},
        ]
        out = select_reference_range(ranges, requested_profile={"age": 55, "age_unit": "years"})
        self.assertEqual(out.get("status"), "selected")
        self.assertEqual((out.get("selected") or {}).get("low"), 3.0)
        self.assertEqual((out.get("selected") or {}).get("high"), 4.0)

    def test_amh_female_25_29_selected_not_ambiguous(self) -> None:
        ranges = [
            {"label": "Homme", "sex": "male", "operator": "range", "low": 4.35, "high": 5.35, "unit": "ng/ml"},
            {"label": "Femme cyclée J2-J4 — 20-24 ans", "sex": "female", "population": "cycled_female_j2_j4", "age_min": 20.0, "age_max": 24.0, "age_unit": "years", "operator": "range", "low": 3.55, "high": 4.33, "unit": "ng/ml"},
            {"label": "Femme cyclée J2-J4 — 25-29 ans", "sex": "female", "population": "cycled_female_j2_j4", "age_min": 25.0, "age_max": 29.0, "age_unit": "years", "operator": "range", "low": 3.03, "high": 3.87, "unit": "ng/ml"},
        ]
        out = select_reference_range(
            ranges,
            requested_profile={"sex": "female", "age_min": 25, "age_max": 29, "age_unit": "years"},
        )
        self.assertEqual(out.get("status"), "selected")
        self.assertEqual((out.get("selected") or {}).get("low"), 3.03)
        self.assertEqual((out.get("selected") or {}).get("high"), 3.87)

    def test_amh_population_without_age_returns_grouped_options(self) -> None:
        ranges = [
            {"label": "Homme", "sex": "male", "operator": "range", "low": 4.35, "high": 5.35, "unit": "ng/ml"},
            {"label": "Femme cyclée J2-J4 — 20-24 ans", "sex": "female", "population": "cycled_female_j2_j4", "age_min": 20.0, "age_max": 24.0, "age_unit": "years", "operator": "range", "low": 3.55, "high": 4.33, "unit": "ng/ml"},
            {"label": "Femme cyclée J2-J4 — 25-29 ans", "sex": "female", "population": "cycled_female_j2_j4", "age_min": 25.0, "age_max": 29.0, "age_unit": "years", "operator": "range", "low": 3.03, "high": 3.87, "unit": "ng/ml"},
            {"label": "Femme cyclée J2-J4 — 30-34 ans", "sex": "female", "population": "cycled_female_j2_j4", "age_min": 30.0, "age_max": 34.0, "age_unit": "years", "operator": "range", "low": 2.34, "high": 3.55, "unit": "ng/ml"},
        ]
        out = select_reference_range(
            ranges,
            requested_profile={"sex": "female", "population": "cycled_female_j2_j4"},
        )
        self.assertEqual(out.get("status"), "grouped_options")
        cands = list(out.get("candidates") or [])
        self.assertTrue(cands)
        self.assertTrue(all(str(c.get("population") or "") == "cycled_female_j2_j4" for c in cands))
        self.assertTrue(all(str(c.get("sex") or "") == "female" for c in cands))

    def test_haptoglobine_female_no_age_grouped_options(self) -> None:
        ranges = [
            {"sex": "female", "age_min": 0.0, "age_max": 1.0, "age_unit": "years", "operator": "range", "low": 0.0, "high": 2.35, "unit": "g/l"},
            {"sex": "female", "age_min": 1.0, "age_max": 12.0, "age_unit": "years", "operator": "range", "low": 0.11, "high": 2.20, "unit": "g/l"},
            {"sex": "female", "age_min": 12.0, "age_max": 60.0, "age_unit": "years", "operator": "range", "low": 0.35, "high": 2.50, "unit": "g/l"},
            {"sex": "female", "age_operator": ">", "age_value": 60.0, "age_unit": "years", "operator": "range", "low": 0.63, "high": 2.73, "unit": "g/l"},
            {"sex": "male", "age_operator": ">", "age_value": 60.0, "age_unit": "years", "operator": "range", "low": 0.4, "high": 2.68, "unit": "g/l"},
        ]
        out = select_reference_range(ranges, requested_profile={"sex": "female"})
        self.assertEqual(out.get("status"), "grouped_options")
        self.assertTrue(all(str(c.get("sex") or "") == "female" for c in (out.get("candidates") or [])))

    def test_haptoglobine_female_over_60_selected(self) -> None:
        ranges = [
            {"sex": "female", "age_operator": ">", "age_value": 60.0, "age_unit": "years", "operator": "range", "low": 0.63, "high": 2.73, "unit": "g/l"},
            {"sex": "male", "age_operator": ">", "age_value": 60.0, "age_unit": "years", "operator": "range", "low": 0.4, "high": 2.68, "unit": "g/l"},
        ]
        out = select_reference_range(ranges, requested_profile={"sex": "female", "age_operator": ">", "age": 60, "age_unit": "years"})
        self.assertEqual(out.get("status"), "selected")
        self.assertEqual((out.get("selected") or {}).get("low"), 0.63)
        self.assertEqual((out.get("selected") or {}).get("high"), 2.73)

    def test_pal_male_no_age_grouped_options(self) -> None:
        ranges = [
            {"sex": "male", "age_min": 1.0, "age_max": 12.0, "age_unit": "years", "operator": "<", "threshold": 500.0, "unit": "UI/L"},
            {"sex": "male", "age_min": 12.0, "age_max": 15.0, "age_unit": "years", "operator": "<", "threshold": 750.0, "unit": "UI/L"},
            {"sex": "male", "age_operator": ">", "age_value": 20.0, "age_unit": "years", "operator": "range", "low": 40.0, "high": 150.0, "unit": "UI/L"},
            {"sex": "female", "age_operator": ">", "age_value": 15.0, "age_unit": "years", "operator": "range", "low": 40.0, "high": 150.0, "unit": "UI/L"},
        ]
        out = select_reference_range(ranges, requested_profile={"sex": "male"})
        self.assertEqual(out.get("status"), "grouped_options")
        self.assertTrue(all(str(c.get("sex") or "") == "male" for c in (out.get("candidates") or [])))

    def test_pal_male_12_15_selected(self) -> None:
        ranges = [
            {"sex": "male", "age_min": 12.0, "age_max": 15.0, "age_unit": "years", "operator": "<", "threshold": 750.0, "unit": "UI/L"},
            {"sex": "male", "age_operator": ">", "age_value": 20.0, "age_unit": "years", "operator": "range", "low": 40.0, "high": 150.0, "unit": "UI/L"},
        ]
        out = select_reference_range(ranges, requested_profile={"sex": "male", "age_min": 12, "age_max": 15, "age_unit": "years"})
        self.assertEqual(out.get("status"), "selected")
        self.assertEqual((out.get("selected") or {}).get("operator"), "<")
        self.assertEqual((out.get("selected") or {}).get("threshold"), 750.0)


if __name__ == "__main__":
    unittest.main()
