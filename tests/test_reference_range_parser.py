from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from reference_range_parser import parse_reference_ranges


class TestReferenceRangeParser(unittest.TestCase):
    def test_calcium_complex(self) -> None:
        raw = (
            "Cordon: 82 - 112 mg/l Nourrisson: 62 - 110 mg/l 0 à 10 jours: 76 - 104 mg/l "
            "10 j à 24 mois: 90 - 110 mg/l 2 à 12 ans: 88 - 108 mg/l Adulte: 84 - 102 mg/l "
            "Homme>60 ans: 88 - 100 mg/l"
        )
        ranges = parse_reference_ranges(raw)
        self.assertGreaterEqual(len(ranges), 7)
        male = next((r for r in ranges if str(r.get("sex")) == "male" and str(r.get("age_operator")) == ">"), None)
        self.assertIsNotNone(male)
        self.assertEqual(male.get("age_value"), 60.0)
        self.assertEqual(male.get("low"), 88.0)
        self.assertEqual(male.get("high"), 100.0)
        self.assertEqual(str(male.get("unit")).lower(), "mg/l")

    def test_creatinine_sex(self) -> None:
        raw = "Enfant: 4 - 9 mg/l Homme : 7,2 - 12,5 mg/l Femme : 5,7 - 11,1 mg/l"
        ranges = parse_reference_ranges(raw)
        male = next(r for r in ranges if r.get("sex") == "male")
        female = next(r for r in ranges if r.get("sex") == "female")
        child = next(r for r in ranges if r.get("population") == "child")
        self.assertEqual((male.get("low"), male.get("high")), (7.2, 12.5))
        self.assertEqual((female.get("low"), female.get("high")), (5.7, 11.1))
        self.assertEqual((child.get("low"), child.get("high")), (4.0, 9.0))

    def test_threshold_simple(self) -> None:
        ranges = parse_reference_ranges("<4,11 IU/ml")
        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0].get("operator"), "<")
        self.assertEqual(ranges[0].get("threshold"), 4.11)
        self.assertEqual(str(ranges[0].get("unit")).lower(), "iu/ml")

    def test_pth_dual_unit_ranges(self) -> None:
        ranges = parse_reference_ranges("(15,00 - 65,00) pg/ml(1.6-6.9 pmol/l)")
        self.assertGreaterEqual(len(ranges), 2)
        self.assertTrue(any(r.get("low") == 15.0 and r.get("high") == 65.0 and str(r.get("unit")).lower() == "pg/ml" for r in ranges))
        self.assertTrue(any(r.get("low") == 1.6 and r.get("high") == 6.9 and str(r.get("unit")).lower() == "pmol/l" for r in ranges))

    def test_mixed_age(self) -> None:
        ranges = parse_reference_ranges("10 j à 24 mois: 90 - 110 mg/l")
        self.assertEqual(len(ranges), 1)
        r = ranges[0]
        self.assertEqual(r.get("age_min"), 10.0)
        self.assertEqual(r.get("age_max"), 24.0)
        self.assertEqual(r.get("age_unit"), "months")

    def test_amh_female_age_subranges(self) -> None:
        raw = (
            "Homme: 4.35-5.35 ng/ml "
            "Femme cyclée J2-J4: -age(20-24 ans) : 3.55-4.33 ng/ml "
            "-age(25-29 ans) : 3.03-3.87 ng/ml -age(30-34 ans) : 2.34-3.55 ng/ml"
        )
        ranges = parse_reference_ranges(raw)
        target = next(
            (
                r
                for r in ranges
                if r.get("sex") == "female"
                and r.get("age_min") == 25.0
                and r.get("age_max") == 29.0
            ),
            None,
        )
        self.assertIsNotNone(target)
        self.assertEqual(target.get("low"), 3.03)
        self.assertEqual(target.get("high"), 3.87)
        self.assertEqual(str(target.get("unit")).lower(), "ng/ml")
        self.assertEqual(target.get("population"), "cycled_female_j2_j4")
        self.assertFalse(any(r.get("label") == "Femme cyclée J2-J4" and r.get("low") == 20.0 and r.get("high") == 24.0 for r in ranges))

    def test_haptoglobine_parent_context(self) -> None:
        raw = (
            "Femme 0 - 1 an: 0 - 2,35 g/l 1 - 12 ans: 0,11 - 2,20 g/l 12 - 60 ans: 0,35 - 2,50 g/l > 60 ans: 0,63 - 2,73 g/l "
            "Homme 0 - 1 an: 0 - 3 g/l 1 - 12 ans: 0,03 - 2,70 g/l 12 - 60 ans: 0,14 - 2,58 g/l > 60 ans: 0,40 - 2,68 g/l"
        )
        ranges = parse_reference_ranges(raw)
        female = [r for r in ranges if r.get("sex") == "female"]
        male = [r for r in ranges if r.get("sex") == "male"]
        self.assertEqual(len(female), 4)
        self.assertEqual(len(male), 4)
        f60 = next(r for r in female if str(r.get("age_operator") or "") == ">")
        m60 = next(r for r in male if str(r.get("age_operator") or "") == ">")
        self.assertEqual((f60.get("low"), f60.get("high")), (0.63, 2.73))
        self.assertEqual((m60.get("low"), m60.get("high")), (0.4, 2.68))

    def test_phosphatase_parent_context(self) -> None:
        raw = "Femme : 1 à 12 ans: < 500 UI/L > 15 ans: 40 - 150 UI/L Homme : 1 à 12 ans: < 500 UI/L 12 à 15 ans: < 750 UI/L >20 ans: 40 - 150 UI/L"
        ranges = parse_reference_ranges(raw)
        male = [r for r in ranges if r.get("sex") == "male"]
        female = [r for r in ranges if r.get("sex") == "female"]
        m12_15 = next((r for r in male if r.get("age_min") == 12.0 and r.get("age_max") == 15.0), None)
        mgt20 = next((r for r in male if str(r.get("age_operator") or "") == ">" and r.get("age_value") == 20.0), None)
        fgt15 = next((r for r in female if str(r.get("age_operator") or "") == ">" and r.get("age_value") == 15.0), None)
        self.assertIsNotNone(m12_15)
        self.assertEqual(m12_15.get("threshold"), 750.0)
        self.assertIsNotNone(mgt20)
        self.assertEqual((mgt20.get("low"), mgt20.get("high")), (40.0, 150.0))
        self.assertIsNotNone(fgt15)
        self.assertEqual((fgt15.get("low"), fgt15.get("high")), (40.0, 150.0))


if __name__ == "__main__":
    unittest.main()
