from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from generate_answer import run_generation


class TestMultiDocComparison(unittest.TestCase):
    def test_glycemie_glucose_between_reports(self) -> None:
        result = run_generation(
            query="Compare les résultats de la Glycémie (Glucose) entre le report 10 et le report 12.",
            index_dir="data/indexes",
        )
        qu = dict(result.get("query_understanding") or {})
        self.assertEqual(str(qu.get("intent") or ""), "multi_doc_comparison")
        self.assertIn("glucose", list(qu.get("requested_analytes") or []))
        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("aucun résultat exploitable", answer)
        self.assertIn("glucose", answer)
        self.assertNotIn("présents dans un rapport et absents", answer)
        self.assertNotIn("différence technique", answer)
        self.assertTrue(("aucun écart numérique" in answer) or ("valeurs identiques" in answer))

    def test_compare_out_of_reference_between_two_reports_without_explicit_analyte(self) -> None:
        result = run_generation(
            query="Compare les anomalies biologiques entre le report 10 et le report 12. Quels résultats sont hors référence dans chaque rapport ?",
            index_dir="data/indexes",
        )
        debug = dict(result.get("debug") or {})
        qu = dict(result.get("query_understanding") or {})
        self.assertEqual(str(debug.get("selected_route") or ""), "doc_pair_comparison")
        self.assertEqual(list(qu.get("requested_doc_ids") or []), ["report_10", "report_12"])
        self.assertEqual(str(qu.get("technical_condition") or ""), "out_of_reference")
        self.assertGreater(len(list(result.get("displayed_evidences") or [])), 0)
        self.assertGreater(len(list(result.get("sources") or [])), 0)
        self.assertNotEqual(str(result.get("generation_mode") or ""), "deterministic_safe_error_response")
        self.assertNotEqual(str((result.get("validation") or {}).get("validation_status") or "").lower(), "fail")
        answer = str(result.get("answer") or "").lower()
        self.assertIn("report_10", answer)
        self.assertIn("report_12", answer)
        self.assertTrue(any(k in answer for k in ["albumine", "triglycerides", "triglycérides"]))
        self.assertTrue(any(k in answer for k in ["bilirubine directe", "ldh", "ckmb"]))


if __name__ == "__main__":
    unittest.main()
