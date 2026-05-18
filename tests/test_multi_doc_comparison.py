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


if __name__ == "__main__":
    unittest.main()
