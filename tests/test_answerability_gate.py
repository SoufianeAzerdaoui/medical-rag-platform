from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_ROOT))

from answerability_gate import evaluate_answerability


class TestAnswerabilityGate(unittest.TestCase):
    def test_answerable_exact(self) -> None:
        out = evaluate_answerability(
            requested_analytes=["acide urique"],
            evidence_rows=[{"analyte": "Acide urique", "analyte_norm": "acide_urique", "doc_id": "report_24"}],
            requested_doc_ids=["report_24"],
        )
        self.assertEqual(out.get("status"), "answerable_exact")

    def test_answerable_alias_tsh_tshus(self) -> None:
        out = evaluate_answerability(
            requested_analytes=["TSH"],
            evidence_rows=[{"analyte": "TSHus", "analyte_norm": "tshus", "doc_id": "report_16"}],
            requested_doc_ids=["report_16"],
        )
        self.assertIn(out.get("status"), {"answerable_alias", "answerable_exact"})

    def test_answerable_topic(self) -> None:
        out = evaluate_answerability(
            requested_analytes=["fonction rénale"],
            evidence_rows=[{"analyte": "Créatinine", "analyte_norm": "creatinine", "doc_id": "report_29"}],
            requested_doc_ids=["report_29"],
        )
        self.assertEqual(out.get("status"), "answerable_topic")

    def test_partially_answerable_multi_doc(self) -> None:
        out = evaluate_answerability(
            requested_analytes=["TSH"],
            evidence_rows=[{"analyte": "TSHus", "analyte_norm": "tshus", "doc_id": "report_16"}],
            requested_doc_ids=["report_10", "report_16", "report_24"],
        )
        self.assertEqual(out.get("status"), "partially_answerable")
        self.assertIn("report_10", list(out.get("missing_doc_ids") or []))

    def test_not_found(self) -> None:
        out = evaluate_answerability(
            requested_analytes=["cortisol"],
            evidence_rows=[{"analyte": "Créatinine", "analyte_norm": "creatinine", "doc_id": "report_12"}],
            requested_doc_ids=["report_12"],
        )
        self.assertEqual(out.get("status"), "not_found")

    def test_not_found_has_priority_over_ambiguity_for_explicit_scope(self) -> None:
        out = evaluate_answerability(
            requested_analytes=["cortisol"],
            evidence_rows=[{"analyte": "Créatinine", "analyte_norm": "creatinine", "doc_id": "report_12"}],
            requested_doc_ids=["report_12"],
            ambiguity_flags=["confidence_below_threshold", "multiple_candidates_clustered"],
        )
        self.assertEqual(out.get("status"), "not_found")

    def test_ambiguous(self) -> None:
        out = evaluate_answerability(
            requested_analytes=[],
            evidence_rows=[],
            requested_doc_ids=[],
            ambiguity_flags=["missing_doc_scope", "confidence_below_threshold"],
        )
        self.assertEqual(out.get("status"), "ambiguous")

    def test_unsafe(self) -> None:
        out = evaluate_answerability(
            requested_analytes=[],
            evidence_rows=[],
            requested_doc_ids=[],
            safety_intent="diagnostic_safety_question",
        )
        self.assertEqual(out.get("status"), "unsafe")

    def test_unsafe_treatment_refusal(self) -> None:
        out = evaluate_answerability(
            requested_analytes=[],
            evidence_rows=[],
            requested_doc_ids=[],
            safety_intent="treatment_refusal",
        )
        self.assertEqual(out.get("status"), "unsafe")
        self.assertEqual(out.get("reason"), "treatment_safety_intent")


if __name__ == "__main__":
    unittest.main()
