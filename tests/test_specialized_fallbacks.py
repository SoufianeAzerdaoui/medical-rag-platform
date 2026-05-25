from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_ROOT))

from specialized_fallbacks import build_specialized_fallback, infer_specialized_fallback_kind


class TestSpecializedFallbacks(unittest.TestCase):
    def test_single_analyte_not_found_template(self) -> None:
        out = build_specialized_fallback(
            kind="single_analyte_not_found",
            requested_analytes=["cortisol"],
            requested_doc_ids=["report_12"],
        )
        self.assertEqual(out.kind, "single_analyte_not_found")
        self.assertIn("Aucun résultat correspondant", out.answer)
        self.assertIn("report 12", out.answer.lower())

    def test_ambiguous_document_scope_template(self) -> None:
        out = build_specialized_fallback(kind="ambiguous_document_scope")
        self.assertEqual(out.kind, "ambiguous_document_scope")
        self.assertIn("Précisez un rapport", out.answer)

    def test_insufficient_evidence_with_numeric_criterion(self) -> None:
        out = build_specialized_fallback(
            kind="insufficient_evidence",
            requested_analytes=["creatinine"],
            requested_doc_ids=["report_10", "report_12"],
            requested_value="2",
            comparison_operator=">",
        )
        self.assertIn("strictement supérieur à 2", out.answer)

    def test_infer_kind_unsafe(self) -> None:
        kind = infer_specialized_fallback_kind(
            answerability_status="unsafe",
            answerability_reason="diagnostic_safety_intent",
            safety_intent="diagnostic_safety_question",
            requested_analytes=[],
            requested_doc_ids=[],
            ambiguity_flags=[],
        )
        self.assertEqual(kind, "diagnosis_refusal")

    def test_infer_kind_partial(self) -> None:
        kind = infer_specialized_fallback_kind(
            answerability_status="partially_answerable",
            answerability_reason="partial_match",
            safety_intent="",
            requested_analytes=["tsh"],
            requested_doc_ids=["report_10", "report_16", "report_24"],
            ambiguity_flags=[],
        )
        self.assertEqual(kind, "partial_answer")

    def test_infer_kind_not_found_single_doc_analyte(self) -> None:
        kind = infer_specialized_fallback_kind(
            answerability_status="not_found",
            answerability_reason="no_compatible_evidence",
            safety_intent="",
            requested_analytes=["cortisol"],
            requested_doc_ids=["report_12"],
            ambiguity_flags=[],
        )
        self.assertEqual(kind, "single_analyte_not_found")


if __name__ == "__main__":
    unittest.main()
