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

from medical_query_planner import understand_medical_query


class TestMedicalQueryPlanner(unittest.TestCase):
    def test_doc_scoped_abnormal_results(self) -> None:
        plan = understand_medical_query("Dans report (19), quels résultats sont hors référence ?")
        self.assertEqual(plan.intent, "doc_scoped_abnormal_results")
        self.assertEqual(plan.scope, "single_document")
        self.assertIn("report_19", plan.requested_doc_ids)
        self.assertIn(plan.technical_condition, {"out_of_reference", "any_result"})

    def test_doc_pair_comparison(self) -> None:
        plan = understand_medical_query("Compare report 10 et report 12 pour le glucose.")
        self.assertEqual(plan.intent, "doc_pair_comparison")
        self.assertEqual(plan.scope, "multi_document")
        self.assertEqual(plan.comparison_targets, ["report_10", "report_12"])

    def test_global_abnormal_search(self) -> None:
        plan = understand_medical_query(
            "Dans tous les rapports disponibles, quels documents contiennent une insuline hors référence ?"
        )
        self.assertEqual(plan.intent, "global_abnormal_search")
        self.assertEqual(plan.scope, "all_documents")
        self.assertIn("insuline", [a.lower() for a in plan.requested_analytes])

    def test_guarded_medical_interpretation(self) -> None:
        plan = understand_medical_query("Est-ce que le report (16) permet de conclure à une hyperthyroïdie ?")
        self.assertEqual(plan.intent, "guarded_medical_interpretation")
        self.assertEqual(plan.safety_mode, "grounded_no_diagnosis_no_treatment")

    def test_reference_range_lookup(self) -> None:
        plan = understand_medical_query("Quelle est la plage normale du calcium pour homme > 60 ans ?")
        self.assertEqual(plan.intent, "reference_range_lookup")
        self.assertFalse(plan.requires_llm_writer)

    def test_open_grounded_medical_question(self) -> None:
        plan = understand_medical_query("Qu'est-ce qui peut expliquer une TSH élevée avec T4 libre élevée ?")
        self.assertIn(plan.intent, {"guarded_medical_interpretation", "open_grounded_medical_question"})
        self.assertIn(plan.scope, {"retrieval_required", "single_document", "all_documents"})

    def test_document_detection_variants(self) -> None:
        plan = understand_medical_query("Dans le rapport 16, donne la valeur de ACTH.")
        self.assertIn("report_16", plan.requested_doc_ids)
        self.assertEqual(plan.intent, "single_analyte_lookup")


if __name__ == "__main__":
    unittest.main()

