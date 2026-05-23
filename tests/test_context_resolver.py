from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from context_resolver import resolve_context_for_turn, resolve_deictic_request
from query_understanding import parse_query_understanding


class TestContextResolver(unittest.TestCase):
    def test_followup_reuses_previous_doc_scope(self) -> None:
        qu = parse_query_understanding("et TSHus ?")
        state = {
            "last_data_context_type": "biological_numeric_results",
            "last_doc_scope": {"doc_ids": ["report_16"]},
            "last_transformable_evidence_pack": {"evidences": [{"analyte": "ACTH", "value_numeric": 23.0}]},
        }
        resolved = resolve_context_for_turn("et TSHus ?", qu, state)
        self.assertTrue(resolved.get("reuse_doc_scope"))
        self.assertEqual((resolved.get("effective_doc_scope") or {}).get("doc_ids"), ["report_16"])
        self.assertFalse(resolved.get("should_skip_retrieval"))

    def test_inventory_render_skips_retrieval(self) -> None:
        qu = parse_query_understanding("ok affiche ça dans une table filtrable par patient, date ou nom de fichier")
        state = {
            "last_data_context_type": "patient_inventory",
            "last_patient_inventory": [{"patient": "PAT_000001"}],
        }
        resolved = resolve_context_for_turn("ok affiche ça dans une table filtrable", qu, state)
        self.assertTrue(resolved.get("reuse_patient_inventory"))
        self.assertTrue(resolved.get("should_skip_retrieval"))

    def test_deictic_table_after_inventory_skips_retrieval(self) -> None:
        qu = parse_query_understanding("affiche ça en table")
        state = {
            "last_data_context_type": "patient_inventory",
            "last_patient_inventory": [{"patient": "PAT_000001"}],
        }
        resolved = resolve_context_for_turn("affiche ça en table", qu, state)
        self.assertTrue(resolved.get("deictic_table_request"))
        self.assertTrue(resolved.get("should_skip_retrieval"))
        self.assertEqual(resolved.get("reason"), "inventory_deictic_table_reuse")

    def test_qualitative_render_skips_retrieval(self) -> None:
        qu = parse_query_understanding("ok affiche ce commentaire dans un bloc commentaire sourcé")
        state = {
            "last_data_context_type": "medical_qualitative_comment",
            "last_qualitative_evidence_pack": {"comment_text": "Valeur seuil..."},
        }
        resolved = resolve_context_for_turn("ok affiche ce commentaire dans un bloc commentaire sourcé", qu, state)
        self.assertTrue(resolved.get("reuse_qualitative_pack"))
        self.assertTrue(resolved.get("should_skip_retrieval"))

    def test_correction_table_followup_reuses_qualitative_context(self) -> None:
        qu = parse_query_understanding("non, dans une table")
        state = {
            "last_data_context_type": "medical_qualitative_comment",
            "last_qualitative_evidence_pack": {"evidences": [{"subject": "Commentaire médical", "display_comment_text": "<4,11 IU/ml"}]},
            "last_displayed_context": {"context_type": "medical_qualitative_comment", "subject": "IMMUNOANALYSE"},
        }
        resolved = resolve_deictic_request("non, dans une table", qu, state)
        self.assertTrue(resolved.get("resolved"))
        self.assertEqual(resolved.get("intent"), "qualitative_comment_render")
        self.assertEqual(resolved.get("render_type"), "text_table")
        self.assertTrue(resolved.get("skip_retrieval"))

    def test_typo_tabl_followup_reuses_qualitative_text_table(self) -> None:
        qu = parse_query_understanding("affiche ca dans une tabl")
        state = {
            "last_data_context_type": "medical_qualitative_comment",
            "last_qualitative_evidence_pack": {"evidences": [{"subject": "Commentaire médical", "display_comment_text": "<4,11 IU/ml"}]},
            "last_displayed_context": {"context_type": "medical_qualitative_comment", "subject": "IMMUNOANALYSE"},
        }
        resolved = resolve_deictic_request("affiche ca dans une tabl", qu, state)
        self.assertTrue(resolved.get("resolved"))
        self.assertEqual(resolved.get("intent"), "qualitative_comment_render")
        self.assertEqual(resolved.get("render_type"), "text_table")

    def test_same_action_followup_keeps_reference_range_lookup_intent(self) -> None:
        qu = parse_query_understanding("et pour TSHus")
        state = {
            "last_intent": "reference_range_lookup",
            "last_data_context_type": "biological_numeric_results",
            "last_displayed_context": {"context_type": "biological_numeric_results", "subject": "ACTH"},
            "last_doc_scope": {"doc_ids": ["report_1"]},
        }
        resolved = resolve_deictic_request("et pour TSHus", qu, state)
        self.assertTrue(resolved.get("resolved"))
        self.assertEqual(resolved.get("intent"), "reference_range_lookup")


if __name__ == "__main__":
    unittest.main()
