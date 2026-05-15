from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from context_resolver import resolve_context_for_turn
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


if __name__ == "__main__":
    unittest.main()
