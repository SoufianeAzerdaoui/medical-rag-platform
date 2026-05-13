from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from backend_api import _CONVERSATION_STATE, _evidence_pack_is_transformable, _get_transformable_context, _update_conversation_state
    _BACKEND_IMPORT_OK = True
except Exception:
    _BACKEND_IMPORT_OK = False
    from scripts.generation.conversation_state_utils import (
        evidence_pack_is_transformable as _evidence_pack_is_transformable,
        get_transformable_context as _get_transformable_context,
        update_conversation_state as _update_state_pure,
    )
    _CONVERSATION_STATE = {}

    def _update_conversation_state(chat_id: str, state: dict, generation: dict, user_message: str) -> None:
        _update_state_pure(
            state_store=_CONVERSATION_STATE,
            chat_id=chat_id,
            state=state,
            generation=generation,
            user_message=user_message,
        )


def _old_medical_pack() -> dict:
    return {
        "evidences": [
            {"analyte": "ACE", "current_value": "22", "value_numeric": 22.0, "doc_id": "report_31", "source": "report (31).pdf"},
            {"analyte": "PSA TOTALE", "current_value": "33", "value_numeric": 33.0, "doc_id": "report_31", "source": "report (31).pdf"},
            {"analyte": "CA 15-3", "current_value": "44", "value_numeric": 44.0, "doc_id": "report_31", "source": "report (31).pdf"},
        ]
    }


class TestConversationStateManagement(unittest.TestCase):
    def tearDown(self) -> None:
        for k in list(_CONVERSATION_STATE.keys()):
            if k.startswith("ut-state-"):
                _CONVERSATION_STATE.pop(k, None)

    def test_a_inventory_invalidates_transformable_pack(self) -> None:
        chat_id = "ut-state-a"
        old_pack = _old_medical_pack()
        _CONVERSATION_STATE[chat_id] = {
            "last_evidence_pack": old_pack,
            "last_transformable_evidence_pack": old_pack,
        }
        generation = {
            "query_understanding": {"intent": "patient_inventory"},
            "answer": "inventory",
            "patients": [{"patient": "PAT_000001", "reports": [{"doc_id": "report", "source_url": "/api/documents/report/pdf"}]}],
            "sources": [],
            "structured_evidence_pack": None,
        }
        _update_conversation_state(chat_id, _CONVERSATION_STATE[chat_id], generation, "list patients")
        st = _CONVERSATION_STATE[chat_id]
        self.assertEqual(st.get("last_intent"), "patient_inventory")
        self.assertTrue(st.get("last_patient_inventory"))
        self.assertIsNone(st.get("last_transformable_evidence_pack"))
        self.assertEqual(st.get("last_evidence_pack"), old_pack)

    def test_b_inventory_count_invalidates_transformable_pack(self) -> None:
        chat_id = "ut-state-b"
        old_pack = _old_medical_pack()
        _CONVERSATION_STATE[chat_id] = {"last_transformable_evidence_pack": old_pack}
        generation = {
            "query_understanding": {"intent": "patient_inventory_count"},
            "answer": "count",
            "structured_evidence_pack": None,
        }
        _update_conversation_state(chat_id, _CONVERSATION_STATE[chat_id], generation, "count")
        self.assertIsNone(_CONVERSATION_STATE[chat_id].get("last_transformable_evidence_pack"))

    def test_c_no_fallback_to_last_evidence_when_transformable_none(self) -> None:
        state = {
            "last_evidence_pack": _old_medical_pack(),
            "last_transformable_evidence_pack": None,
        }
        self.assertIsNone(_get_transformable_context(state))

    def test_d_valid_medical_pack_stays_transformable(self) -> None:
        chat_id = "ut-state-d"
        new_pack = {
            "evidences": [
                {
                    "analyte": "ACTH",
                    "current_value": "23,00 pg/ml",
                    "value_numeric": 23.0,
                    "unit": "pg/ml",
                    "doc_id": "report_16",
                    "source": "report (16).pdf — page 1, ligne 1",
                }
            ]
        }
        generation = {
            "query_understanding": {"intent": "doc_scoped_results"},
            "answer": "ok",
            "structured_evidence_pack": new_pack,
            "sources": [{"doc_id": "report_16", "url": "/api/documents/report_16/pdf?page=1"}],
        }
        _update_conversation_state(chat_id, {}, generation, "acth")
        st = _CONVERSATION_STATE[chat_id]
        self.assertEqual(st.get("last_evidence_pack"), new_pack)
        self.assertEqual(st.get("last_transformable_evidence_pack"), new_pack)

    def test_e_non_transformable_pack_detection(self) -> None:
        self.assertFalse(_evidence_pack_is_transformable(None))
        self.assertFalse(_evidence_pack_is_transformable({}))
        self.assertFalse(_evidence_pack_is_transformable({"evidences": []}))
        self.assertFalse(_evidence_pack_is_transformable({"evidences": [{"patient": "PAT_000001"}]}))
        self.assertTrue(_evidence_pack_is_transformable(_old_medical_pack()))

    def test_followup_doc_scope_is_preserved_in_state(self) -> None:
        chat_id = "ut-state-doc-scope"
        first_generation = {
            "query_understanding": {"intent": "doc_scoped_results", "requested_doc_ids": ["report_31"]},
            "answer": "ACTH...",
            "structured_evidence_pack": {
                "requested_doc_ids": ["report_31"],
                "evidences": [{"analyte": "ACTH", "value_numeric": 23.0, "doc_id": "report_31", "source": "report (31).pdf"}],
            },
            "sources": [{"doc_id": "report_31", "url": "/api/documents/report_31/pdf"}],
        }
        _update_conversation_state(chat_id, {}, first_generation, "montre-moi ACTH du dernier rapport")
        st = _CONVERSATION_STATE[chat_id]
        self.assertEqual(st.get("last_doc_scope"), ["report_31"])

        second_generation = {
            "generation_mode": "deterministic_response_transform",
            "query_understanding": {"intent": "response_transform"},
            "answer": "Je n’ai pas de résultats biologiques numériques récents...",
            "structured_evidence_pack": {},
        }
        _update_conversation_state(chat_id, st, second_generation, "affiche ça en radar")
        st2 = _CONVERSATION_STATE[chat_id]
        self.assertEqual(st2.get("last_doc_scope"), ["report_31"])

    def test_qualitative_context_is_recorded_as_non_transformable(self) -> None:
        chat_id = "ut-state-qualitative"
        generation = {
            "query_understanding": {
                "intent": "unstructured",
                "requested_context_type": "medical_qualitative_comment",
            },
            "answer": "Commentaire troponine...",
            "structured_evidence_pack": None,
        }
        _update_conversation_state(chat_id, {}, generation, "montre le commentaire sur la troponine")
        st = _CONVERSATION_STATE[chat_id]
        self.assertEqual(st.get("last_data_context_type"), "medical_qualitative_comment")
        self.assertIsNone(st.get("last_transformable_evidence_pack"))

    def test_response_transform_no_context_message_when_last_intent_inventory(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="Donne-moi les mêmes résultats sous forme radar chart.",
            index_dir="data/indexes",
            previous_structured_evidence_pack=None,
            previous_context_intent="patient_inventory",
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("inventaire de patients", answer)
        self.assertNotIn("ace", answer)
        self.assertNotIn("psa", answer)
        self.assertNotIn("ca 15-3", answer)
        self.assertNotIn("commentaire", answer)
        self.assertNotIn("report_20", answer)
        self.assertNotIn("valeur seuil", answer)
        self.assertIsNone(result.get("visualization"))
        self.assertIsNone(result.get("chart_data"))

    def test_qualitative_comment_pack_not_transformable(self) -> None:
        pack = {
            "results": [
                {
                    "analyte": "Commentaire",
                    "value": "Valeur seuil : 20 mUI/l ...",
                    "unit": "qualitative",
                    "reference": "Qualitatif",
                    "result_kind": "qualitative",
                    "source_id": "report_20",
                }
            ]
        }
        self.assertFalse(_evidence_pack_is_transformable(pack))

    def test_data_context_preserved_after_response_transform_no_context(self) -> None:
        chat_id = "ut-state-preserve-context"
        state = {
            "last_intent": "patient_inventory",
            "last_patient_inventory": [{"patient": "PAT_000001"}],
            "last_data_context_intent": "patient_inventory",
            "last_data_context_type": "patient_inventory",
            "last_transformable_evidence_pack": None,
        }
        _CONVERSATION_STATE[chat_id] = state
        generation = {
            "generation_mode": "deterministic_response_transform",
            "query_understanding": {"intent": "response_transform"},
            "answer": "Je n’ai pas de résultats biologiques numériques récents...",
            "structured_evidence_pack": {},
        }
        _update_conversation_state(chat_id, state, generation, "affiche ça en radar")
        updated = _CONVERSATION_STATE[chat_id]
        self.assertEqual(updated.get("last_intent"), "response_transform_no_context")
        self.assertEqual(updated.get("last_data_context_intent"), "patient_inventory")
        self.assertEqual(updated.get("last_data_context_type"), "patient_inventory")
        self.assertTrue(updated.get("last_patient_inventory"))

    def test_visualization_recommendation_after_inventory_is_advisory_only(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="ok si ces donnees ne sont pas des valeurs transformables, recommande-moi une visualisation qui correspond a ce type de donnees",
            index_dir="data/indexes",
            previous_structured_evidence_pack=None,
            previous_context_intent="response_transform_no_context",
            previous_data_context_intent="patient_inventory",
            previous_data_context_type="patient_inventory",
            previous_has_patient_inventory=True,
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("inventaire", answer)
        self.assertIn("patients", answer)
        self.assertIn("rapports", answer)
        self.assertTrue(("table" in answer) or ("cartes" in answer))
        self.assertTrue(("accordeon" in answer) or ("timeline" in answer) or ("accordéon" in answer))
        self.assertNotIn("format visuel demande", answer)
        self.assertNotIn("j affiche donc", answer)
        self.assertNotIn("bonjour", answer)
        self.assertNotIn("analytes", answer)
        self.assertIsNone(result.get("visualization"))
        self.assertIsNone(result.get("chart_data"))

    def test_inventory_visualization_render_with_context(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        patients = [
            {"patient": "PAT_000001", "report_count": 16, "reports": [{"doc_id": "report_1", "source_url": "/api/documents/report_1/pdf"}]},
            {"patient": "PAT_000002", "report_count": 16, "reports": [{"doc_id": "report_16", "source_url": "/api/documents/report_16/pdf"}]},
        ]
        result = run_generation(
            query="affiche ça dans liste accordéon pour ouvrir les rapports de chaque patient",
            index_dir="data/indexes",
            previous_structured_evidence_pack=None,
            previous_context_intent="response_transform_no_context",
            previous_data_context_intent="patient_inventory",
            previous_data_context_type="patient_inventory",
            previous_has_patient_inventory=True,
            previous_patient_inventory=patients,
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("accordéon", answer)
        self.assertNotIn("cartes patient", answer)
        self.assertNotIn("bonjour", answer)
        self.assertTrue(result.get("patients"))
        self.assertEqual(((result.get("inventory_view") or {}).get("type")), "report_accordion")
        self.assertIsNone(result.get("visualization"))
        self.assertIsNone(result.get("chart_data"))

    def test_inventory_visualization_render_filterable_table(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        patients = [
            {"patient": "PAT_000001", "report_count": 16, "reports": [{"doc_id": "report_1", "source_url": "/api/documents/report_1/pdf"}]},
            {"patient": "PAT_000002", "report_count": 16, "reports": [{"doc_id": "report_16", "source_url": "/api/documents/report_16/pdf"}]},
        ]
        result = run_generation(
            query="ok affiche ça dans une table filtrable par patient, date ou nom de fichier",
            index_dir="data/indexes",
            previous_structured_evidence_pack=None,
            previous_context_intent="response_transform_no_context",
            previous_data_context_intent="patient_inventory",
            previous_data_context_type="patient_inventory",
            previous_has_patient_inventory=True,
            previous_patient_inventory=patients,
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("table structurée", answer)
        self.assertNotIn("cartes patient", answer)
        self.assertTrue(result.get("patients"))
        self.assertEqual(((result.get("inventory_view") or {}).get("type")), "filterable_table")
        self.assertIsNone(result.get("visualization"))
        self.assertIsNone(result.get("chart_data"))

    def test_inventory_visualization_render_timeline_fallback(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        patients = [
            {"patient": "PAT_000001", "report_count": 16, "reports": [{"doc_id": "report_1", "source_url": "/api/documents/report_1/pdf"}]},
            {"patient": "PAT_000002", "report_count": 16, "reports": [{"doc_id": "report_16", "source_url": "/api/documents/report_16/pdf"}]},
        ]
        result = run_generation(
            query="ok affiche ça en timeline documentaire",
            index_dir="data/indexes",
            previous_structured_evidence_pack=None,
            previous_context_intent="response_transform_no_context",
            previous_data_context_intent="patient_inventory",
            previous_data_context_type="patient_inventory",
            previous_has_patient_inventory=True,
            previous_patient_inventory=patients,
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("pas encore implémentée", answer)
        self.assertEqual(answer.count("pas encore implémentée"), 1)
        self.assertEqual(answer.count("vue cartes patient"), 1)
        self.assertEqual(((result.get("inventory_view") or {}).get("type")), "document_timeline")
        self.assertTrue(result.get("patients"))
        self.assertIsNone(result.get("visualization"))
        self.assertIsNone(result.get("chart_data"))

    def test_inventory_visualization_render_without_context(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="affiche avec des cartes patient",
            index_dir="data/indexes",
            previous_structured_evidence_pack=None,
            previous_context_intent="small_talk",
            previous_data_context_intent="",
            previous_data_context_type="",
            previous_has_patient_inventory=False,
            previous_patient_inventory=None,
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("inventaire patient recent", answer.replace("récent", "recent"))
        self.assertNotIn("bonjour", answer)
        self.assertIsNone(result.get("visualization"))
        self.assertIsNone(result.get("chart_data"))

    def test_visualization_recommendation_for_qualitative_comment(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="quelle visualisation recommandes-tu à ce commentaire ?",
            index_dir="data/indexes",
            previous_structured_evidence_pack=None,
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_has_patient_inventory=False,
            previous_patient_inventory=None,
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("commentaire", answer)
        self.assertIn("tableau texte", answer)
        self.assertNotIn("cartes patient", answer)
        self.assertNotIn("accordéon", answer)
        self.assertIsNone(result.get("visualization"))
        self.assertIsNone(result.get("chart_data"))


if __name__ == "__main__":
    unittest.main()
