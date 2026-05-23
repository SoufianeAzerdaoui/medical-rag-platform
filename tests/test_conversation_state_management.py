from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


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
    from scripts.generation.conversation_state_utils import migrate_conversation_state
else:
    from scripts.generation.conversation_state_utils import migrate_conversation_state


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
        self.assertGreaterEqual(int(st.get("state_version") or 0), 2)

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
        self.assertEqual((st.get("last_doc_scope") or {}).get("doc_ids"), ["report_31"])

        second_generation = {
            "generation_mode": "deterministic_response_transform",
            "query_understanding": {"intent": "response_transform"},
            "answer": "Je n’ai pas de résultats biologiques numériques récents...",
            "structured_evidence_pack": {},
        }
        _update_conversation_state(chat_id, st, second_generation, "affiche ça en radar")
        st2 = _CONVERSATION_STATE[chat_id]
        self.assertEqual((st2.get("last_doc_scope") or {}).get("doc_ids"), ["report_31"])

    def test_followup_analyte_updates_transformable_pack_to_last_displayed_result(self) -> None:
        chat_id = "ut-state-followup-tshus"
        first_generation = {
            "query_understanding": {"intent": "doc_scoped_results", "requested_doc_ids": ["report_16"], "requested_analytes": ["acth"]},
            "answer": "ACTH ...",
            "structured_evidence_pack": {
                "requested_doc_ids": ["report_16"],
                "requested_analytes": ["acth"],
                "evidences": [{"analyte": "ACTH", "analyte_norm": "acth", "value_numeric": 23.0, "doc_id": "report_16", "source": "report (16).pdf — page 1, ligne 1"}],
            },
            "displayed_evidences": [{"analyte": "ACTH", "analyte_norm": "acth", "value_numeric": 23.0, "doc_id": "report_16", "source": "report (16).pdf — page 1, ligne 1"}],
            "sources": [{"doc_id": "report_16", "label": "report (16).pdf — page 1, ligne 1"}],
        }
        _update_conversation_state(chat_id, {}, first_generation, "montre ACTH du dernier rapport")
        second_generation = {
            "query_understanding": {"intent": "doc_scoped_results", "requested_doc_ids": ["report_16"], "requested_analytes": ["tshus"]},
            "answer": "TSHus ...",
            "structured_evidence_pack": {
                "requested_doc_ids": ["report_16"],
                "requested_analytes": ["tshus"],
                "evidences": [
                    {"analyte": "ACTH", "analyte_norm": "acth", "value_numeric": 23.0, "doc_id": "report_16", "source": "report (16).pdf — page 1, ligne 1"},
                    {"analyte": "TSHus", "analyte_norm": "tshus", "value_numeric": 55.0, "doc_id": "report_16", "source": "report (16).pdf — page 1, ligne 4"},
                ],
            },
            # What matters for next transform is what was shown to the user now: TSHus only.
            "displayed_evidences": [{"analyte": "TSHus", "analyte_norm": "tshus", "value_numeric": 55.0, "doc_id": "report_16", "source": "report (16).pdf — page 1, ligne 4"}],
            "sources": [{"doc_id": "report_16", "label": "report (16).pdf — page 1, ligne 4"}],
        }
        _update_conversation_state(chat_id, _CONVERSATION_STATE[chat_id], second_generation, "et TSHus ?")
        st = _CONVERSATION_STATE[chat_id]
        pack = st.get("last_transformable_evidence_pack") or {}
        displayed_pack = st.get("last_displayed_evidence_pack") or {}
        evs = list(pack.get("evidences") or [])
        displayed_evs = list(displayed_pack.get("evidences") or [])
        self.assertEqual(len(evs), 1)
        self.assertEqual(len(displayed_evs), 1)
        self.assertEqual(str(evs[0].get("analyte_norm") or "").lower(), "tshus")
        self.assertEqual(str(displayed_evs[0].get("analyte_norm") or "").lower(), "tshus")
        self.assertNotEqual(str(evs[0].get("analyte_norm") or "").lower(), "acth")

    def test_unstructured_followup_with_numeric_displayed_updates_transformable_pack(self) -> None:
        chat_id = "ut-state-unstructured-followup"
        _CONVERSATION_STATE[chat_id] = {
            "last_intent": "doc_scoped_results",
            "last_data_context_type": "biological_numeric_results",
            "last_transformable_evidence_pack": {
                "evidences": [{"analyte": "ACTH", "analyte_norm": "acth", "value_numeric": 23.0, "doc_id": "report_16"}]
            },
        }
        generation = {
            "query_understanding": {"intent": "unstructured", "requested_analytes": ["tshus"], "requested_doc_ids": ["report_16"]},
            "answer": "TSHus = 55,00 mUI/L",
            "displayed_evidences": [
                {
                    "analyte": "TSHus",
                    "analyte_norm": "tshus",
                    "value_numeric": 55.0,
                    "value_raw": "55,00",
                    "unit": "mUI/L",
                    "reference_range": "0,35 à 4,94 mUI/L",
                    "interpretation_status": "above_reference",
                    "doc_id": "report_16",
                    "source": "report (16).pdf — page 1, ligne 4",
                    "page_number": 1,
                    "row_index": 4,
                }
            ],
            "structured_evidence_pack": {
                "evidences": [
                    {"analyte": "ACTH", "analyte_norm": "acth", "value_numeric": 23.0, "doc_id": "report_16"},
                    {"analyte": "TSHus", "analyte_norm": "tshus", "value_numeric": 55.0, "doc_id": "report_16"},
                ]
            },
            "sources": [{"doc_id": "report_16", "label": "report (16).pdf — page 1, ligne 4"}],
        }
        _update_conversation_state(chat_id, _CONVERSATION_STATE[chat_id], generation, "et TSHus ?")
        st = _CONVERSATION_STATE[chat_id]
        self.assertEqual(st.get("last_data_context_type"), "biological_numeric_results")
        pack = st.get("last_transformable_evidence_pack") or {}
        evs = list(pack.get("evidences") or [])
        self.assertEqual(len(evs), 1)
        self.assertEqual(str(evs[0].get("analyte_norm") or "").lower(), "tshus")
        self.assertIn("55", str(evs[0].get("current_value") or ""))
        self.assertIn("4,94", str(evs[0].get("reference") or ""))
        self.assertEqual(str(evs[0].get("technical_status_code") or ""), "above_reference")
        self.assertEqual(int(evs[0].get("row") or 0), 4)

    def test_transform_pack_preserves_followup_tshus_fidelity(self) -> None:
        from scripts.generation.generate_answer import _build_response_transform_pack, parse_query_understanding

        qu = parse_query_understanding("affiche ça en graphique")
        prev_pack = {
            "intent": "doc_scoped_results",
            "output_format": "list",
            "evidences": [
                {
                    "doc_id": "report_16",
                    "analyte": "TSHus",
                    "analyte_norm": "tshus",
                    "value_raw": "55,00",
                    "value_numeric": 55.0,
                    "unit": "mUI/L",
                    "reference_range": "0,35 à 4,94 mUI/L",
                    "interpretation_status": "above_reference",
                    "source": "report (16).pdf — page 1, ligne 4",
                    "source_pdf": "report (16).pdf",
                    "page_number": 1,
                    "row_index": 4,
                }
            ],
        }
        transformed = _build_response_transform_pack(query="affiche ça en graphique", query_understanding=qu, previous_pack=prev_pack)
        evs = list(transformed.get("evidences") or [])
        self.assertEqual(len(evs), 1)
        ev = evs[0]
        self.assertEqual(str(ev.get("analyte_norm") or "").lower(), "tshus")
        self.assertIn("55", str(ev.get("current_value") or ""))
        self.assertIn("4,94", str(ev.get("reference") or ""))
        self.assertEqual(str(ev.get("technical_status_code") or ""), "above_reference")
        self.assertIn("au-dessus de la référence", str(ev.get("technical_status") or ""))
        self.assertEqual(int(ev.get("row") or 0), 4)

    def test_normalize_result_status_mapping(self) -> None:
        from scripts.generation.generate_answer import normalize_result_status

        self.assertEqual(
            normalize_result_status({"technical_status_code": "above_reference"})["display_status"],
            "au-dessus de la référence",
        )
        self.assertEqual(
            normalize_result_status({"technical_status_code": "within_reference"})["display_status"],
            "dans la référence",
        )
        self.assertEqual(
            normalize_result_status({"technical_status_code": "below_reference"})["display_status"],
            "en dessous de la référence",
        )
        self.assertEqual(
            normalize_result_status({"technical_status_code": "normal"})["display_status"],
            "dans la référence",
        )

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

    def test_qualitative_context_overrides_old_inventory_context(self) -> None:
        chat_id = "ut-state-qual-overrides-inventory"
        _CONVERSATION_STATE[chat_id] = {
            "last_data_context_type": "patient_inventory",
            "last_data_context_intent": "patient_inventory",
            "last_patient_inventory": [{"patient": "PAT_000001"}],
        }
        generation = {
            "query_understanding": {
                "intent": "comment_without_measured_value",
                "requested_context_type": "medical_qualitative_comment",
            },
            "answer": "Voici le commentaire retrouvé sur la troponine...",
            "structured_evidence_pack": {
                "evidences": [
                    {
                        "result_kind": "comment",
                        "subject": "Troponine",
                        "display_comment_text": "Valeur seuil au 99ème percentile : 26 ng/l.",
                        "source": "report (18).pdf — page 1, ligne 1",
                        "viewer_url": "/viewer/pdf?doc_id=report_18&page=1",
                    }
                ]
            },
        }
        _update_conversation_state(chat_id, _CONVERSATION_STATE[chat_id], generation, "montre le commentaire sur la troponine")
        st = _CONVERSATION_STATE[chat_id]
        self.assertEqual(st.get("last_data_context_type"), "medical_qualitative_comment")
        self.assertTrue(isinstance(st.get("last_qualitative_evidence_pack"), dict))
        self.assertIsNone(st.get("last_transformable_evidence_pack"))
        self.assertTrue(st.get("last_patient_inventory"))

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

    def test_direct_analyte_request_after_inventory_does_not_stay_in_transform_mode(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        try:
            result = run_generation(
                query="donne moi le resultat de AMH",
                index_dir="data/indexes",
                previous_structured_evidence_pack=None,
                previous_context_intent="patient_inventory",
                previous_data_context_intent="patient_inventory",
                previous_data_context_type="patient_inventory",
                previous_has_patient_inventory=True,
                previous_patient_inventory=[{"patient": "PAT_000001"}],
            )
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("pas de resultats biologiques numeriques recents a transformer", answer)
        self.assertNotEqual(str(result.get("generation_mode") or ""), "deterministic_response_transform")

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

    def test_state_migration_contract_defaults(self) -> None:
        migrated = migrate_conversation_state({}, conversation_id="ut-state-migrate")
        self.assertEqual(migrated.get("conversation_id"), "ut-state-migrate")
        self.assertEqual(migrated.get("last_data_context_type"), "none")
        self.assertIn("state_version", migrated)

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

    def test_inventory_radar_request_explains_non_transformable_then_fallback_cards(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        previous_inventory = [
            {
                "patient": "PAT_000001",
                "report_count": 2,
                "report_range_label": "report.pdf → report (1).pdf",
                "reports": [
                    {"doc_id": "report", "filename": "report.pdf", "viewer_url": "/viewer/pdf?doc_id=report"},
                    {"doc_id": "report_1", "filename": "report (1).pdf", "viewer_url": "/viewer/pdf?doc_id=report_1"},
                ],
            }
        ]
        result = run_generation(
            query="affiche ça en radar chart",
            index_dir="data/indexes",
            previous_context_intent="patient_inventory",
            previous_data_context_type="patient_inventory",
            previous_has_patient_inventory=True,
            previous_patient_inventory=previous_inventory,
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("n’est pas transformable en radar chart", answer)
        self.assertIn("vue d’inventaire adaptée", answer)

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

    def test_inventory_recommendation_then_deictic_table(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        patients = [
            {"patient": "PAT_000001", "report_count": 16, "reports": [{"doc_id": "report_1", "source_url": "/api/documents/report_1/pdf"}]},
            {"patient": "PAT_000002", "report_count": 16, "reports": [{"doc_id": "report_16", "source_url": "/api/documents/report_16/pdf"}]},
        ]
        result = run_generation(
            query="affiche ça en table",
            index_dir="data/indexes",
            previous_structured_evidence_pack=None,
            previous_context_intent="visualization_recommendation",
            previous_data_context_intent="patient_inventory",
            previous_data_context_type="patient_inventory",
            previous_has_patient_inventory=True,
            previous_patient_inventory=patients,
        )
        answer = str(result.get("answer") or "").lower()
        self.assertTrue(result.get("patients"))
        self.assertEqual(((result.get("inventory_view") or {}).get("type")), "filterable_table")
        self.assertIsNone(result.get("visualization"))
        self.assertIsNone(result.get("chart_data"))
        self.assertNotIn("pus", answer)
        self.assertNotIn("résidus alimentaires", answer)

    def test_deictic_table_after_biological_context_reuses_transformable(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        prev_pack = {
            "evidences": [
                {
                    "doc_id": "report_16",
                    "analyte": "ACTH",
                    "analyte_norm": "acth",
                    "current_value": "23,00",
                    "unit": "pg/ml",
                    "reference": "4,70 - 48,80 pg/ml",
                    "technical_status_code": "within_reference",
                    "technical_status": "dans la référence",
                    "source": "report (16).pdf — page 1, ligne 1",
                },
                {
                    "doc_id": "report_16",
                    "analyte": "TSHus",
                    "analyte_norm": "tshus",
                    "current_value": "55,00",
                    "unit": "mUI/L",
                    "reference": "0,35 à 4,94 mUI/l",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                    "source": "report (16).pdf — page 1, ligne 4",
                },
            ]
        }
        result = run_generation(
            query="affiche ça en table",
            index_dir="data/indexes",
            previous_structured_evidence_pack=prev_pack,
            previous_context_intent="doc_scoped_results",
            previous_data_context_intent="doc_scoped_results",
            previous_data_context_type="biological_numeric_results",
            previous_doc_scope=["report_16"],
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("acth", answer)
        self.assertIn("tsh", answer)
        self.assertIsNone(result.get("visualization"))
        self.assertIsNone(result.get("chart_data"))

    def test_deictic_graph_after_biological_context_no_crash(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        prev_pack = {
            "evidences": [
                {
                    "doc_id": "report_16",
                    "analyte": "ACTH",
                    "analyte_norm": "acth",
                    "current_value": "23,00",
                    "unit": "pg/ml",
                    "reference": "4,70 - 48,80 pg/ml",
                    "technical_status_code": "within_reference",
                    "technical_status": "dans la référence",
                    "source": "report (16).pdf — page 1, ligne 1",
                    "source_pdf": "report (16).pdf",
                    "page": 1,
                    "row": 1,
                },
                {
                    "doc_id": "report_16",
                    "analyte": "TSHus",
                    "analyte_norm": "tshus",
                    "current_value": "55,00",
                    "unit": "mUI/L",
                    "reference": "0,35 à 4,94 mUI/l",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                    "source": "report (16).pdf — page 1, ligne 4",
                    "source_pdf": "report (16).pdf",
                    "page": 1,
                    "row": 4,
                },
            ]
        }
        result = run_generation(
            query="affiche ça en graphique",
            index_dir="data/indexes",
            previous_structured_evidence_pack=prev_pack,
            previous_context_intent="doc_scoped_results",
            previous_data_context_intent="doc_scoped_results",
            previous_data_context_type="biological_numeric_results",
            previous_doc_scope=["report_16"],
        )
        self.assertIsInstance(result, dict)
        self.assertNotEqual(str(((result.get("query_understanding") or {}).get("intent") or "")).strip().lower(), "unknown")
        self.assertEqual(str(((result.get("query_understanding") or {}).get("intent") or "")).strip().lower(), "response_transform")
        self.assertTrue(isinstance(result.get("sources"), list))
        answer = str(result.get("answer") or "").lower()
        self.assertIn("tsh", answer)
        self.assertNotIn("pus", answer)

    def test_deictic_table_without_context_returns_no_context_message(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="affiche ça en table",
            index_dir="data/indexes",
            previous_structured_evidence_pack=None,
            previous_context_intent="",
            previous_data_context_intent="",
            previous_data_context_type="none",
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("contexte précédent exploitable", answer)
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

    def test_visualization_recommendation_prefers_qualitative_over_inventory(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="quelle visualisation recommandes-tu à ce commentaire ?",
            index_dir="data/indexes",
            previous_structured_evidence_pack=None,
            previous_context_intent="response_transform_no_context",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_has_patient_inventory=True,
            previous_patient_inventory=[{"patient": "PAT_000001"}],
            previous_qualitative_evidence_pack={
                "evidences": [
                    {
                        "result_kind": "comment",
                        "subject": "Troponine",
                        "display_comment_text": "Valeur seuil au 99ème percentile : 26 ng/l.",
                    }
                ]
            },
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("carte d’information médicale".lower(), answer)
        self.assertIn("bloc commentaire sourcé".lower(), answer)
        self.assertIn("tableau texte", answer)
        self.assertIn("note interprétative".lower(), answer)
        self.assertNotIn("cartes patient", answer)
        self.assertNotIn("accordéon", answer)
        self.assertNotIn("timeline documentaire", answer)
        self.assertNotIn("graphique en barres", answer)
        self.assertNotIn("radar", answer)

    def test_qualitative_comment_render_text_table(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "subject": "Troponine",
                    "display_comment_text": "Valeur seuil au 99ème percentile : 26 ng/l.",
                    "source": "report (18).pdf — page 1, ligne 1",
                    "viewer_url": "/viewer/pdf?doc_id=report_18&page=1",
                }
            ]
        }
        result = run_generation(
            query="ok affiche ce commentaire dans un tableau texte : sujet, commentaire, source cliquable",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
        )
        answer = str(result.get("answer") or "")
        self.assertIn("| Sujet | Commentaire | Source |", answer)
        self.assertIn("](/viewer/pdf?doc_id=report_18&page=1)", answer)
        self.assertIn("Troponine", answer)
        self.assertIn("report (18).pdf — page 1, ligne 1", answer)
        self.assertNotIn("sqlite_deterministic", answer)
        self.assertNotIn("| Commentaire médical |", answer)
        self.assertNotIn("Bloc commentaire sourcé", answer)
        self.assertFalse(result.get("sources"))
        self.assertIsNone(result.get("visualization"))
        self.assertIsNone(result.get("chart_data"))

    def test_qualitative_comment_render_interpretive_note(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "subject": "Troponine",
                    "display_comment_text": "Valeur seuil au 99ème percentile : 26 ng/l.",
                    "source": "report (18).pdf — page 1, ligne 1",
                    "viewer_url": "/viewer/pdf?doc_id=report_18&page=1",
                }
            ]
        }
        result = run_generation(
            query="ok affiche ce commentaire dans un encadré de note interprétative",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
        )
        answer = str(result.get("answer") or "")
        self.assertIn("Note interprétative", answer)
        self.assertNotIn("Bloc commentaire sourcé", answer)
        self.assertIn("](/viewer/pdf?doc_id=report_18&page=1)", answer)
        self.assertFalse(result.get("sources"))
        self.assertIsNone(result.get("visualization"))
        self.assertIsNone(result.get("chart_data"))

    def test_qualitative_comment_render_block_uses_displayed_subject_not_generic_comment(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "analyte": "Commentaire",
                    "display_comment_text": "Valeur seuil au 99ème percentile : 26 ng/l.",
                    "source": "report (18).pdf — page 1, ligne 1",
                    "source_pdf": "report (18).pdf",
                    "page": 1,
                    "row": 1,
                }
            ]
        }
        result = run_generation(
            query="affiche ce commentaire dans un bloc commentaire sourcé",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
            previous_displayed_context={"context_type": "medical_qualitative_comment", "subject": "Troponine"},
        )
        answer = str(result.get("answer") or "")
        self.assertIn("Sujet : Troponine", answer)
        self.assertNotIn("Sujet : Commentaire", answer)

    def test_source_followup_exact_wording_and_subject(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="donne-moi la source exacte de ce commentaire troponine",
            index_dir="data/indexes",
            previous_displayed_context={
                "context_type": "medical_qualitative_comment",
                "subject": "Troponine",
                "sources": [
                    {
                        "label": "report (18).pdf — page 1, ligne 1",
                        "source_pdf": "report (18).pdf",
                        "page": 1,
                        "line": 1,
                    }
                ],
            },
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("source exacte", answer)
        self.assertIn("troponine", answer)
        self.assertIn("report (18).pdf — page 1, ligne 1", str(result.get("answer") or ""))
        self.assertNotIn("sur la commentaire", answer)
        self.assertNotIn("ce résultat", answer)

    def test_qualitative_comment_render_no_fake_clickable_source(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "subject": "Troponine",
                    "display_comment_text": "Valeur seuil au 99ème percentile : 26 ng/l.",
                    "source": "report (18).pdf — page 1, ligne 1",
                }
            ]
        }
        result = run_generation(
            query="affiche ce commentaire dans un tableau texte : sujet, commentaire, source cliquable",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
        )
        answer = str(result.get("answer") or "")
        self.assertIn("source non cliquable disponible uniquement en texte", answer.lower())
        self.assertNotIn("](/viewer/", answer)
        self.assertIn("report (18).pdf — page 1, ligne 1", answer)
        self.assertNotIn("sqlite_deterministic", answer)

    def test_qualitative_correction_followup_non_dans_une_table_keeps_context(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "subject": "Commentaire médical",
                    "display_comment_text": "Valeur seuil : 20 mUI/l (Indicative car pas de signification diagnostique : sécrétion pulsatile)",
                    "source": "report (20).pdf — page 1, ligne 1",
                    "source_pdf": "report (20).pdf",
                    "page": 1,
                    "row": 1,
                }
            ]
        }
        result = run_generation(
            query="non, dans une table",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
            previous_displayed_context={"context_type": "medical_qualitative_comment", "subject": "Commentaire médical"},
        )
        answer = str(result.get("answer") or "")
        self.assertIn("| Sujet | Commentaire | Source |", answer)
        self.assertIn("20 mUI/l", answer)
        self.assertIn("report (20).pdf — page 1, ligne 1", answer)
        self.assertNotIn("PIGF", answer)
        self.assertNotIn("report (1).pdf", answer)

    def test_last_qualitative_pack_preserves_subject_and_pdf_source(self) -> None:
        chat_id = "ut-state-qualitative-preserve-source"
        generation = {
            "query_understanding": {
                "intent": "comment_without_measured_value",
                "requested_context_type": "medical_qualitative_comment",
            },
            "answer": "Voici le commentaire retrouvé sur la troponine...",
            "structured_evidence_pack": {
                "evidences": [
                    {
                        "result_kind": "comment",
                        "subject": "Troponine",
                        "display_comment_text": "Valeur seuil au 99ème percentile : 26 ng/l.",
                        "source_pdf": "report (18).pdf",
                        "page": 1,
                        "row": 1,
                        "source": "report (18).pdf — page 1, ligne 1",
                    }
                ]
            },
        }
        _update_conversation_state(chat_id, {}, generation, "montre le commentaire sur la troponine")
        st = _CONVERSATION_STATE[chat_id]
        qp = st.get("last_qualitative_evidence_pack") or {}
        evs = list(qp.get("evidences") or [])
        self.assertTrue(evs)
        self.assertEqual(str(evs[0].get("subject") or ""), "Troponine")
        self.assertEqual(str(evs[0].get("source_pdf") or ""), "report (18).pdf")

    def test_source_followup_uses_last_displayed_context_no_retrieval(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="d'où vient ce commentaire ?",
            index_dir="data/indexes",
            previous_displayed_context={
                "context_type": "medical_qualitative_comment",
                "subject": "Troponine",
                "sources": [
                    {
                        "label": "report (18).pdf — page 1, ligne 1",
                        "source_pdf": "report (18).pdf",
                        "page": 1,
                        "line": 1,
                        "viewer_url": "/viewer/pdf?doc_id=report_18&page=1",
                    }
                ],
            },
        )
        answer = str(result.get("answer") or "")
        self.assertIn("troponine", answer.lower())
        self.assertIn("report (18).pdf — page 1, ligne 1", answer)
        self.assertNotIn("report (28).pdf", answer)
        self.assertEqual((result.get("retrieval") or {}).get("answerability", {}).get("reason"), "source_followup_no_retrieval")

    def test_resolution_arbitration_prefers_deictic_when_resolved(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")

        result = run_generation(
            query="d'où vient ce commentaire ?",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_displayed_context={
                "context_type": "medical_qualitative_comment",
                "subject": "Troponine",
                "sources": [
                    {
                        "label": "report (18).pdf — page 1, ligne 1",
                        "source_pdf": "report (18).pdf",
                        "doc_id": "report_18",
                        "page": 1,
                        "line": 1,
                    }
                ],
            },
        )
        dbg = dict(result.get("debug") or {})
        arb = dict(dbg.get("resolution_arbitration") or {})
        self.assertEqual(str(arb.get("chosen") or ""), "deictic")
        self.assertIn("priority_rule", arb)

    def test_source_followup_falls_back_to_previous_qualitative_pack_source(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="d'où vient ce commentaire ?",
            index_dir="data/indexes",
            previous_displayed_context={
                "context_type": "medical_qualitative_comment",
                "subject": "Troponine",
                "sources": [],
            },
            previous_qualitative_evidence_pack={
                "evidences": [
                    {
                        "subject": "Troponine",
                        "source": "report (18).pdf — page 1, ligne 1",
                        "source_pdf": "report (18).pdf",
                        "page": 1,
                        "row": 1,
                    }
                ]
            },
        )
        answer = str(result.get("answer") or "")
        self.assertIn("report (18).pdf — page 1, ligne 1", answer)
        self.assertNotIn("source non disponible", answer.lower())

    def test_qualitative_graph_request_explicit_refusal_and_textual_view(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "subject": "Troponine",
                    "display_comment_text": "Valeur seuil au 99ème percentile : 26 ng/l.",
                    "source": "report (18).pdf — page 1, ligne 1",
                }
            ]
        }
        result = run_generation(
            query="affiche ça en graphique",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("donnée qualitative textuelle", answer)
        self.assertIn("vue textuelle sourcée", answer)
        self.assertIsNone(result.get("chart_data"))

    def test_context_summary_qualitative_in_3_points_uses_context_no_retrieval(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "subject": "Troponine",
                    "display_comment_text": (
                        "Valeur seuil au 99ème percentile : 26 ng/l. "
                        "Attention : Elévation de la troponine dans des situations autres que le SCA."
                    ),
                    "source": "report (18).pdf — page 1, ligne 1",
                    "source_pdf": "report (18).pdf",
                    "page": 1,
                    "row": 1,
                }
            ]
        }
        result = run_generation(
            query="résume ce commentaire en 3 points",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
            previous_displayed_context={"context_type": "medical_qualitative_comment", "subject": "Troponine"},
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("résumé en 3 points", answer)
        self.assertIn("1.", answer)
        self.assertIn("2.", answer)
        self.assertTrue(("3." in answer) or ("je ne peux extraire que" in answer))
        # Point 1 and 2 should not be duplicated verbatim.
        lines = [ln.strip() for ln in str(result.get("answer") or "").splitlines() if ln.strip()]
        p1 = next((ln for ln in lines if ln.startswith("1.")), "")
        p2 = next((ln for ln in lines if ln.startswith("2.")), "")
        self.assertTrue(p1 and p2 and p1 != p2)
        self.assertNotIn("sqlite_deterministic", answer)
        self.assertNotIn("source :", answer)
        sources = list(result.get("sources") or [])
        self.assertTrue(any("report (18).pdf" in str(s.get("label") or "") for s in sources))
        self.assertEqual((result.get("retrieval") or {}).get("answerability", {}).get("reason"), "qualitative_comment_summary_no_retrieval")

    def test_context_summary_numeric_single_uses_previous_numeric_context(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "analyte": "TSHus",
                    "current_value": "55,00",
                    "value_raw": "55,00",
                    "value_numeric": 55.0,
                    "unit": "mUI/L",
                    "reference_range": "0,35 à 4,94 mUI/l",
                    "technical_status": "au-dessus de la référence",
                    "source_pdf": "report (16).pdf",
                    "page": 1,
                    "line": 4,
                }
            ]
        }
        result = run_generation(
            query="résume ça en 3 points",
            index_dir="data/indexes",
            previous_context_intent="doc_scoped_results",
            previous_data_context_intent="doc_scoped_results",
            previous_data_context_type="biological_numeric_results",
            previous_structured_evidence_pack=pack,
            previous_displayed_evidence_pack=pack,
            previous_displayed_context={"context_type": "biological_numeric_results", "subject": "TSHus"},
        )
        answer = str(result.get("answer") or "")
        self.assertIn("Résumé en 3 points", answer)
        self.assertIn("TSHus", answer)
        self.assertIn("55,00", answer)
        self.assertIn("0,35 à 4,94", answer)
        self.assertEqual((result.get("retrieval") or {}).get("answerability", {}).get("reason"), "context_summary_no_retrieval")

    def test_context_summary_no_context(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="résume ça en 3 points",
            index_dir="data/indexes",
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("je n’ai pas de contexte précédent à résumer".lower(), answer)
        self.assertEqual((result.get("retrieval") or {}).get("answerability", {}).get("status"), "not_required")

    def test_context_summary_qualitative_non_troponine_not_hardcoded(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "subject": "GH",
                    "display_comment_text": "Commentaire : dosage à interpréter dans le contexte clinique et l’âge du patient.",
                    "source_pdf": "report (21).pdf",
                    "page": 1,
                    "line": 2,
                }
            ]
        }
        result = run_generation(
            query="résume ce commentaire en 3 points",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
            previous_displayed_context={"context_type": "medical_qualitative_comment", "subject": "GH"},
        )
        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("troponine", answer)
        self.assertNotIn("sca", answer)

    def test_source_followup_uses_mock_source_no_hardcoded_report(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="d’où vient ce commentaire ?",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_displayed_context={
                "context_type": "medical_qualitative_comment",
                "subject": "Marqueur X",
                "sources": [
                    {
                        "label": "mock_report.pdf — page 2, ligne 9",
                        "source_pdf": "mock_report.pdf",
                        "page": 2,
                        "line": 9,
                    }
                ],
            },
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("mock_report.pdf", answer)
        self.assertNotIn("report (18).pdf", answer)

    def test_same_action_for_subject_reuses_previous_doc_scope(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="fais la même chose pour TSHus",
            index_dir="data/indexes",
            previous_context_intent="doc_scoped_results",
            previous_data_context_intent="doc_scoped_results",
            previous_data_context_type="biological_numeric_results",
            previous_doc_scope=["report_16"],
            previous_displayed_context={"context_type": "biological_numeric_results", "subject": "ACTH"},
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("tshus", answer)
        self.assertIn("report (16).pdf", answer)
        self.assertNotIn("report (1).pdf", answer)
        self.assertNotIn("report (11).pdf", answer)

    def test_qualitative_fiche_from_deictic_request(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "subject": "Troponine",
                    "display_comment_text": "Valeur seuil au 99ème percentile : 26 ng/l.",
                    "source": "report (18).pdf — page 1, ligne 1",
                    "source_pdf": "report (18).pdf",
                    "page": 1,
                    "row": 1,
                }
            ]
        }
        result = run_generation(
            query="mets ça sous forme de fiche",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
            previous_displayed_context={"context_type": "medical_qualitative_comment", "subject": "Troponine"},
        )
        answer = str(result.get("answer") or "")
        self.assertIn("Carte d’information médicale", answer)
        self.assertIn("Sujet : Troponine", answer)
        self.assertNotIn("aucun résultat précédent exploitable", answer.lower())

    def test_qualitative_deictic_table_keeps_all_comments(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "subject": "Commentaire médical",
                    "display_comment_text": "Valeur seuil au 99ème percentile : 26 ng/l.",
                    "source": "report (18).pdf — page 1, ligne 1",
                    "source_pdf": "report (18).pdf",
                    "page": 1,
                    "row": 1,
                },
                {
                    "subject": "Commentaire médical",
                    "display_comment_text": "Valeur seuil : 20 mUI/l (Indicative).",
                    "source": "report (18).pdf — page 1, ligne 2",
                    "source_pdf": "report (18).pdf",
                    "page": 1,
                    "row": 2,
                },
                {
                    "subject": "Commentaire médical",
                    "display_comment_text": "<4,11 IU/ml",
                    "source": "report (18).pdf — page 1, ligne 3",
                    "source_pdf": "report (18).pdf",
                    "page": 1,
                    "row": 3,
                },
            ]
        }
        result = run_generation(
            query="affiche ça en table",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
            previous_displayed_context={"context_type": "medical_qualitative_comment", "subject": "IMMUNOANALYSE"},
        )
        answer = str(result.get("answer") or "")
        self.assertIn("| Sujet | Commentaire | Source |", answer)
        self.assertIn("26 ng/l.", answer)
        self.assertIn("20 mUI/l", answer)
        self.assertIn("<4,11 IU/ml", answer)
        self.assertIn("report (18).pdf — page 1, ligne 1", answer)
        self.assertIn("report (18).pdf — page 1, ligne 2", answer)
        self.assertIn("report (18).pdf — page 1, ligne 3", answer)

    def test_deictic_no_context_guard_no_retrieval(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        result = run_generation(
            query="affiche ça en tableau",
            index_dir="data/indexes",
            previous_displayed_context=None,
            previous_data_context_type="none",
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("je n’ai pas de contexte précédent à reformater".lower(), answer)
        self.assertEqual((result.get("retrieval") or {}).get("answerability", {}).get("reason"), "deictic_no_context_guard")

    def test_result_status_followup_keeps_source(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "analyte": "TSHus",
                    "current_value": "55,00",
                    "unit": "mUI/L",
                    "reference": "0,35 à 4,94 mUI/l",
                    "technical_status": "au-dessus de la référence",
                    "source_pdf": "report (16).pdf",
                    "doc_id": "report_16",
                    "page": 1,
                    "line": 4,
                    "source_label": "report (16).pdf — page 1, ligne 4",
                }
            ]
        }
        result = run_generation(
            query="ce résultat est-il hors référence ?",
            index_dir="data/indexes",
            previous_context_intent="doc_scoped_results",
            previous_data_context_intent="doc_scoped_results",
            previous_data_context_type="biological_numeric_results",
            previous_displayed_evidence_pack=pack,
            previous_displayed_context={"context_type": "biological_numeric_results", "subject": "TSHus"},
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("source : report (16).pdf — page 1, ligne 4".lower(), answer)

    def test_context_summary_numeric_mock_not_acth_tshus(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "analyte": "TEST_X",
                    "analyte_norm": "test_x",
                    "current_value": "12",
                    "value_raw": "12",
                    "value_numeric": 12.0,
                    "unit": "u",
                    "reference_range": "5-10",
                    "technical_status": "au-dessus de la référence",
                    "source_pdf": "mock_report.pdf",
                    "page": 2,
                    "line": 9,
                }
            ]
        }
        result = run_generation(
            query="résume cette valeur en 3 points",
            index_dir="data/indexes",
            previous_context_intent="doc_scoped_results",
            previous_data_context_intent="doc_scoped_results",
            previous_data_context_type="biological_numeric_results",
            previous_structured_evidence_pack=pack,
            previous_displayed_evidence_pack=pack,
            previous_displayed_context={"context_type": "biological_numeric_results", "subject": "TEST_X"},
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("test_x", answer)
        self.assertNotIn("tshus", answer)
        self.assertNotIn("acth", answer)
        self.assertNotIn("report_16", answer)

    def test_context_summary_non_medical_fallback_has_no_troponine_terms(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "subject": "Note libre",
                    "display_comment_text": "Texte court sans détail.",
                    "source_pdf": "mock_report.pdf",
                    "page": 1,
                    "line": 1,
                }
            ]
        }
        result = run_generation(
            query="résume ce commentaire en 5 points",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
            previous_displayed_context={"context_type": "medical_qualitative_comment", "subject": "Note libre"},
        )
        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("troponine", answer)

    def test_list_all_comments_returns_three_unique_without_duplicates(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")

        mock_rows = [
            {
                "chunk_id": "c18-1",
                "doc_id": "report_18",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Valeur seuil au 99ème percentile : 26 ng/l. Attention : Elévation de la troponine...",
                "text_for_keyword": "Commentaire : Valeur seuil au 99ème percentile : 26 ng/l. Attention : Elévation de la troponine...",
                "text_for_embedding": "Commentaire : Valeur seuil au 99ème percentile : 26 ng/l. Attention : Elévation de la troponine...",
                "section": "Commentaire",
                "section_norm": "commentaire",
                "reference_range": "",
                "source_pdf": "report (18).pdf",
                "page_number": 1,
                "row_index": 1,
            },
            {
                "chunk_id": "c18-dup",
                "doc_id": "report_18",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Valeur seuil au 99eme percentile : 26 ng/l Attention : Elevation de la troponine...",
                "text_for_keyword": "Valeur seuil au 99eme percentile : 26 ng/l Attention : Elevation de la troponine...",
                "text_for_embedding": "Valeur seuil au 99eme percentile : 26 ng/l Attention : Elevation de la troponine...",
                "section": "Commentaire",
                "section_norm": "commentaire",
                "reference_range": "",
                "source_pdf": "report (18).pdf",
                "page_number": 1,
                "row_index": None,
            },
            {
                "chunk_id": "c18-low",
                "doc_id": "report_18",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Commentaire. Resume du rapport medical. Type de document : biology_report. Laboratoire : LABORATOIRE.",
                "text_for_keyword": "Commentaire. Resume du rapport medical. Type de document : biology_report. Laboratoire : LABORATOIRE.",
                "text_for_embedding": "Commentaire. Resume du rapport medical. Type de document : biology_report. Laboratoire : LABORATOIRE.",
                "section": "Résumé",
                "section_norm": "resume",
                "reference_range": "",
                "source_pdf": "report (18).pdf",
                "page_number": 1,
                "row_index": 99,
            },
            {
                "chunk_id": "c20-1",
                "doc_id": "report_20",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Valeur seuil : 20 mUI/l (Indicative car pas de signification diagnostique : sécrétion pulsatile).",
                "text_for_keyword": "Commentaire : Valeur seuil : 20 mUI/l (Indicative car pas de signification diagnostique : sécrétion pulsatile).",
                "text_for_embedding": "Commentaire : Valeur seuil : 20 mUI/l (Indicative car pas de signification diagnostique : sécrétion pulsatile).",
                "section": "Commentaire",
                "section_norm": "commentaire",
                "reference_range": "",
                "source_pdf": "report (20).pdf",
                "page_number": 1,
                "row_index": 1,
            },
            {
                "chunk_id": "c20-dup",
                "doc_id": "report_20",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Valeur seuil : 20 mUI/l (indicative car pas de signification diagnostique : secretion pulsatile).",
                "text_for_keyword": "Valeur seuil : 20 mUI/l (indicative car pas de signification diagnostique : secretion pulsatile).",
                "text_for_embedding": "Valeur seuil : 20 mUI/l (indicative car pas de signification diagnostique : secretion pulsatile).",
                "section": "Commentaire",
                "section_norm": "commentaire",
                "reference_range": "",
                "source_pdf": "report (20).pdf",
                "page_number": 1,
                "row_index": None,
            },
            {
                "chunk_id": "c28-1",
                "doc_id": "report_28",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Commentaire : <4,11 IU/ml.",
                "text_for_keyword": "Commentaire : <4,11 IU/ml.",
                "text_for_embedding": "Commentaire : <4,11 IU/ml.",
                "section": "Commentaire",
                "section_norm": "commentaire",
                "reference_range": "",
                "source_pdf": "report (28).pdf",
                "page_number": 1,
                "row_index": 2,
            },
            {
                "chunk_id": "c28-low",
                "doc_id": "report_28",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Laboratory results. Section medicale.",
                "text_for_keyword": "Laboratory results. Section medicale.",
                "text_for_embedding": "Laboratory results. Section medicale.",
                "section": "Résumé",
                "section_norm": "resume",
                "reference_range": "",
                "source_pdf": "report (28).pdf",
                "page_number": 1,
                "row_index": 90,
            },
        ]

        with (
            patch("scripts.generation.generate_answer._fetch_global_comment_rows", return_value=mock_rows),
            patch(
                "scripts.generation.generate_answer.compose_professional_answer",
                return_value={"answer": "placeholder", "mode": "deterministic_professional_fallback", "llm_error": None},
            ),
        ):
            result = run_generation(
                query="liste moi tous les commentaires",
                index_dir="data/indexes",
            )

        answer = str(result.get("answer") or "")
        self.assertIn("Commentaires retrouvés", answer)
        self.assertEqual(answer.count("\n- "), 3)
        self.assertNotIn("Source :", answer)
        self.assertNotIn("Resume du rapport medical", answer)
        self.assertNotIn("Laboratory results", answer)
        src_labels = [str(s.get("label") or "") for s in list(result.get("sources") or [])]
        self.assertEqual(len(src_labels), 3)
        self.assertTrue(any("report (18).pdf" in lbl for lbl in src_labels))
        self.assertTrue(any("report (20).pdf" in lbl for lbl in src_labels))
        self.assertTrue(any("report (28).pdf" in lbl for lbl in src_labels))

    def test_latest_report_comment_fallback_to_latest_doc_with_comment(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")

        # report_30 is considered latest but has no comment rows; fallback should pick report_28.
        scoped_rows: list[dict] = []
        global_rows = [
            {
                "chunk_id": "c18-1",
                "doc_id": "report_18",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Valeur seuil au 99ème percentile : 26 ng/l.",
                "text_for_keyword": "Commentaire : Valeur seuil au 99ème percentile : 26 ng/l.",
                "text_for_embedding": "Commentaire : Valeur seuil au 99ème percentile : 26 ng/l.",
                "section": "Commentaire",
                "section_norm": "commentaire",
                "reference_range": "",
                "source_pdf": "report (18).pdf",
                "page_number": 1,
                "row_index": 1,
            },
            {
                "chunk_id": "c28-1",
                "doc_id": "report_28",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Commentaire : <4,11",
                "unit": "IU/ml",
                "text_for_keyword": "Commentaire : <4,11",
                "text_for_embedding": "Commentaire : <4,11",
                "section": "Commentaire",
                "section_norm": "commentaire",
                "reference_range": "",
                "source_pdf": "report (28).pdf",
                "page_number": 1,
                "row_index": 2,
            },
        ]

        def _fake_fetch_doc_lab_rows(**_: dict) -> list[dict]:
            return scoped_rows

        with (
            patch("scripts.generation.generate_answer._resolve_latest_doc_id", return_value="report_30"),
            patch("scripts.generation.generate_answer._fetch_doc_lab_rows", side_effect=_fake_fetch_doc_lab_rows),
            patch("scripts.generation.generate_answer._fetch_global_comment_rows", return_value=global_rows),
            patch(
                "scripts.generation.generate_answer.compose_professional_answer",
                return_value={"answer": "placeholder", "mode": "deterministic_professional_fallback", "llm_error": None},
            ),
        ):
            result = run_generation(
                query="liste moi le commentaire du dernier rapport",
                index_dir="data/indexes",
            )

        answer = str(result.get("answer") or "")
        self.assertIn("Commentaires retrouvés", answer)
        self.assertIn("<4,11 IU/ml", answer)
        src_labels = [str(s.get("label") or "") for s in list(result.get("sources") or [])]
        self.assertEqual(len(src_labels), 1)
        self.assertTrue(any("report (28).pdf" in lbl for lbl in src_labels))

    def test_single_comment_request_returns_only_one_latest_comment(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")

        mock_rows = [
            {
                "chunk_id": "c18-1",
                "doc_id": "report_18",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Valeur seuil au 99ème percentile : 26 ng/l.",
                "text_for_keyword": "Commentaire : Valeur seuil au 99ème percentile : 26 ng/l.",
                "text_for_embedding": "Commentaire : Valeur seuil au 99ème percentile : 26 ng/l.",
                "section": "Commentaire",
                "section_norm": "commentaire",
                "reference_range": "",
                "source_pdf": "report (18).pdf",
                "page_number": 1,
                "row_index": 1,
            },
            {
                "chunk_id": "c20-1",
                "doc_id": "report_20",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Valeur seuil : 20 mUI/l.",
                "text_for_keyword": "Commentaire : Valeur seuil : 20 mUI/l.",
                "text_for_embedding": "Commentaire : Valeur seuil : 20 mUI/l.",
                "section": "Commentaire",
                "section_norm": "commentaire",
                "reference_range": "",
                "source_pdf": "report (20).pdf",
                "page_number": 1,
                "row_index": 1,
            },
            {
                "chunk_id": "c28-1",
                "doc_id": "report_28",
                "analyte": "Commentaire",
                "analyte_norm": "commentaire",
                "value_raw": "Commentaire : <4,11",
                "unit": "IU/ml",
                "text_for_keyword": "Commentaire : <4,11",
                "text_for_embedding": "Commentaire : <4,11",
                "section": "Commentaire",
                "section_norm": "commentaire",
                "reference_range": "",
                "source_pdf": "report (28).pdf",
                "page_number": 1,
                "row_index": 2,
            },
        ]

        with (
            patch("scripts.generation.generate_answer._fetch_global_comment_rows", return_value=mock_rows),
            patch(
                "scripts.generation.generate_answer.compose_professional_answer",
                return_value={"answer": "placeholder", "mode": "deterministic_professional_fallback", "llm_error": None},
            ),
        ):
            result = run_generation(
                query="liste une seule commentaire",
                index_dir="data/indexes",
            )

        answer = str(result.get("answer") or "")
        self.assertIn("Commentaires retrouvés", answer)
        # Only one numbered item should remain.
        self.assertEqual(answer.count("\n1. **"), 1)
        self.assertNotIn("\n2. **", answer)
        src_labels = [str(s.get("label") or "") for s in list(result.get("sources") or [])]
        self.assertEqual(len(src_labels), 1)
        self.assertTrue(any("report (28).pdf" in lbl for lbl in src_labels))

    def test_context_summary_limitation_singular_wording(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible dans cet environnement: {exc}")
        pack = {
            "evidences": [
                {
                    "subject": "GH",
                    "display_comment_text": "Commentaire unique.",
                    "source_pdf": "report (21).pdf",
                    "page": 1,
                    "line": 2,
                }
            ]
        }
        result = run_generation(
            query="résume ce commentaire en 3 points",
            index_dir="data/indexes",
            previous_context_intent="comment_without_measured_value",
            previous_data_context_intent="comment_without_measured_value",
            previous_data_context_type="medical_qualitative_comment",
            previous_qualitative_evidence_pack=pack,
            previous_displayed_context={"context_type": "medical_qualitative_comment", "subject": "GH"},
        )
        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("1 point distincts", answer)


if __name__ == "__main__":
    unittest.main()
