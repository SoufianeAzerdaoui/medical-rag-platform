from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
GENERATION_ROOT = SCRIPTS_ROOT / "generation"
for root in (SCRIPTS_ROOT, GENERATION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from generate_answer import run_generation
from evaluation.benchmark_llm_writers import _extract_response_fields
from retrieval.models import RetrievalResult, SearchResponse


class _FakeSearchEngine:
    def __init__(self, response: SearchResponse) -> None:
        self._response = response

    def search(self, **_: object) -> SearchResponse:
        return self._response

    def close(self) -> None:
        return None


class _FailIfCalledSearchEngine:
    def search(self, **_: object) -> SearchResponse:
        raise AssertionError("search() should not be called for response_transform")

    def close(self) -> None:
        return None


def _mk_result(*, chunk_id: str, doc_id: str, analyte: str, analyte_norm: str, value_raw: str) -> RetrievalResult:
    metadata = {
        "analyte": analyte,
        "analyte_norm": analyte_norm,
        "value_raw": value_raw,
        "value_numeric": value_raw.replace(",", "."),
        "unit": "uU/mL",
        "reference_range": "4 à 20 µIU/mL",
        "interpretation_status": "within_reference",
        "previous_result_present": 1,
        "previous_result_value_raw": "2,00",
        "source_kind": "chu_text_fallback",
        "row_index": 1,
        "page_number": 1,
    }
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_type="lab_result",
        document_type="lab_report",
        source_pdf=f"docs/{doc_id}.pdf",
        page_number=1,
        text=f"{analyte} {value_raw}",
        text_preview=f"{analyte} {value_raw}",
        metadata=metadata,
        retrieval_mode="keyword",
        match_reason=["fake"],
    )


class TestGenerationDocScope(unittest.TestCase):
    def test_llm_timeout_circuit_opens_only_after_threshold(self) -> None:
        ga = __import__("generate_answer")
        old_enabled = os.environ.get("MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ENABLED")
        old_routes = os.environ.get("MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ROUTES")
        old_threshold = os.environ.get("MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_FAILURE_THRESHOLD")
        route = "doc_scoped_biological_summary"
        model = "qwen2.5:7b-instruct"
        try:
            os.environ["MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ENABLED"] = "1"
            os.environ["MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ROUTES"] = route
            os.environ["MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_FAILURE_THRESHOLD"] = "2"
            ga._clear_llm_timeout_circuit(route, model)
            self.assertFalse(ga._is_llm_timeout_circuit_open(route, model))
            failures = ga._record_llm_timeout_failure(route, model)
            self.assertEqual(failures, 1)
            self.assertFalse(ga._is_llm_timeout_circuit_open(route, model))
            failures = ga._record_llm_timeout_failure(route, model)
            self.assertEqual(failures, 2)
            self.assertTrue(ga._is_llm_timeout_circuit_open(route, model))
            ga._record_llm_timeout_success(route, model)
            self.assertFalse(ga._is_llm_timeout_circuit_open(route, model))
        finally:
            ga._clear_llm_timeout_circuit(route, model)
            if old_enabled is None:
                os.environ.pop("MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ENABLED", None)
            else:
                os.environ["MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ENABLED"] = old_enabled
            if old_routes is None:
                os.environ.pop("MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ROUTES", None)
            else:
                os.environ["MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_ROUTES"] = old_routes
            if old_threshold is None:
                os.environ.pop("MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_FAILURE_THRESHOLD", None)
            else:
                os.environ["MEDICAL_RAG_LLM_TIMEOUT_CIRCUIT_FAILURE_THRESHOLD"] = old_threshold

    def test_doc_scoped_biological_summary_llm_profile_compact_is_stricter(self) -> None:
        ga = __import__("generate_answer")
        qu = ga.parse_query_understanding(
            "Fais une synthèse biologique courte du report 12. Limite-toi à 3 à 5 lignes."
        )
        profile = ga._doc_scoped_biological_summary_llm_profile(qu)
        self.assertEqual(str(profile.get("render_profile") or ""), "compact_biological_summary")
        self.assertEqual(int(profile.get("max_rows") or 0), 5)
        self.assertEqual(int(profile.get("max_abnormal_rows") or 0), 5)
        self.assertEqual(int(profile.get("max_within_rows") or 0), 1)
        self.assertEqual(int(profile.get("timeout_ms") or 0), 70000)
        self.assertEqual(int(profile.get("num_predict") or 0), 120)
        self.assertEqual(int(profile.get("num_ctx_cap") or 0), 2048)

    def test_doc_scoped_biological_summary_llm_profile_editorial_keeps_balanced_budget(self) -> None:
        ga = __import__("generate_answer")
        qu = ga.parse_query_understanding(
            "Fais une synthèse biologique éditoriale du report 12. Rédige un texte naturel et professionnel."
        )
        profile = ga._doc_scoped_biological_summary_llm_profile(qu)
        self.assertEqual(str(profile.get("render_profile") or ""), "editorial_biological_summary")
        self.assertEqual(int(profile.get("max_rows") or 0), 5)
        self.assertEqual(int(profile.get("max_abnormal_rows") or 0), 5)
        self.assertEqual(int(profile.get("max_within_rows") or 0), 1)
        self.assertEqual(int(profile.get("timeout_ms") or 0), 85000)
        self.assertEqual(int(profile.get("num_predict") or 0), 150)

    def test_doc_scoped_biological_summary_rewrites_too_sparse_llm_narrative(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {"analyte": "Acide urique", "technical_status_code": "below_reference", "value_with_unit": "23 mg/L", "reference_short": "25 - 70"},
            {"analyte": "Ammonium", "technical_status_code": "below_reference", "value_with_unit": "20 µg/dL", "reference_short": "35 - 80"},
            {"analyte": "Bilirubine Directe", "technical_status_code": "above_reference", "value_with_unit": "6 mg/L", "reference_short": "0.00 - 5.00"},
            {"analyte": "Créatinine", "technical_status_code": "above_reference", "value_with_unit": "23 mg/L", "reference_short": "4 - 9"},
            {"analyte": "LDH", "technical_status_code": "above_reference", "value_with_unit": "250 UI/L", "reference_short": "125 - 243"},
            {"analyte": "CK-MB", "technical_status_code": "above_reference", "value_with_unit": "40 UI/L", "reference_short": "< 25"},
        ]
        weak_llm = (
            "L'acide urique est bas à 23 mg/L, l'ammonium est bas à 20 µg/dl, et la bilirubine directe est élevé à 6 mg/L. "
            "Conclusion: anomalies métaboliques notables nécessitent un suivi clinique."
        )
        rendered = ga._render_biological_summary_from_contract(
            llm_answer=weak_llm,
            evidences=evidences,
            max_lines=4,
            no_diagnosis=True,
            render_profile="compact_biological_summary",
        )
        low = rendered.lower()
        self.assertNotIn("élevé à 6 mg/l", low)
        self.assertIn("créatinine", low)
        self.assertIn("ldh", low)
        self.assertIn("conclusion technique", low)
        self.assertIn("sans diagnostic", low)

    def test_report12_summary_uses_full_document_rows_and_flags_sparse_candidate(self) -> None:
        ga = __import__("generate_answer")
        qu = ga.parse_query_understanding(
            "Fais une synthèse biologique courte du report 12. Limite-toi à 3 à 5 lignes, mentionne uniquement les anomalies majeures, les résultats dans la référence et une conclusion prudente, sans diagnostic."
        )
        rows = [
            {"analyte": "Acide urique", "technical_status_code": "below_reference", "value_with_unit": "23 mg/L", "reference_short": "25 - 70"},
            {"analyte": "Ammonium", "technical_status_code": "below_reference", "value_with_unit": "20 µg/dL", "reference_short": "35 - 80"},
            {"analyte": "Bilirubine Directe", "technical_status_code": "above_reference", "value_with_unit": "6 mg/L", "reference_short": "0.00 - 5.00"},
            {"analyte": "Créatinine", "technical_status_code": "above_reference", "value_with_unit": "23 mg/L", "reference_short": "4 - 9"},
            {"analyte": "LDH", "technical_status_code": "above_reference", "value_with_unit": "250 UI/L", "reference_short": "125 - 243"},
            {"analyte": "CK-MB", "technical_status_code": "above_reference", "value_with_unit": "40 UI/L", "reference_short": "< 25"},
            {"analyte": "APOLIPOPROTÉINE A1", "technical_status_code": "above_reference", "value_with_unit": "2.3 g/L", "reference_short": "1.1 - 1.6"},
            {"analyte": "ASAT", "technical_status_code": "within_reference", "value_with_unit": "31 UI/L", "reference_short": "10 - 40"},
            {"analyte": "ALAT", "technical_status_code": "within_reference", "value_with_unit": "22 UI/L", "reference_short": "10 - 45"},
        ]
        rendered = ga._build_doc_scoped_biological_summary_answer(
            rows,
            max_lines=4,
            no_diagnosis=True,
            render_profile="compact_biological_summary",
        )
        low = rendered.lower()
        expected_labels = ["bilirubine directe", "créatinine", "ldh", "ck-mb", "apolipoprotéine a1", "acide urique"]
        self.assertGreaterEqual(sum(1 for label in expected_labels if label in low), 4)
        self.assertTrue(ga._doc_scoped_biological_summary_needs_repair(
            answer="AMMONIUM (écart documenté)",
            evidences=rows,
            query_understanding=qu,
        ))
        gate = ga._evaluate_summary_quality_gate(
            answer="AMMONIUM (écart documenté)",
            selected_route="doc_scoped_biological_summary",
            displayed_evidences=rows,
        )
        self.assertIn("summary_too_poor_for_available_facts", gate.get("reasons") or [])

    def test_doc_scoped_biological_summary_selection_keeps_five_abnormal_rows_when_available(self) -> None:
        ga = __import__("generate_answer")
        qu = ga.parse_query_understanding(
            "Fais une synthèse biologique courte du report 12. Limite-toi à 3 à 5 lignes, mentionne uniquement les anomalies majeures, les résultats dans la référence et une conclusion prudente, sans diagnostic."
        )
        structured_pack = {
            "question": "report 12",
            "intent": "doc_scoped_summary",
            "evidences": [
                {"doc_id": "report_12", "analyte": "Acide urique", "current_value": "23", "unit": "mg/L", "reference_range": "25 - 70", "technical_status_code": "below_reference"},
                {"doc_id": "report_12", "analyte": "Ammonium", "current_value": "20", "unit": "µg/dL", "reference_range": "35 - 80", "technical_status_code": "below_reference"},
                {"doc_id": "report_12", "analyte": "Bilirubine Directe", "current_value": "6", "unit": "mg/L", "reference_range": "0.00 - 5.00", "technical_status_code": "above_reference"},
                {"doc_id": "report_12", "analyte": "Créatinine", "current_value": "23", "unit": "mg/L", "reference_range": "4 - 9", "technical_status_code": "above_reference"},
                {"doc_id": "report_12", "analyte": "LDH", "current_value": "250", "unit": "UI/L", "reference_range": "125 - 243", "technical_status_code": "above_reference"},
                {"doc_id": "report_12", "analyte": "CK-MB", "current_value": "40", "unit": "UI/L", "reference_range": "< 25", "technical_status_code": "above_reference"},
                {"doc_id": "report_12", "analyte": "APO A1", "current_value": "2.3", "unit": "g/L", "reference_range": "1.1 - 1.6", "technical_status_code": "above_reference"},
                {"doc_id": "report_12", "analyte": "ASAT", "current_value": "31", "unit": "UI/L", "reference_range": "10 - 40", "technical_status_code": "within_reference"},
            ],
            "rows": [],
        }
        llm_pack, selected_count = ga._build_llm_evidence_pack(
            query_understanding=qu,
            structured_pack=structured_pack,
            selected_route="doc_scoped_biological_summary",
        )
        debug = llm_pack.get("summary_selection_debug") or {}
        self.assertGreaterEqual(int(selected_count or 0), 6)
        self.assertGreaterEqual(int(debug.get("llm_abnormal_rows_count") or 0), 5)
        self.assertGreaterEqual(int(debug.get("llm_within_rows_count") or 0), 1)

    def test_sparse_doc_scoped_summary_forces_repair_and_rejects_warning_only_acceptance(self) -> None:
        ga = __import__("generate_answer")
        qu = ga.parse_query_understanding(
            "Fais une synthèse biologique courte du report 12. Limite-toi à 3 à 5 lignes, mentionne uniquement les anomalies majeures, les résultats dans la référence et une conclusion prudente, sans diagnostic."
        )
        rows = [
            {"analyte": "Bilirubine Directe", "technical_status_code": "above_reference", "value_with_unit": "6 mg/L", "reference_short": "0.00 - 5.00"},
            {"analyte": "Créatinine", "technical_status_code": "above_reference", "value_with_unit": "23 mg/L", "reference_short": "4 - 9"},
            {"analyte": "LDH", "technical_status_code": "above_reference", "value_with_unit": "250 UI/L", "reference_short": "125 - 243"},
            {"analyte": "CK-MB", "technical_status_code": "above_reference", "value_with_unit": "40 UI/L", "reference_short": "< 25"},
            {"analyte": "APO A1", "technical_status_code": "above_reference", "value_with_unit": "2.3 g/L", "reference_short": "1.1 - 1.6"},
            {"analyte": "Ammonium", "technical_status_code": "below_reference", "value_with_unit": "20 µg/dL", "reference_short": "35 - 80"},
            {"analyte": "ASAT", "technical_status_code": "within_reference", "value_with_unit": "31 UI/L", "reference_short": "10 - 40"},
            {"analyte": "ALAT", "technical_status_code": "within_reference", "value_with_unit": "22 UI/L", "reference_short": "10 - 45"},
            {"analyte": "CRP", "technical_status_code": "within_reference", "value_with_unit": "1 mg/L", "reference_short": "< 5"},
        ]
        sparse = "AMMONIUM (écart documenté)"
        self.assertTrue(ga._doc_scoped_biological_summary_needs_repair(answer=sparse, evidences=rows, query_understanding=qu))
        repaired = ga._repair_sparse_doc_scoped_biological_summary_answer(answer=sparse, evidences=rows, query_understanding=qu)
        low = repaired.lower()
        expected_labels = ["bilirubine directe", "créatinine", "ldh", "ck-mb", "apo a1", "ammonium"]
        self.assertGreaterEqual(sum(1 for label in expected_labels if label in low), 4)
        self.assertIn("source", low)

    def test_doc_scoped_repair_requires_material_improvement(self) -> None:
        ga = __import__("generate_answer")
        qu = ga.parse_query_understanding(
            "Fais une synthèse biologique courte du report 12. Limite-toi à 3 à 5 lignes, mentionne uniquement les anomalies majeures, les résultats dans la référence et une conclusion prudente, sans diagnostic."
        )
        rows = [
            {"analyte": "Bilirubine Directe", "technical_status_code": "above_reference", "value_with_unit": "6 mg/L", "reference_short": "0.00 - 5.00"},
            {"analyte": "Créatinine", "technical_status_code": "above_reference", "value_with_unit": "23 mg/L", "reference_short": "4 - 9"},
            {"analyte": "LDH", "technical_status_code": "above_reference", "value_with_unit": "250 UI/L", "reference_short": "125 - 243"},
            {"analyte": "CK-MB", "technical_status_code": "above_reference", "value_with_unit": "40 UI/L", "reference_short": "< 25"},
            {"analyte": "APO A1", "technical_status_code": "above_reference", "value_with_unit": "2.3 g/L", "reference_short": "1.1 - 1.6"},
            {"analyte": "Ammonium", "technical_status_code": "below_reference", "value_with_unit": "20 µg/dL", "reference_short": "35 - 80"},
        ]
        candidate = "Résumé biologique court — Bilirubine Directe (écart documenté), LDH (écart documenté)."
        errors = ga._doc_scoped_biological_summary_repair_errors(
            candidate_answer=candidate,
            repaired_answer=candidate,
            evidences=rows,
            query_understanding=qu,
        )
        self.assertIn("repair_not_materially_different", errors)
        self.assertIn("repair_missing_major_anomaly_coverage", errors)
        self.assertIn("repair_missing_no_within_reference_sentence", errors)

    def test_doc_scoped_biological_summary_rewrite_rejects_generic_clinical_opening(self) -> None:
        ga = __import__("generate_answer")
        self.assertTrue(ga._llm_summary_requires_professional_rewrite(
            "Ouverture clinique: Patient présente des anomalies sur plusieurs marqueurs enzymatiques et métaboliques. Conclusion: Requiert une évaluation clinique approfondie."
        ))
        self.assertFalse(ga._llm_summary_requires_professional_rewrite(
            "Le bilan montre plusieurs écarts biologiques documentés, sans diagnostic."
        ))

    def test_doc_scoped_biological_summary_uses_precise_status_labels(self) -> None:
        ga = __import__("generate_answer")
        rows = [
            {"analyte": "Bilirubine Directe", "technical_status_code": "above_reference", "value_with_unit": "6 mg/L", "reference_short": "0.00 - 5.00"},
            {"analyte": "Ammonium", "technical_status_code": "below_reference", "value_with_unit": "20 µg/dL", "reference_short": "35 - 80"},
            {"analyte": "ASAT", "technical_status_code": "within_reference", "value_with_unit": "31 UI/L", "reference_short": "10 - 40"},
        ]
        rendered = ga._build_doc_scoped_biological_summary_answer(
            rows,
            max_lines=7,
            no_diagnosis=True,
            render_profile="doctor_note_reference_ranges",
        )
        low = rendered.lower()
        self.assertIn("plages et statuts documentés", low)
        self.assertIn("au-dessus de la référence", low)
        self.assertIn("en dessous de la référence", low)
        self.assertIn("dans la référence", low)
        self.assertNotIn("écart documenté", low)

    def test_doc_scoped_biological_summary_adds_prudent_context_line_for_needs_context(self) -> None:
        ga = __import__("generate_answer")
        rows = [
            {"analyte": "Créatinine", "technical_status_code": "needs_clinical_context", "value_with_unit": "23 mg/L", "reference_short": "4 - 9"},
            {"analyte": "Acide urique", "technical_status_code": "needs_clinical_context", "value_with_unit": "23 mg/L", "reference_short": "25 - 70"},
            {"analyte": "Bilirubine Directe", "technical_status_code": "above_reference", "value_with_unit": "6 mg/L", "reference_short": "0.00 - 5.00"},
        ]
        rendered = ga._build_doc_scoped_biological_summary_answer(
            rows,
            max_lines=6,
            no_diagnosis=True,
            render_profile="compact_biological_summary",
        )
        low = rendered.lower()
        self.assertIn("lecture prudente", low)
        self.assertIn("créatinine", low)
        self.assertIn("acide urique", low)

    def test_compact_biological_summary_fallback_keeps_values_and_no_within_sentence(self) -> None:
        ga = __import__("generate_answer")
        rows = [
            {"analyte": "Bilirubine Directe", "technical_status_code": "above_reference", "value_with_unit": "6 mg/L", "reference_short": "0.00 - 5.00"},
            {"analyte": "LDH", "technical_status_code": "above_reference", "value_with_unit": "250 UI/L", "reference_short": "125 - 243"},
            {"analyte": "CK-MB", "technical_status_code": "above_reference", "value_with_unit": "40 UI/L", "reference_short": "< 25"},
            {"analyte": "APO A1", "technical_status_code": "above_reference", "value_with_unit": "2.3 g/L", "reference_short": "1.1 - 1.6"},
            {"analyte": "Ammonium", "technical_status_code": "below_reference", "value_with_unit": "20 µg/dL", "reference_short": "35 - 80"},
        ]
        rendered = ga._build_doc_scoped_biological_summary_answer(
            rows,
            max_lines=6,
            no_diagnosis=True,
            render_profile="compact_biological_summary",
        )
        low = rendered.lower()
        self.assertIn("bilirubine directe = 6 mg/l", low)
        self.assertIn("ldh = 250 ui/l", low)
        self.assertIn("ammonium = 20 µg/dl", low)
        self.assertIn("aucun résultat dans la référence n’est mis en avant", low)

    def test_compact_biological_summary_uses_value_numeric_when_raw_missing(self) -> None:
        ga = __import__("generate_answer")
        rows = [
            {"analyte": "Bilirubine Directe", "technical_status_code": "above_reference", "value_numeric": 6.0, "unit": "mg/L", "reference_short": "0.00 - 5.00"},
            {"analyte": "LDH", "technical_status_code": "above_reference", "value_numeric": 250.0, "unit": "UI/L", "reference_short": "125 - 243"},
            {"analyte": "Ammonium", "technical_status_code": "below_reference", "value_numeric": 20.0, "unit": "µg/dL", "reference_short": "35 - 80"},
        ]
        rendered = ga._build_doc_scoped_biological_summary_answer(
            rows,
            max_lines=6,
            no_diagnosis=True,
            render_profile="compact_biological_summary",
        )
        low = rendered.lower()
        self.assertIn("bilirubine directe =", low)
        self.assertNotIn("non disponible", low)

    def test_final_summary_quality_gate_rejects_non_disponible_when_structured_values_exist(self) -> None:
        ga = __import__("generate_answer")
        rows = [
            {"analyte": "Bilirubine Directe", "technical_status_code": "above_reference", "value_numeric": 6.0, "unit": "mg/L"},
            {"analyte": "LDH", "technical_status_code": "above_reference", "value_numeric": 250.0, "unit": "UI/L"},
            {"analyte": "CKMB (CPKMB)", "technical_status_code": "above_reference", "value_numeric": 40.0, "unit": "UI/L"},
            {"analyte": "APO A1", "technical_status_code": "above_reference", "value_numeric": 2.3, "unit": "g/L"},
            {"analyte": "AMMONIUM", "technical_status_code": "below_reference", "value_numeric": 20.0, "unit": "µg/dL"},
        ]
        gate = ga._evaluate_summary_quality_gate(
            answer=(
                "Points biologiques notables : Bilirubine Directe = non disponible, au-dessus de la référence, "
                "LDH = non disponible, au-dessus de la référence, CKMB (CPKMB) = non disponible, au-dessus de la référence."
            ),
            selected_route="doc_scoped_biological_summary",
            displayed_evidences=rows,
        )
        self.assertFalse(bool(gate.get("pass")))
        self.assertIn("missing_values_in_final_answer_despite_structured_values", list(gate.get("reasons") or []))

    def test_general_conversation_bonjour_fast_path_no_retrieval(self) -> None:
        result = run_generation(
            query="Bonjour.",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "general_conversation")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_general_conversation")
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})

    def test_diagnostic_safety_messages_are_config_driven(self) -> None:
        ga = __import__("generate_answer")
        cfg = {
            "diagnostic_safety": {
                "generic": {
                    "cancer_refusal": "CFG_REFUS",
                    "markers_intro": "CFG_INTRO",
                    "closing": "CFG_CLOSE",
                }
            }
        }
        pack = {
            "intent": "diagnostic_safety_question",
            "question": "Peut-on conclure à un cancer ?",
            "evidences": [],
            "requested_doc_ids": ["report_1"],
            "requested_analytes": [],
        }
        with mock.patch("generate_answer.get_assistant_messages_config", return_value=cfg):
            answer = ga.render_evidence_pack_deterministic(pack, "paragraph")
        self.assertIn("CFG_REFUS", answer)
        self.assertIn("CFG_INTRO", answer)
        self.assertIn("CFG_CLOSE", answer)

    def test_clarification_messages_are_config_driven(self) -> None:
        cfg = {
            "clarifications": {
                "abnormal_without_scope": "CFG_ABNORMAL_SCOPE",
                "abnormal_without_scope_conclusion": "CFG_ABNORMAL_CONCLUSION",
                "global_summary_no_scope": "CFG_GLOBAL_SCOPE",
            }
        }
        with mock.patch("generate_answer.get_assistant_messages_config", return_value=cfg):
            result = run_generation(
                query="les résultats anormaux",
                mode="keyword",
                top_k=20,
                index_dir="data/indexes",
            )
            answer = str(result.get("answer") or "")
            self.assertIn("CFG_ABNORMAL_SCOPE", answer)
            self.assertIn("CFG_ABNORMAL_CONCLUSION", answer)
            ga = __import__("generate_answer")
            empty_global = ga._render_global_biological_summary_answer([])
            self.assertIn("CFG_GLOBAL_SCOPE", empty_global)

    def test_guarded_thyroid_postprocess_patterns_are_config_driven(self) -> None:
        ga = __import__("generate_answer")
        cfg = {
            "diagnostic_safety": {
                "strong_suggestion_patterns": [r"orient\w*\s+vers\s+une?\s+hyperthyro"],
                "forbidden_clinical_style_patterns": [r"(?im)^.*EXAM_CHECK.*$"],
                "limitation_sentence": "LIM_CFG",
                "discordance_replacement": "DISCORD_CFG",
            }
        }
        raw = (
            "Ces résultats orientent vers une hyperthyroïdie primaire.\n"
            "EXAM_CHECK : faire plus de tests.\n"
            "Conclusion technique : ancien texte."
        )
        with mock.patch("generate_answer.get_safety_guardrails_config", return_value=cfg):
            out = ga._ensure_guarded_thyroid_conclusion(raw)
        self.assertIn("DISCORD_CFG", out)
        self.assertIn("LIM_CFG", out)
        self.assertNotIn("orientent vers", out.lower())

    def test_validator_strong_suggestion_pattern_is_config_driven(self) -> None:
        av = __import__("answer_validator")
        cfg = {
            "diagnostic_safety": {
                "thyroid_topic_keywords": ["hyperthyro"],
                "strong_suggestion_patterns": [r"\borient\w*\s+vers\s+une?\s+hyperthyro"],
                "explicit_negation_markers": ["ne permet pas de conclure"],
            }
        }
        with mock.patch("answer_validator.get_safety_guardrails_config", return_value=cfg):
            val = av.validate_answer(
                query="Compatible avec une hyperthyroïdie ?",
                answer_text="TSHus élevée ; ces résultats orientent vers une hyperthyroïdie.",
                evidence_pack=[
                    {
                        "analyte": "TSHus",
                        "analyte_norm": "tshus",
                        "current_value": "55",
                        "value_raw": "55",
                        "unit": "mUI/L",
                        "reference_range": "0.3-4.0",
                        "technical_status_code": "above_reference",
                        "technical_status": "au-dessus de la référence",
                    }
                ],
                displayed_evidences=[
                    {
                        "analyte": "TSHus",
                        "analyte_norm": "tshus",
                        "current_value": "55",
                        "value_raw": "55",
                        "unit": "mUI/L",
                        "reference_range": "0.3-4.0",
                        "technical_status_code": "above_reference",
                        "technical_status": "au-dessus de la référence",
                    }
                ],
                generation_mode="hybrid_structured_llm_writer",
                retrieval_status="answerable",
                diagnostic_safety_intent=True,
                query_intents={"diagnostic_safety_question": True},
            )
        self.assertIn("diagnostic_suggestion_too_strong", list(val.get("errors") or []))

    def test_general_conversation_identity_fast_path_no_retrieval(self) -> None:
        result = run_generation(
            query="t'es qui ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "general_conversation")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_general_conversation")
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})
        self.assertEqual(len(result.get("sources") or []), 0)
        answer = str(result.get("answer") or "").lower()
        self.assertIn("assistant rag médical", answer)
        self.assertNotIn("tshus", answer)
        self.assertNotIn("report_", answer)

    def test_general_conversation_capability_fast_path_no_retrieval(self) -> None:
        result = run_generation(
            query="qu'est-ce que tu peux faire ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "general_conversation")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_general_conversation")
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})
        self.assertEqual(len(result.get("sources") or []), 0)
        self.assertEqual(len(result.get("displayed_evidences") or []), 0)
        answer = str(result.get("answer") or "").lower()
        self.assertIn("vous pouvez me demander", answer)

    def test_general_conversation_thanks_fast_path_no_retrieval(self) -> None:
        result = run_generation(
            query="Merci",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "general_conversation")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_general_conversation")
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})
        self.assertEqual(len(result.get("sources") or []), 0)
        self.assertEqual(len(result.get("displayed_evidences") or []), 0)

    def test_mixed_greeting_medical_ask_stays_medical(self) -> None:
        result = run_generation(
            query="Bonjour, peux-tu résumer le report 16 ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertNotEqual(str((result.get("debug") or {}).get("selected_route") or ""), "general_conversation")
        self.assertNotEqual(str(result.get("generation_mode") or ""), "deterministic_general_conversation")
        self.assertTrue(bool(result.get("sources")))
        self.assertTrue(bool(result.get("displayed_evidences")))

    def test_report12_exposes_structured_counter_metrics(self) -> None:
        result = run_generation(
            query="Fais une synthèse biologique courte du report 12. Limite-toi à 3 à 5 lignes, mentionne uniquement les anomalies majeures, les résultats dans la référence et une conclusion prudente, sans diagnostic.",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertGreater(int(result.get("displayed_evidences_count") or 0), 0)
        self.assertGreater(int(result.get("evidence_pack_count") or 0), 0)
        self.assertGreater(int(result.get("lab_result_count") or 0), 0)
        self.assertGreater(int(result.get("value_numeric_count") or 0), 0)
        self.assertGreater(int(result.get("structured_values_count") or 0), 0)
        self.assertGreater(int(result.get("sources_count") or 0), 0)
        debug = dict(result.get("debug") or {})
        self.assertGreater(int(debug.get("displayed_evidences_count") or 0), 0)
        self.assertGreater(int(debug.get("evidence_pack_count") or 0), 0)
        self.assertGreater(int(debug.get("lab_result_count") or 0), 0)
        self.assertGreater(int(debug.get("value_numeric_count") or 0), 0)
        self.assertGreater(int(debug.get("structured_values_count") or 0), 0)
        self.assertGreater(int(debug.get("sources_count") or 0), 0)
        self.assertGreater(int(result.get("above_reference_count") or 0), 0)
        self.assertGreater(int(result.get("below_reference_count") or 0), 0)
        self.assertGreater(int(result.get("major_anomalies_count") or 0), 0)
        self.assertGreaterEqual(int(result.get("selected_normal_results_count") or 0), 0)
        self.assertGreater(int(debug.get("above_reference_count") or 0), 0)
        self.assertGreater(int(debug.get("below_reference_count") or 0), 0)
        self.assertGreater(int(debug.get("major_anomalies_count") or 0), 0)
        self.assertGreaterEqual(int(debug.get("selected_normal_results_count") or 0), 0)
        self.assertIn("llm_quality_gate", result)
        self.assertIn("final_answer_quality_gate", result)
        self.assertIn("llm_quality_gate", debug)
        self.assertIn("final_answer_quality_gate", debug)
        self.assertIn("quality_final_status", result)
        self.assertIn("synthesis_quality_reason", result)
        self.assertIn("selected_major_anomalies_for_fallback", result)
        self.assertIn("selected_major_anomalies_for_fallback", debug)

    def test_single_report_request_filters_out_other_docs(self) -> None:
        keep = _mk_result(
            chunk_id="chk_report_19_insuline",
            doc_id="report_19",
            analyte="INSULINE",
            analyte_norm="insuline",
            value_raw="4,90",
        )
        leak = _mk_result(
            chunk_id="chk_report_28_insuline",
            doc_id="report_28",
            analyte="INSULINE",
            analyte_norm="insuline",
            value_raw="1,00",
        )
        response = SearchResponse(
            query="x",
            mode="keyword",
            filters={},
            top_results=[keep, leak],
            context_chunks=[keep, leak],
            sources=[
                {"doc_id": "report_19", "chunk_id": "chk_report_19_insuline"},
                {"doc_id": "report_28", "chunk_id": "chk_report_28_insuline"},
            ],
            answerability={"status": "answerable", "reason": "test"},
        )

        result = run_generation(
            query="Dans report 19, compare l’insuline et la T4 libre avec leurs résultats antérieurs.",
            mode="keyword",
            top_k=5,
            index_dir="data/indexes",
            search_engine=_FakeSearchEngine(response),
        )

        docs = {str(ev.get("doc_id") or "") for ev in (result.get("displayed_evidences") or [])}
        self.assertEqual(docs, {"report_19"})
        validation = result.get("validation") or {}
        self.assertFalse(bool(validation.get("requested_doc_id_mismatch")))

    def test_multi_report_request_keeps_only_requested_docs(self) -> None:
        r11 = _mk_result(
            chunk_id="chk_report_11_crp",
            doc_id="report_11",
            analyte="CRP",
            analyte_norm="crp",
            value_raw="10,00",
        )
        r12 = _mk_result(
            chunk_id="chk_report_12_crp",
            doc_id="report_12",
            analyte="CRP",
            analyte_norm="crp",
            value_raw="20,00",
        )
        r31 = _mk_result(
            chunk_id="chk_report_31_crp",
            doc_id="report_31",
            analyte="CRP",
            analyte_norm="crp",
            value_raw="30,00",
        )
        response = SearchResponse(
            query="x",
            mode="keyword",
            filters={},
            top_results=[r11, r12, r31],
            context_chunks=[r11, r12, r31],
            sources=[
                {"doc_id": "report_11", "chunk_id": "chk_report_11_crp"},
                {"doc_id": "report_12", "chunk_id": "chk_report_12_crp"},
                {"doc_id": "report_31", "chunk_id": "chk_report_31_crp"},
            ],
            answerability={"status": "answerable", "reason": "test"},
        )

        result = run_generation(
            query="Compare report 12 et report 11 sur CRP.",
            mode="keyword",
            top_k=5,
            index_dir="data/indexes",
            search_engine=_FakeSearchEngine(response),
        )

        docs = {str(ev.get("doc_id") or "") for ev in (result.get("displayed_evidences") or [])}
        self.assertTrue(docs.issubset({"report_11", "report_12"}))
        self.assertNotIn("report_31", docs)

    def test_response_transform_uses_previous_evidence_without_retrieval(self) -> None:
        first = run_generation(
            query="Dans report 19, compare l’insuline et la T4 libre avec leurs résultats antérieurs. Retourne la réponse sous forme de tableau.",
            mode="keyword",
            top_k=5,
            index_dir="data/indexes",
        )
        previous_pack = first.get("structured_evidence_pack") or {}
        self.assertTrue(previous_pack.get("evidences"))

        transformed = run_generation(
            query="Convertis la réponse précédente en JSON strict.",
            mode="keyword",
            top_k=5,
            index_dir="data/indexes",
            search_engine=_FailIfCalledSearchEngine(),
            previous_structured_evidence_pack=previous_pack,
        )
        answer = str(transformed.get("answer") or "")
        self.assertTrue(answer.startswith("{") and answer.endswith("}"))
        self.assertIn("report_19", answer)

    def test_response_transform_paragraph_clears_inherited_table_columns(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding("Convertis la réponse précédente en style paragraphe médical pro.")
        previous_pack = {
            "requested_table_columns": ["analyte", "valeur_actuelle", "reference", "source"],
            "evidences": [
                {
                    "doc_id": "report_24",
                    "analyte": "CRP",
                    "current_value": "7",
                    "unit": "mg/l",
                    "reference": "0 - 5",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                }
            ],
        }
        transformed = ga._build_response_transform_pack(
            query="Convertis la réponse précédente en style paragraphe médical pro.",
            query_understanding=qu,
            previous_pack=previous_pack,
        )
        self.assertEqual(str(transformed.get("output_format") or ""), "paragraph")
        self.assertEqual(list(transformed.get("requested_table_columns") or []), [])

    def test_doc_summary_anomalies_only_filters_within_reference(self) -> None:
        result = run_generation(
            query="Résume uniquement les anomalies biologiques du report (16), sans poser de diagnostic.",
            mode="keyword",
            top_k=5,
            index_dir="data/indexes",
        )
        structured = dict(result.get("structured_evidence_pack") or {})
        evidences = list(structured.get("evidences") or [])
        self.assertTrue(evidences)
        statuses = {
            str(ev.get("technical_status_code") or ev.get("interpretation_status") or "").strip().lower()
            for ev in evidences
        }
        self.assertTrue(statuses.issubset({"above_reference", "below_reference"}))
        self.assertNotIn("within_reference", statuses)
        self.assertIn(
            str(result.get("generation_mode") or ""),
            {
                "hybrid_structured_llm_writer",
                "deterministic_safety_fallback_after_llm_validation_failure",
                "deterministic_doc_scoped_abnormal_results",
            },
        )
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})
        dbg = dict(result.get("debug") or {})
        self.assertEqual(str(dbg.get("selected_route") or ""), "doc_scoped_abnormal_results")
        self.assertTrue(str(dbg.get("route_reason") or ""))

    def test_global_insuline_out_of_reference_routes_to_global_abnormal_search(self) -> None:
        result = run_generation(
            query="Dans tous les rapports disponibles, quels documents contiennent une insuline hors référence ? Donne le document, la valeur, la référence et le statut.",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertIn(
            str(result.get("generation_mode") or ""),
            {
                "hybrid_structured_llm_writer",
                "deterministic_safety_fallback_after_llm_validation_failure",
                "deterministic_global_analyte_abnormal_search",
            },
        )
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})
        answer = str(result.get("answer") or "").lower()
        self.assertIn("insuline", answer)
        self.assertIn("report_19", answer)
        self.assertIn("report_16", answer)
        self.assertNotIn("plage de référence", answer)
        dbg = dict(result.get("debug") or {})
        self.assertEqual(str(dbg.get("selected_route") or ""), "global_analyte_abnormal_search")

    def test_guarded_hyperthyroid_doc16_uses_thyroid_block(self) -> None:
        result = run_generation(
            query="Est-ce que le report (16) permet de conclure à une hyperthyroïdie ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertIn(
            str(result.get("generation_mode") or ""),
            {
                "hybrid_structured_llm_writer",
                "deterministic_safety_fallback_after_llm_validation_failure",
                "deterministic_guarded_medical_interpretation",
            },
        )
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
        answer = str(result.get("answer") or "").lower()
        self.assertIn("t4 libre", answer)
        self.assertIn("t3 libre", answer)
        self.assertIn("tshus", answer)
        self.assertIn("discordant", answer)
        self.assertNotIn("acth", answer)
        dbg = dict(result.get("debug") or {})
        self.assertEqual(str(dbg.get("selected_route") or ""), "doc_scoped_medical_interpretation_guarded")

    def test_global_vitamin_low_directional_filter(self) -> None:
        try:
            result = run_generation(
                query="Quels rapports montrent une vitamine D ou vitamine B12 basse ? Donne uniquement les résultats réellement présents dans les documents.",
                mode="keyword",
                top_k=50,
                index_dir="data/indexes",
            )
        except RuntimeError as exc:
            self.skipTest(f"retrieval backend indisponible dans cet environnement: {exc}")
        qu = dict(result.get("query_understanding") or {})
        self.assertEqual(str(qu.get("technical_condition") or ""), "below_reference")
        self.assertNotEqual(str(result.get("generation_mode") or ""), "deterministic_reference_range_lookup")
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "global_analyte_abnormal_search")
        evidences = list((result.get("structured_evidence_pack") or {}).get("evidences") or [])
        self.assertTrue(evidences)
        statuses = {
            str(ev.get("technical_status_code") or ev.get("interpretation_status") or "").strip().lower()
            for ev in evidences
        }
        self.assertTrue(statuses.issubset({"below_reference"}))
        for ev in evidences:
            analyte = str(ev.get("analyte") or "").upper()
            value = str(ev.get("current_value") or ev.get("value_raw") or "")
            if "VITAMINE D" in analyte:
                self.assertNotIn("55", value)
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")

    def test_global_crp_above_directional_filter_and_embedded_reference(self) -> None:
        result = run_generation(
            query="Y a-t-il des rapports avec CRP supérieure à la référence ?",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        qu = dict(result.get("query_understanding") or {})
        self.assertEqual(str(qu.get("technical_condition") or ""), "above_reference")
        self.assertNotEqual(str(result.get("generation_mode") or ""), "deterministic_reference_range_lookup")
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "global_analyte_abnormal_search")
        evidences = list((result.get("structured_evidence_pack") or {}).get("evidences") or [])
        self.assertTrue(evidences)
        for ev in evidences:
            status = str(ev.get("technical_status_code") or ev.get("interpretation_status") or "").strip().lower()
            self.assertEqual(status, "above_reference")
            analyte = str(ev.get("analyte") or "")
            self.assertEqual(analyte.strip().upper(), "CRP")
            reference = str(ev.get("reference") or "")
            self.assertNotIn("non disponible", reference.lower())
            doc_id = str(ev.get("doc_id") or "").strip().lower()
            value = str(ev.get("current_value") or ev.get("value_raw") or "").strip()
            if doc_id == "report_12":
                self.assertNotEqual(value, "1.2")
                self.assertNotEqual(value, "1,2")
        displayed = list(result.get("displayed_evidences") or [])
        self.assertEqual(len(displayed), len(evidences))
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
        self.assertEqual(list((result.get("query_understanding") or {}).get("requested_doc_ids") or []), [])

    def test_global_crp_does_not_inherit_previous_doc_scope(self) -> None:
        result = run_generation(
            query="Y a-t-il une élévation de la CRP dans les rapports disponibles ? Donne les documents et les valeurs.",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
            previous_doc_scope=["report_31", "report_9"],
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "global_analyte_abnormal_search")
        qu = dict(result.get("query_understanding") or {})
        self.assertEqual(list(qu.get("requested_doc_ids") or []), [])
        self.assertEqual(str(qu.get("technical_condition") or ""), "above_reference")
        self.assertNotEqual(str(result.get("generation_mode") or ""), "deterministic_reference_range_lookup")

    def test_toxicology_urine_no_uric_crystal_confusion(self) -> None:
        result = run_generation(
            query="Quels rapports comportent une recherche de toxiques urinaires, et quelles familles sont testées ?",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertIn(
            str((result.get("debug") or {}).get("selected_route") or ""),
            {"global_toxicology_search", "global_qualitative_toxicology_search", "open_grounded_medical_question"},
        )
        self.assertNotEqual(str(result.get("generation_mode") or "").strip().lower(), "llm")
        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("cristaux d'acide urique", answer)
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})

    def test_report12_biological_summary_route(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "0"
        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": (
                    "Anormaux : CRP = 7 mg/l (réf 0.00 - 5.00, au-dessus); Réserve Alcaline = 20 mmol/l (réf 21,00 - 28,00, en dessous).\n"
                    "Résultats dans la référence uniquement : ACIDE URIQUE; Calcium; Chlore; Phosphore; Protéines totales; Potassium SANGUIN; Sodium SANGUIN; LDH.\n"
                    "Conclusion technique : synthèse descriptive limitée aux données disponibles."
                ),
                "llm_error": None,
                "prompt_chars": 1800,
                "llm_prompt_tokens_estimate": 450,
                "use_micro_prompt": True,
                "llm_call_skipped_due_prompt_budget": False,
                "llm_prompt_first_500": "x",
                "llm_prompt_last_500": "y",
            }
        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro):
                result = run_generation(
                    query="Résume le report 12 en quelques lignes, avec une partie anomalies et une partie résultats normaux, sans conclusion diagnostique.",
                    mode="keyword",
                    top_k=50,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "doc_scoped_biological_summary")
        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("policy_level") or ""), "hybrid_controlled")
        self.assertEqual(str(debug.get("generation_strategy") or ""), "llm_writer_expected")
        self.assertIn(debug.get("llm_expected"), {True, 1})
        self.assertIn(debug.get("llm_writer_attempted"), {True, 1})
        self.assertIn(
            str(result.get("generation_mode") or ""),
            {"hybrid_structured_llm_writer", "deterministic_doc_scoped_biological_summary"},
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("anorm", answer)
        self.assertIn("normaux", str(result.get("answer") or "").lower())
        self.assertNotIn("anormaux : aucune anomalie objectivée", answer)
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})
        stage = dict(debug.get("stage_timings_ms") or {})
        self.assertGreaterEqual(float(stage.get("llm_writer_ms") or 0.0), 0.0)

    def test_report24_short_note_summary_keeps_abnormal_when_present(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": (
                    "Anormaux : CRP = 7 mg/l (réf 0.00 - 5.00, au-dessus); Réserve Alcaline = 20 mmol/l (réf 21,00 - 28,00, en dessous).\n"
                    "Résultats dans la référence uniquement : ACIDE URIQUE; Calcium; Chlore; Phosphore; Protéines totales; Potassium SANGUIN; Sodium SANGUIN; LDH.\n"
                    "Conclusion technique : synthèse descriptive limitée aux données disponibles."
                ),
                "llm_error": None,
                "prompt_chars": 1800,
                "llm_prompt_tokens_estimate": 450,
                "use_micro_prompt": True,
                "llm_call_skipped_due_prompt_budget": False,
                "llm_prompt_first_500": "x",
                "llm_prompt_last_500": "y",
            }
        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro):
                result = run_generation(
                    query="Résume le report 24 comme une note courte pour un médecin, en restant strictement descriptif et sans diagnostic.",
                    mode="keyword",
                    top_k=50,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "doc_scoped_biological_summary")
        self.assertNotEqual(str((result.get("validation") or {}).get("validation_status") or "").lower(), "fail")
        debug = dict(result.get("debug") or {})
        self.assertGreater(int(debug.get("abnormal_rows_count") or 0), 0)
        self.assertGreater(int(debug.get("llm_abnormal_rows_count") or 0), 0)
        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("anormaux : aucun", answer)
        self.assertTrue(("crp" in answer) or ("reserve alcaline" in answer) or ("réserve alcaline" in answer))
        self.assertGreater(len(list(result.get("displayed_evidences") or [])), 0)
        self.assertGreater(len(list(result.get("sources") or [])), 0)

    def test_biological_summary_false_no_abnormal_claim_triggers_correction(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": (
                    "Anormaux : Aucun fait anormal fourni.\n"
                    "Résultats dans la référence uniquement : ACIDE URIQUE 40 mg/l.\n"
                    "Conclusion technique : synthèse descriptive limitée aux données disponibles, sans diagnostic."
                ),
                "llm_candidate_answer": (
                    "Anormaux : Aucun fait anormal fourni.\n"
                    "Résultats dans la référence uniquement : ACIDE URIQUE 40 mg/l.\n"
                    "Conclusion technique : synthèse descriptive limitée aux données disponibles, sans diagnostic."
                ),
                "llm_error": None,
                "use_micro_prompt": True,
            }

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro):
                result = run_generation(
                    query="Résume le report 24 comme une note courte pour un médecin, en restant strictement descriptif et sans diagnostic.",
                    mode="keyword",
                    top_k=50,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("anormaux : aucun", answer)
        self.assertNotEqual(str((result.get("validation") or {}).get("validation_status") or "").lower(), "fail")
        self.assertIn(str((result.get("generation_mode") or "")), {"hybrid_structured_llm_writer", "deterministic_doc_scoped_biological_summary"})

    def test_biological_summary_zero_anomaly_doc_allows_no_abnormal_phrase(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "0"
        try:
            result = run_generation(
                query="Résume le report 13 comme une note courte, strictement descriptive et sans diagnostic.",
                mode="keyword",
                top_k=50,
                index_dir="data/indexes",
            )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        self.assertIn(str((result.get("validation") or {}).get("validation_status") or ""), {"pass", "warning"})
        answer = str(result.get("answer") or "").lower()
        if "anormaux : aucun" in answer or "aucun fait anormal" in answer:
            self.assertIn("aucun", answer)

    def test_biological_summary_small_llm_budget_prioritizes_abnormal_rows(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        captured: dict[str, object] = {}

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            captured["llm_pack"] = kwargs.get("llm_pack")
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": (
                    "Anormaux : bilirubine directe élevée, LDH élevée.\n"
                    "Résultats dans la référence uniquement : aucun résultat strictement dans la référence parmi les éléments sélectionnés.\n"
                    "Conclusion technique : synthèse descriptive limitée aux données disponibles, sans diagnostic."
                ),
                "llm_candidate_answer": (
                    "Anormaux : bilirubine directe élevée, LDH élevée.\n"
                    "Résultats dans la référence uniquement : aucun résultat strictement dans la référence parmi les éléments sélectionnés.\n"
                    "Conclusion technique : synthèse descriptive limitée aux données disponibles, sans diagnostic."
                ),
                "llm_error": None,
                "use_micro_prompt": True,
            }

        override_policy = {
            "prompt_target_chars": 2500,
            "prompt_hard_limit_chars": 3500,
            "num_predict": 180,
            "timeout_ms": 60000,
            "max_evidence_rows": 4,
            "use_micro_prompt": True,
        }
        try:
            with mock.patch.dict("generate_answer._LEVEL2_LLM_PROMPT_POLICY", {"doc_scoped_biological_summary": override_policy}, clear=False), mock.patch(
                "generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro
            ):
                run_generation(
                    query="Fais une synthèse médico-biologique du report 12 en 6 lignes maximum, en séparant les anomalies et les résultats rassurants. Ne donne pas de diagnostic.",
                    mode="keyword",
                    top_k=50,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        llm_pack = dict(captured.get("llm_pack") or {})
        evidences = list(llm_pack.get("evidences") or [])
        self.assertLessEqual(len(evidences), 4)
        abnormal_count = 0
        for ev in evidences:
            status = str(ev.get("status") or "").lower()
            if any(k in status for k in ["au-dessus", "au dessus", "en dessous", "above_reference", "below_reference"]):
                abnormal_count += 1
        self.assertGreater(abnormal_count, 0)

    def test_report10_hierarchy_trigger_route(self) -> None:
        result = run_generation(
            query="Dans le report 10, hiérarchise les anomalies biologiques selon leur importance technique.",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "doc_scoped_priority_anomalies")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_doc_scoped_priority_anomalies")

    def test_doc_scoped_priority_anomalies_scoring_and_no_llm_call_when_forced_off(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "0"
        try:
            result = run_generation(
                query="Dans report (10), liste les anomalies importantes par ordre de priorité technique.",
                mode="keyword",
                top_k=50,
                index_dir="data/indexes",
            )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "doc_scoped_priority_anomalies")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_doc_scoped_priority_anomalies")
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
        validation_errors = list(result.get("validation", {}).get("errors") or [])
        self.assertNotIn("requested_doc_id_mismatch", validation_errors)
        self.assertNotIn("doc_id_mismatch", validation_errors)
        self.assertNotIn("output_columns_not_respected", validation_errors)
        self.assertNotIn("exact_columns_not_respected", validation_errors)
        displayed = list(result.get("displayed_evidences") or [])
        self.assertTrue(displayed)
        for ev in displayed:
            self.assertIn("priority_score", ev)
            self.assertIn("priority_level", ev)
            self.assertIn("priority_reason", ev)
            self.assertIn(str(ev.get("priority_level")), {"high", "moderate", "low"})
            self.assertIn(str(ev.get("technical_status_code") or ""), {"above_reference", "below_reference"})
        answer = str(result.get("answer") or "").lower()
        self.assertIn("priorité", answer)
        self.assertNotIn("non présent pmol", answer)
        stage = dict((result.get("debug") or {}).get("stage_timings_ms") or {})
        self.assertIn("llm_writer_ms", stage)
        self.assertIn(stage.get("llm_writer_ms"), {0, 0.0})
        srcs = list(result.get("sources") or [])
        self.assertTrue(srcs)
        self.assertTrue(all("ligne" in str(s.get("label") or "").lower() for s in srcs))

    def test_multi_doc_comparison_three_docs_keeps_requested_scope(self) -> None:
        result = run_generation(
            query="Compare les reports 16, 19 et 31 sur l’insuline et le bilan thyroïdien, en précisant les données manquantes.",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertIn(
            str(result.get("generation_mode") or ""),
            {
                "hybrid_structured_llm_writer",
                "deterministic_safety_fallback_after_llm_validation_failure",
                "deterministic_multi_doc_comparison",
            },
        )
        self.assertNotEqual(str(result.get("generation_mode") or ""), "deterministic_reference_range_lookup")
        qu = dict(result.get("query_understanding") or {})
        self.assertEqual(str(qu.get("intent") or ""), "multi_doc_comparison")
        self.assertEqual(list(qu.get("requested_doc_ids") or []), ["report_16", "report_19", "report_31"])
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "multi_doc_comparison")
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})
        errors = list(result.get("validation", {}).get("errors") or [])
        self.assertNotIn("analyte_overmatch", errors)
        self.assertNotIn("false_missing_item", errors)
        answer = str(result.get("answer") or "").lower()
        self.assertIn("report_16", answer)
        self.assertIn("report_19", answer)
        self.assertIn("report_31", answer)
        self.assertNotIn("report_29", answer)

    def test_no_final_llm_fail_for_unstructured_medical_query(self) -> None:
        result = run_generation(
            query="Les rapports disponibles permettent-ils d’affirmer une pathologie endocrinienne active ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertIn(
            str((result.get("debug") or {}).get("selected_route") or ""),
            {"open_grounded_medical_question", "global_qualitative_toxicology_search"},
        )
        self.assertFalse(
            str(result.get("generation_mode") or "").strip().lower() == "llm"
            and str(result.get("validation", {}).get("validation_status") or "").strip().lower() == "fail"
        )
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})

    def test_diagnostic_report16_no_llm_when_forced_off(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "0"
        try:
            result = run_generation(
                query="Quel diagnostic évoques-tu à partir du report 16 ?",
                mode="keyword",
                top_k=20,
                index_dir="data/indexes",
            )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        answer = str(result.get("answer") or "")
        self.assertTrue(answer.startswith("Je ne peux pas poser ni évoquer un diagnostic à partir de ces résultats seuls."))
        stage = dict((result.get("debug") or {}).get("stage_timings_ms") or {})
        self.assertIn(stage.get("llm_writer_ms"), {0, 0.0})
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})

    def test_doc_pair_comparison_missing_values_validation_non_fail(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "0"
        try:
            result = run_generation(
                query="Compare report 19 au report 16 pour l’insuline et les paramètres thyroïdiens.",
                mode="keyword",
                top_k=50,
                index_dir="data/indexes",
            )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("non présent pmol", answer)
        self.assertNotIn("non présent mui", answer)
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})

    def test_doc_not_found_returns_no_evidence_contract(self) -> None:
        result = run_generation(
            query="Peux-tu analyser le report 99 ? Je veux les anomalies biologiques retrouvées.",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        answer = str(result.get("answer") or "").lower()
        self.assertIn("aucun résultat biologique exploitable", answer)
        self.assertIn(str(result.get("generation_mode") or ""), {"deterministic_no_evidence_response", "deterministic_doc_scoped_abnormal_results"})
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})

    def test_reference_range_acide_urique_homme_deterministic(self) -> None:
        result = run_generation(
            query="Quelle est la plage physiologique de l’acide urique chez l’homme ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_reference_range_lookup")
        self.assertIn("acide_urique", list((result.get("query_understanding") or {}).get("requested_analytes") or []))
        answer = str(result.get("answer") or "").lower()
        self.assertIn("35", answer)
        self.assertIn("72", answer)
        self.assertNotIn("chunk_id", answer)
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})
        self.assertLessEqual(len(list(result.get("sources") or [])), 3)
        self.assertNotIn("max_display_results_exceeded_for_simple_query", list((result.get("validation") or {}).get("warnings") or []))

    def test_reference_range_acide_urique_femme_deterministic(self) -> None:
        result = run_generation(
            query="Quelle est la plage normale de l’acide urique chez la femme ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_reference_range_lookup")
        answer = str(result.get("answer") or "").lower()
        self.assertIn("26", answer)
        self.assertIn("60", answer)
        self.assertNotIn("chunk_id", answer)
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})
        self.assertLessEqual(len(list(result.get("sources") or [])), 3)
        self.assertNotIn("max_display_results_exceeded_for_simple_query", list((result.get("validation") or {}).get("warnings") or []))

    def test_doc_scoped_reference_range_query_keeps_reference_range_lookup_route(self) -> None:
        result = run_generation(
            query="Dans le report 12, donne la plage de MAGNESIUM PLASMATIQUE.",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "reference_range_lookup")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_reference_range_lookup")
        answer = str(result.get("answer") or "").lower()
        self.assertIn("magnesium", answer)
        self.assertNotIn("valeur :", answer)

    def test_doc_scoped_single_analyte_status_route(self) -> None:
        result = run_generation(
            query="Dans le report 24, quelle est la valeur d’acide urique et est-elle dans la référence pour une femme ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "doc_scoped_single_analyte_status")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_single_analyte_lookup")
        answer = str(result.get("answer") or "").lower()
        self.assertIn("acide urique", answer)
        self.assertNotIn("insuline", answer)
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})

    def test_global_acide_urique_below_reference(self) -> None:
        result = run_generation(
            query="Dans tous les rapports disponibles, quels documents montrent un acide urique en dessous de la référence ?",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "global_analyte_abnormal_search")
        self.assertEqual(str((result.get("query_understanding") or {}).get("technical_condition") or ""), "below_reference")
        self.assertIn("acide_urique", list((result.get("query_understanding") or {}).get("requested_analytes") or []))
        evs = list((result.get("structured_evidence_pack") or {}).get("evidences") or [])
        self.assertTrue(evs)
        statuses = {str(ev.get("technical_status_code") or "").strip().lower() for ev in evs}
        self.assertTrue(statuses.issubset({"below_reference"}))

    def test_global_toxicology_search_route(self) -> None:
        result = run_generation(
            query="Quels rapports comportent une recherche de toxiques urinaires ?",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "global_toxicology_search")
        self.assertIn(
            str(result.get("generation_mode") or ""),
            {"deterministic_global_toxicology_search", "deterministic_safe_error_response"},
        )
        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("cristaux d'acide urique", answer)
        if str(result.get("generation_mode") or "") == "deterministic_global_toxicology_search":
            self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
        else:
            self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})

    def test_global_urine_toxicology_reports_and_families(self) -> None:
        result = run_generation(
            query="Quels rapports comportent une recherche de toxiques urinaires, et quelles familles sont testées ?",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "global_toxicology_search")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_global_toxicology_search")
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
        self.assertTrue(list(result.get("displayed_evidences") or []))
        self.assertLessEqual(len(list(result.get("displayed_evidences") or [])), 10)
        answer = str(result.get("answer") or "").lower()
        self.assertIn("| document |", answer)
        self.assertIn("| nature |", answer)
        self.assertNotIn("ecbu", answer)
        self.assertNotIn("cristaux d'acide urique", answer)

    def test_global_blood_toxicology_reports_and_parameters(self) -> None:
        result = run_generation(
            query="Quels rapports contiennent une pharmacotoxicologie sanguine, et quels paramètres sont recherchés ?",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "global_toxicology_search")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_global_toxicology_search")
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
        self.assertTrue(list(result.get("displayed_evidences") or []))
        self.assertLessEqual(len(list(result.get("displayed_evidences") or [])), 10)
        answer = str(result.get("answer") or "").lower()
        self.assertIn("| document |", answer)
        self.assertIn("| nature |", answer)
        self.assertNotIn("cristaux", answer)

    def test_global_biological_summary_route(self) -> None:
        result = run_generation(
            query="Sur l’ensemble des rapports disponibles, fais une synthèse courte des anomalies biologiques principales, sans diagnostic.",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "global_biological_summary")
        self.assertNotEqual(str(result.get("generation_mode") or ""), "")
        self.assertGreater(len(list(result.get("displayed_evidences") or [])), 0)
        self.assertGreater(len(list(result.get("sources") or [])), 0)
        self.assertNotEqual(str((result.get("validation") or {}).get("validation_status") or "").lower(), "fail")

    def test_global_priority_anomalies_summary_route(self) -> None:
        result = run_generation(
            query="Parmi les rapports disponibles, quels résultats biologiques méritent le plus d’attention technique ? Résume en style médical professionnel, sans diagnostic.",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "global_priority_anomalies_summary")
        self.assertNotIn("context_summary_render", str(result.get("generation_mode") or ""))
        self.assertGreater(len(list(result.get("displayed_evidences") or [])), 0)
        self.assertGreater(len(list(result.get("sources") or [])), 0)
        self.assertNotEqual(str((result.get("validation") or {}).get("validation_status") or "").lower(), "fail")

    def test_response_transform_no_context_controlled(self) -> None:
        result = run_generation(
            query="Reformule cette réponse en style professionnel.",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertIn(
            str(result.get("generation_mode") or ""),
            {"deterministic_response_transform", "deterministic_context_summary_render", "deterministic_no_evidence_response", "no_evidence"},
        )
        self.assertNotEqual(str((result.get("validation") or {}).get("validation_status") or "").lower(), "fail")

    def test_doc_scoped_single_analyte_hors_reference_question_can_answer_within_reference(self) -> None:
        result = run_generation(
            query="Dans le report 12, la phosphatase alcaline à 40 UI/L est-elle hors référence chez une femme adulte ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "doc_scoped_single_analyte_status")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_single_analyte_lookup")
        self.assertEqual(str((result.get("validation") or {}).get("validation_status") or ""), "pass")
        answer = str(result.get("answer") or "").lower()
        self.assertIn("40 ui/l", answer)
        self.assertIn("dans la référence".lower(), answer)
        warnings = list((result.get("validation") or {}).get("warnings") or [])
        self.assertNotIn("filter_violation_hors_reference", warnings)
        self.assertNotIn("downgraded_non_fact_error:filter_violation_hors_reference", warnings)

    def test_report29_creatinine_single_analyte_status_not_not_found(self) -> None:
        result = run_generation(
            query="Dans le report 29, la créatinine est-elle normale, basse ou élevée ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "doc_scoped_single_analyte_status")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_single_analyte_lookup")
        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("non retrouvé", answer)
        self.assertTrue(("créatinine" in answer) or ("creatinine" in answer))
        self.assertTrue(any(ch.isdigit() for ch in answer))
        self.assertIn("statut technique", answer)
        self.assertGreater(len(list(result.get("sources") or [])), 0)
        self.assertNotEqual(str((result.get("validation") or {}).get("validation_status") or "").lower(), "fail")

    def test_report27_toxicology_threshold_above_only(self) -> None:
        result = run_generation(
            query="Dans le report 27, quels résultats de pharmacotoxicologie urinaire dépassent leur seuil de référence ?",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "doc_scoped_toxicology_threshold_search")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_doc_scoped_toxicology_threshold_search")
        evs = list(result.get("displayed_evidences") or [])
        self.assertTrue(evs)
        for ev in evs:
            self.assertEqual(str(ev.get("technical_status_code") or "").strip().lower(), "above_reference")

    def test_report25_toxicology_majority_summary_not_safe_error(self) -> None:
        result = run_generation(
            query="Dans le report 25, les toxiques urinaires sont-ils majoritairement sous les seuils ? Donne une réponse technique sans diagnostic.",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "doc_scoped_toxicology_summary")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_doc_scoped_toxicology_summary")
        self.assertNotIn("deterministic_safe_error_response", str(result.get("generation_mode") or ""))
        answer = str(result.get("answer") or "").lower()
        self.assertIn("sous seuil", answer)
        self.assertIn("au-dessus", answer)

    def test_report27_toxicology_summary_under_and_above_with_sources(self) -> None:
        result = run_generation(
            query="Résume les résultats de pharmacotoxicologie urinaire du report 27 en distinguant les résultats sous seuil et ceux au-dessus du seuil. Ne donne aucune interprétation clinique.",
            mode="keyword",
            top_k=50,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "doc_scoped_toxicology_summary")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_doc_scoped_toxicology_summary")
        self.assertNotEqual(str((result.get("validation") or {}).get("validation_status") or "").lower(), "fail")
        self.assertGreater(len(list(result.get("displayed_evidences") or [])), 0)
        self.assertGreater(len(list(result.get("sources") or [])), 0)
        answer = str(result.get("answer") or "").lower()
        self.assertIn("sous seuil", answer)
        self.assertIn("au-dessus", answer)

    def test_strict_route_forced_llm_fact_drift_falls_back_to_deterministic(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        old_strict_style = os.environ.get("MEDICAL_RAG_STRICT_STYLE_LLM_ALLOWED")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        os.environ["MEDICAL_RAG_STRICT_STYLE_LLM_ALLOWED"] = "1"

        def _fake_compose(**kwargs: object) -> dict[str, object]:
            mode = str(kwargs.get("mode") or "")
            if mode == "fallback":
                return {
                    "mode": "deterministic_doc_scoped_abnormal_results",
                    "answer": "Réponse déterministe de secours.",
                    "llm_error": None,
                }
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": "INSULINE 999 uU/mL (faux).",
                "llm_error": None,
                "llm_candidate_answer": "INSULINE 999 uU/mL (faux).",
            }

        try:
            with mock.patch("generate_answer.compose_professional_answer", side_effect=_fake_compose):
                result = run_generation(
                    query="Résume uniquement les anomalies biologiques du report (16), sans poser de diagnostic.",
                    mode="keyword",
                    top_k=20,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
            if old_strict_style is None:
                os.environ.pop("MEDICAL_RAG_STRICT_STYLE_LLM_ALLOWED", None)
            else:
                os.environ["MEDICAL_RAG_STRICT_STYLE_LLM_ALLOWED"] = old_strict_style

        self.assertTrue(str(result.get("generation_mode") or "").startswith("deterministic_"))
        self.assertNotIn("999", str(result.get("answer") or ""))
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("selected_policy") or ""), "deterministic_strict")
        self.assertEqual(str(debug.get("facts_source") or ""), "evidence_rows_only")
        self.assertTrue("validation_errors" in debug or "validation" in debug)

    def test_level1_routes_never_use_llm_even_when_force_enabled(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        try:
            result = run_generation(
                query="Dans tous les rapports disponibles, quels documents contiennent une insuline hors référence ?",
                mode="keyword",
                top_k=30,
                index_dir="data/indexes",
            )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("selected_policy") or ""), "deterministic_strict")
        self.assertIn(debug.get("llm_writer_used"), {False, 0})
        stage = dict(debug.get("stage_timings_ms") or {})
        self.assertIn(stage.get("llm_writer_ms"), {0, 0.0})

    def test_level2_biological_summary_uses_hybrid_policy_limits(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"

        captured: dict[str, object] = {}

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            captured["selected_route"] = kwargs.get("selected_route")
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": "Anormaux : test. Normaux / rassurants : test.",
                "llm_error": None,
                "prompt_chars": 1800,
                "llm_prompt_tokens_estimate": 450,
                "use_micro_prompt": True,
                "llm_call_skipped_due_prompt_budget": False,
                "llm_prompt_first_500": "x",
                "llm_prompt_last_500": "y",
            }

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro):
                result = run_generation(
                    query="Résume le report 12 en quelques lignes, avec une partie anomalies et une partie résultats normaux, sans conclusion diagnostique.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        self.assertEqual(captured.get("selected_route"), "doc_scoped_biological_summary")
        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("policy_level") or ""), "hybrid_controlled")
        self.assertEqual(str(debug.get("generation_strategy") or ""), "llm_writer_expected")
        self.assertIn(debug.get("llm_expected"), {True, 1})
        self.assertGreater(int(debug.get("llm_prompt_tokens_estimate") or 0), 0)
        self.assertLessEqual(int(debug.get("llm_evidence_rows_count") or 0), 6)
        self.assertTrue(bool(debug.get("use_micro_prompt")))
        self.assertLessEqual(int(debug.get("prompt_hard_limit_chars") or 0), 3500)

    def test_biological_summary_uses_llm_even_without_summary_feature_flag(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        if "MEDICAL_RAG_FORCE_LLM_WRITER" in os.environ:
            os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)

        def _feature_enabled(name: str, default: bool = False) -> bool:
            if str(name) == "LLM_SUMMARY_WRITER_ENABLED":
                return False
            if str(name) in {"LLM_GLOBAL_ENABLED", "LLM_REWRITE_ENABLED", "LLM_FALLBACK_NON_CRITICAL_ONLY"}:
                return True
            return default
        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": (
                    "Anormaux : CRP = 7 mg/l (réf 0.00 - 5.00, au-dessus); Réserve Alcaline = 20 mmol/l (réf 21,00 - 28,00, en dessous).\n"
                    "Résultats dans la référence uniquement : ACIDE URIQUE; Calcium; Chlore; Phosphore; Protéines totales; Potassium SANGUIN; Sodium SANGUIN; LDH.\n"
                    "Conclusion technique : synthèse descriptive limitée aux données disponibles."
                ),
                "llm_error": None,
                "prompt_chars": 1800,
                "llm_prompt_tokens_estimate": 450,
                "use_micro_prompt": True,
                "llm_call_skipped_due_prompt_budget": False,
                "llm_prompt_first_500": "x",
                "llm_prompt_last_500": "y",
            }
        try:
            with mock.patch("generate_answer._is_feature_enabled", side_effect=_feature_enabled), mock.patch(
                "generate_answer._compose_level2_micro_prompt_answer",
                side_effect=_fake_micro,
            ):
                result = run_generation(
                    query="Fais une synthèse médico-biologique du report 12 en 6 lignes maximum, en séparant les anomalies et les résultats rassurants. Ne donne pas de diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is not None:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("selected_route") or ""), "doc_scoped_biological_summary")
        self.assertEqual(str(debug.get("policy_level") or ""), "hybrid_controlled")
        self.assertEqual(str(debug.get("generation_strategy") or ""), "llm_writer_expected")
        self.assertIn(debug.get("llm_expected"), {True, 1})
        self.assertIn(debug.get("llm_writer_attempted"), {True, 1})
        self.assertNotEqual(str(debug.get("llm_skipped_reason") or ""), "biological_summary_deterministic_preferred")
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
        self.assertIn(
            str(result.get("generation_mode") or ""),
            {"hybrid_structured_llm_writer", "deterministic_doc_scoped_biological_summary"},
        )

    def test_note_medecin_doc_scoped_uses_numeric_summary_context(self) -> None:
        result = run_generation(
            query="Fais une note médecin courte pour report 12, sans diagnostic.",
            mode="keyword",
            top_k=30,
            index_dir="data/indexes",
        )
        self.assertNotEqual(str(result.get("generation_mode") or ""), "deterministic_no_evidence_response")
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "doc_scoped_biological_summary")
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")

    def test_summary_writer_opt_in_can_use_llm_when_feature_flag_enabled(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        if "MEDICAL_RAG_FORCE_LLM_WRITER" in os.environ:
            os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)

        def _feature_enabled(name: str, default: bool = False) -> bool:
            if str(name) == "LLM_SUMMARY_WRITER_ENABLED":
                return True
            if str(name) in {"LLM_GLOBAL_ENABLED", "LLM_REWRITE_ENABLED", "LLM_FALLBACK_NON_CRITICAL_ONLY"}:
                return True
            return default

        def _fake_micro(**_kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": (
                    "Anormaux : Réserve Alcaline (en dessous).\n"
                    "Résultats dans la référence uniquement : Phosphore, LDH.\n"
                    "Conclusion technique : synthèse descriptive limitée aux données disponibles."
                ),
                "llm_error": None,
            }

        try:
            with mock.patch("generate_answer._is_feature_enabled", side_effect=_feature_enabled):
                with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro):
                    result = run_generation(
                        query="Résume report 24 en 5 lignes max, strictement technique.",
                        mode="keyword",
                        top_k=30,
                        index_dir="data/indexes",
                    )
        finally:
            if old_force is not None:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
            else:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)

        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("selected_route") or ""), "doc_scoped_biological_summary")
        self.assertIn(debug.get("llm_writer_attempted"), {True, 1})
        self.assertEqual(str(debug.get("generation_strategy") or ""), "llm_writer_expected")
        self.assertEqual(str(debug.get("llm_route_class") or ""), "llm_allowed")
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")

    def test_biological_summary_contract_renderer_prevents_abnormal_in_within_section(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {
                "doc_id": "report_24",
                "page": 1,
                "row": 1,
                "analyte": "CRP",
                "analyte_label": "CRP",
                "display_name": "CRP",
                "source_analyte": "CRP",
                "interpretation_status": "needs_clinical_context",
                "current_value": "7",
                "reference": "0 - 5 mg/l",
            },
            {
                "doc_id": "report_24",
                "page": 1,
                "row": 2,
                "analyte": "Phosphore",
                "analyte_label": "Phosphore",
                "display_name": "Phosphore",
                "source_analyte": "Phosphore",
                "interpretation_status": "needs_clinical_context",
                "current_value": "30",
                "reference": "23 - 47 mg/l",
            },
        ]
        llm_bad_json = (
            '{"anormaux":[],"within_reference":["CRP","Phosphore"],'
            '"conclusion":"Conclusion technique : synthèse descriptive."}'
        )
        rendered = ga._render_biological_summary_from_contract(
            llm_answer=llm_bad_json,
            evidences=evidences,
            max_lines=5,
            no_diagnosis=True,
        )
        normalized = rendered.lower()
        self.assertIn("anormaux : crp", normalized)
        self.assertIn("résultats dans la référence uniquement : phosphore", normalized)
        within_line = next((ln for ln in rendered.splitlines() if ln.lower().startswith("résultats dans la référence uniquement")), "")
        self.assertNotIn("crp", within_line.lower())

    def test_biological_summary_contract_renderer_supports_status_field_only(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {
                "doc_id": "report_24",
                "page": 1,
                "row": 1,
                "analyte": "CRP",
                "analyte_label": "CRP",
                "display_name": "CRP",
                "source_analyte": "CRP",
                "status": "au-dessus de la référence",
            },
            {
                "doc_id": "report_24",
                "page": 1,
                "row": 2,
                "analyte": "Phosphore",
                "analyte_label": "Phosphore",
                "display_name": "Phosphore",
                "source_analyte": "Phosphore",
                "status": "dans la référence",
            },
        ]
        llm_bad_json = (
            '{"anormaux":[],"within_reference":["CRP","Phosphore"],'
            '"conclusion":"Conclusion technique : synthèse descriptive."}'
        )
        rendered = ga._render_biological_summary_from_contract(
            llm_answer=llm_bad_json,
            evidences=evidences,
            max_lines=5,
            no_diagnosis=True,
        )
        normalized = rendered.lower()
        self.assertIn("anormaux : crp", normalized)
        self.assertIn("résultats dans la référence uniquement : phosphore", normalized)
        within_line = next((ln for ln in rendered.splitlines() if ln.lower().startswith("résultats dans la référence uniquement")), "")
        self.assertNotIn("crp", within_line.lower())

    def test_biological_summary_contract_renderer_preserves_all_abnormal_rows(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {
                "doc_id": "report_24",
                "page": 1,
                "row": 1,
                "analyte": "CRP",
                "analyte_label": "CRP",
                "display_name": "CRP",
                "source_analyte": "CRP",
                "technical_status_code": "above_reference",
            },
            {
                "doc_id": "report_24",
                "page": 1,
                "row": 2,
                "analyte": "Réserve Alcaline",
                "analyte_label": "Réserve Alcaline",
                "display_name": "Réserve Alcaline",
                "source_analyte": "Réserve Alcaline",
                "technical_status_code": "below_reference",
            },
            {
                "doc_id": "report_24",
                "page": 1,
                "row": 3,
                "analyte": "Phosphore",
                "analyte_label": "Phosphore",
                "display_name": "Phosphore",
                "source_analyte": "Phosphore",
                "technical_status_code": "within_reference",
            },
        ]
        llm_partial_json = (
            '{"anormaux":["CRP"],"within_reference":["Phosphore"],'
            '"conclusion":"Conclusion technique : synthèse descriptive."}'
        )
        rendered = ga._render_biological_summary_from_contract(
            llm_answer=llm_partial_json,
            evidences=evidences,
            max_lines=5,
            no_diagnosis=True,
        )
        normalized = rendered.lower()
        self.assertIn("crp", normalized)
        self.assertIn("réserve alcaline", normalized)

    def test_biological_summary_contract_renderer_compact_rows_do_not_collapse(self) -> None:
        ga = __import__("generate_answer")
        # Simulate compact LLM rows (no doc_id/page/row).
        evidences = [
            {
                "analyte": "CRP",
                "status": "au-dessus de la référence",
                "value_with_unit": "7 mg/l",
                "reference_short": "0-5 mg/l",
                "source_label": "report (24).pdf — page 1, ligne 9",
            },
            {
                "analyte": "Réserve Alcaline",
                "status": "en dessous de la référence",
                "value_with_unit": "20 mmol/l",
                "reference_short": "21-28 mmol/l",
                "source_label": "report (24).pdf — page 1, ligne 10",
            },
            {
                "analyte": "Phosphore",
                "status": "dans la référence",
                "value_with_unit": "30 mg/l",
                "reference_short": "23-47 mg/l",
                "source_label": "report (24).pdf — page 1, ligne 4",
            },
        ]
        llm_json = (
            '{"anormaux":["CRP"],"within_reference":["Phosphore"],'
            '"conclusion":"Conclusion technique : synthèse descriptive."}'
        )
        rendered = ga._render_biological_summary_from_contract(
            llm_answer=llm_json,
            evidences=evidences,
            max_lines=5,
            no_diagnosis=True,
        )
        normalized = rendered.lower()
        self.assertIn("crp", normalized)
        self.assertIn("réserve alcaline", normalized)
        self.assertIn("phosphore", normalized)

    def test_biological_summary_contract_renderer_preserves_safe_llm_narrative(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {
                "doc_id": "report_24",
                "page": 1,
                "row": 1,
                "analyte": "CRP",
                "analyte_label": "CRP",
                "display_name": "CRP",
                "source_analyte": "CRP",
                "interpretation_status": "needs_clinical_context",
                "current_value": "7",
                "reference": "0 - 5 mg/l",
            },
            {
                "doc_id": "report_24",
                "page": 1,
                "row": 2,
                "analyte": "Phosphore",
                "analyte_label": "Phosphore",
                "display_name": "Phosphore",
                "source_analyte": "Phosphore",
                "interpretation_status": "needs_clinical_context",
                "current_value": "30",
                "reference": "23 - 47 mg/l",
            },
        ]
        llm_narrative = (
            "Le bilan met en avant une CRP au-dessus de la référence parmi les éléments sélectionnés.\n"
            "Le phosphore reste dans la référence et apporte un point de stabilité descriptif.\n"
            "La lecture reste prudente et strictement limitée aux données du rapport."
        )
        rendered = ga._render_biological_summary_from_contract(
            llm_answer=llm_narrative,
            evidences=evidences,
            max_lines=5,
            no_diagnosis=True,
        )
        self.assertIn("Le bilan met en avant une CRP", rendered)
        self.assertIn("phosphore reste dans la référence", rendered)
        self.assertIn("Conclusion technique", rendered)
        self.assertNotIn("Anormaux :", rendered)

    def test_biological_summary_template_includes_value_and_reference(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {
                "analyte": "CRP",
                "current_value": "7",
                "unit": "mg/l",
                "reference": "0 - 5 mg/l",
                "technical_status_code": "above_reference",
            },
            {
                "analyte": "Réserve Alcaline",
                "current_value": "20",
                "unit": "mmol/l",
                "reference": "21 - 28 mmol/l",
                "technical_status_code": "below_reference",
            },
            {
                "analyte": "Phosphore",
                "current_value": "30",
                "unit": "mg/l",
                "reference": "23 - 47 mg/l",
                "technical_status_code": "within_reference",
            },
        ]
        rendered = ga._build_doc_scoped_biological_summary_answer(
            evidences,
            max_lines=5,
            no_diagnosis=True,
        )
        self.assertIn("CRP = 7 mg/l", rendered)
        self.assertIn("réf 0 - 5", rendered)
        self.assertIn("Réserve Alcaline = 20 mmol/l", rendered)
        self.assertIn("réf 21 - 28", rendered)
        self.assertIn("Phosphore = 30 mg/l", rendered)

    def test_biological_summary_template_avoids_placeholder_reference_duplication(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {
                "analyte": "Cholestérol total",
                "current_value": "1.60",
                "unit": "g/l",
                "reference": "",
                "technical_status_code": "below_reference",
            }
        ]
        rendered = ga._build_doc_scoped_biological_summary_answer(
            evidences,
            max_lines=5,
            no_diagnosis=True,
        )
        low = rendered.lower()
        self.assertIn("cholestérol total = 1.60 g/l (en dessous)", low)
        self.assertNotIn("réf réf. disponible", low)
        self.assertNotIn("(réf non disponible", low)

    def test_biological_summary_template_lists_normal_names_only_when_many(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {
                "doc_id": "report_24",
                "analyte": "CRP",
                "current_value": "7",
                "unit": "mg/l",
                "reference": "0 - 5 mg/l",
                "technical_status_code": "above_reference",
            },
            {
                "doc_id": "report_24",
                "analyte": "Réserve Alcaline",
                "current_value": "20",
                "unit": "mmol/l",
                "reference": "21 - 28 mmol/l",
                "technical_status_code": "below_reference",
            },
        ]
        evidences.extend(
            [
                {
                    "doc_id": "report_24",
                    "analyte": f"N{i}",
                    "current_value": "1",
                    "unit": "u",
                    "reference": "0 - 2",
                    "technical_status_code": "within_reference",
                }
                for i in range(1, 7)
            ]
        )
        rendered = ga._build_doc_scoped_biological_summary_answer(
            evidences,
            max_lines=5,
            no_diagnosis=True,
        )
        self.assertIn("Résultats dans la référence uniquement : N1", rendered)
        self.assertNotIn("N1 = 1 u", rendered)
        self.assertNotIn("+1 autre(s)", rendered)

    def test_biological_summary_doctor_note_profile_renders_narrative_style(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {
                "doc_id": "report_12",
                "page": 1,
                "analyte": "Créatinine",
                "current_value": "23",
                "unit": "mg/l",
                "reference": "4 - 9 mg/l",
                "technical_status_code": "above_reference",
            },
            {
                "doc_id": "report_12",
                "page": 2,
                "analyte": "Bilirubine Directe",
                "current_value": "6",
                "unit": "mg/l",
                "reference": "0 - 5 mg/l",
                "technical_status_code": "above_reference",
            },
            {
                "doc_id": "report_12",
                "page": 3,
                "analyte": "Magnésium",
                "current_value": "20",
                "unit": "mg/l",
                "reference": "15 - 22 mg/l",
                "technical_status_code": "within_reference",
            },
        ]
        rendered = ga._build_doc_scoped_biological_summary_answer(
            evidences,
            max_lines=6,
            no_diagnosis=True,
            render_profile="doctor_note",
        )
        low = rendered.lower()
        self.assertIn("note de synthèse médicale", low)
        self.assertIn("points biologiques notables", low)
        self.assertIn("sans diagnostic médical", low)
        self.assertIn("conclusion technique", low)
        self.assertIn("source :", low)
        self.assertNotIn("anormaux :", low)
        self.assertTrue(rendered.startswith("Note de synthèse médicale"))
        self.assertIn("\n\n", rendered)

    def test_biological_summary_contract_doctor_note_enforces_default_conclusion(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {
                "doc_id": "report_12",
                "page": 1,
                "analyte": "Créatinine",
                "technical_status_code": "above_reference",
                "status": "au-dessus de la référence",
            },
            {
                "doc_id": "report_12",
                "page": 2,
                "analyte": "Magnésium",
                "technical_status_code": "within_reference",
                "status": "dans la référence",
            },
        ]
        llm_json_missing_conclusion = '{"anormaux":["Créatinine"],"within_reference":["Magnésium"]}'
        rendered = ga._render_biological_summary_from_contract(
            llm_answer=llm_json_missing_conclusion,
            evidences=evidences,
            max_lines=6,
            no_diagnosis=True,
            render_profile="doctor_note",
        )
        low = rendered.lower()
        self.assertIn("conclusion technique :", low)
        self.assertIn("sans diagnostic", low)

    def test_safe_llm_summary_conclusion_rejects_interpretive_conclusion(self) -> None:
        ga = __import__("generate_answer")
        conclusion = ga._safe_llm_summary_conclusion(
            "Conclusion technique : Les résultats anormaux indiquent une inflammation.",
            no_diagnosis=True,
        )
        self.assertIsNone(conclusion)

    def test_safe_llm_summary_conclusion_rejects_crude_list_style(self) -> None:
        ga = __import__("generate_answer")
        conclusion = ga._safe_llm_summary_conclusion(
            "Analytes anormaux incluent ACIDE URIQUE, AMMONIUM et GGT, sans diagnostic.",
            no_diagnosis=True,
        )
        self.assertIsNone(conclusion)

    def test_doc_scoped_summary_render_profile_detects_doctor_note(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding("Fais une note médecin courte pour report 12, sans diagnostic.")
        profile = ga._doc_scoped_summary_render_profile(qu)
        self.assertEqual(profile, "doctor_note")

    def test_doc_scoped_summary_render_profile_detects_doctor_note_reference_ranges(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding(
            "Tu peux faire une note sur les plages physiologiques et les références dans report 12 ?"
        )
        profile = ga._doc_scoped_summary_render_profile(qu)
        self.assertEqual(profile, "doctor_note_reference_ranges")

    def test_detect_answer_style_short_and_editorial(self) -> None:
        qu_mod = __import__("query_understanding")
        self.assertEqual(
            qu_mod.detect_answer_style(
                "Fais une synthèse biologique courte du report 12. Limite-toi à 3 à 5 lignes."
            ),
            "short",
        )
        self.assertEqual(
            qu_mod.detect_answer_style(
                "Fais une synthèse biologique éditoriale du report 12 avec une phrase d'ouverture et un texte naturel et professionnel."
            ),
            "editorial",
        )

    def test_render_biological_summary_rewrites_crude_llm_conclusion_with_profiled_synthesis(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {
                "doc_id": "report_12",
                "page": 1,
                "row": 1,
                "analyte": "Bilirubine Directe",
                "current_value": "6",
                "unit": "mg/L",
                "reference": "0.00 - 5.00",
                "technical_status_code": "above_reference",
                "status": "au-dessus de la référence",
            },
            {
                "doc_id": "report_12",
                "page": 1,
                "row": 2,
                "analyte": "Créatinine",
                "current_value": "23",
                "unit": "mg/L",
                "reference": "4 - 9",
                "technical_status_code": "above_reference",
                "status": "au-dessus de la référence",
            },
            {
                "doc_id": "report_12",
                "page": 1,
                "row": 3,
                "analyte": "LDH",
                "current_value": "250",
                "unit": "UI/L",
                "reference": "125,00 - 243,00",
                "technical_status_code": "above_reference",
                "status": "au-dessus de la référence",
            },
            {
                "doc_id": "report_12",
                "page": 2,
                "row": 4,
                "analyte": "Magnésium",
                "current_value": "20",
                "unit": "mg/L",
                "reference": "15 - 22",
                "technical_status_code": "within_reference",
                "status": "dans la référence",
            },
        ]
        llm_answer = (
            "Anormaux : Bilirubine Directe = 6 mg/L (réf 0.00 - 5.00, au-dessus).\n"
            "Résultats dans la référence uniquement : Magnésium = 20 mg/L.\n"
            "Conclusion technique : Analytes anormaux incluent ACIDE URIQUE, AMMONIUM, Bilirubine Directe et GGT, sans diagnostic."
        )
        rendered = ga._render_biological_summary_from_contract(
            llm_answer=llm_answer,
            evidences=evidences,
            max_lines=6,
            no_diagnosis=True,
            render_profile="compact_biological_summary",
        )
        low = rendered.lower()
        self.assertIn("conclusion technique : le bilan", low)
        self.assertIn("bilirubine directe", low)
        self.assertIn("créatinine", low)
        self.assertIn("magnésium", low)
        self.assertNotIn("analytes anormaux incluent", low)
        self.assertNotIn("acide urique", low)
        self.assertNotIn("ammonium", low)
        self.assertNotIn("ggt", low)

    def test_finalize_doc_scoped_biological_llm_answer_rewrites_contract_style_output(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding(
            "Fais une synthèse biologique courte du report 12. Limite-toi à 3 à 5 lignes, sans diagnostic."
        )
        evidences = [
            {
                "doc_id": "report_12",
                "page": 1,
                "row": 8,
                "analyte": "Bilirubine Directe",
                "current_value": "6",
                "unit": "mg/L",
                "reference": "0.00 - 5.00",
                "technical_status_code": "above_reference",
                "status": "au-dessus de la référence",
            },
            {
                "doc_id": "report_12",
                "page": 1,
                "row": 13,
                "analyte": "Créatinine",
                "current_value": "23",
                "unit": "mg/L",
                "reference": "4 - 9",
                "technical_status_code": "above_reference",
                "status": "au-dessus de la référence",
            },
        ]
        raw = (
            "Anormaux : Bilirubine Directe = 6 mg/l (réf 0.00 - 5.00, au-dessus); "
            "Créatinine = 23 mg/l (réf 4 - 9, au-dessus).\n"
            "Résultats dans la référence uniquement : aucun résultat strictement dans la référence parmi les éléments sélectionnés.\n"
            "Conclusion technique : Analytes anormaux et normaux identifiés, mais pas de conclusions diagnostiques ici, sans diagnostic."
        )
        rendered = ga._finalize_doc_scoped_biological_llm_answer(
            llm_answer=raw,
            displayed_evidences=evidences,
            query_understanding=qu,
        )
        low = rendered.lower()
        self.assertIn("conclusion technique : le bilan", low)
        self.assertNotIn("anormaux :", low)
        self.assertNotIn("résultats dans la référence uniquement", low)
        self.assertNotIn("analytes anormaux et normaux identifiés", low)

    def test_doc_scoped_biological_summary_uses_repaired_llm_answer_after_candidate_validation_failure(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        candidate = (
            "Anormaux : Bilirubine Directe = 6 mg/l (réf 0.00 - 5.00, au-dessus).\n"
            "Résultats dans la référence uniquement : aucun résultat strictement dans la référence parmi les éléments sélectionnés.\n"
            "Conclusion technique : Analytes anormaux et normaux identifiés, sans diagnostic."
        )
        repaired = (
            "Le bilan met surtout en évidence une élévation de la bilirubine directe et de la créatinine parmi les anomalies les plus notables. "
            "Aucun résultat strictement dans la référence n’est mis en avant dans les éléments sélectionnés. "
            "Cette synthèse reste strictement descriptive et doit être interprétée avec prudence, sans diagnostic."
        )
        call_count = {"n": 0}

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            call_count["n"] += 1
            answer = candidate if call_count["n"] == 1 else repaired
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": answer,
                "llm_candidate_answer": answer,
                "llm_error": None,
                "use_micro_prompt": True,
            }

        original_validate = __import__("generate_answer").validate_answer

        def _validate_candidate_then_repair(*args, **kwargs):
            answer_text = str(kwargs.get("answer_text") or "").strip()
            if answer_text == candidate.strip():
                return {"validation_status": "fail", "errors": ["output_format_not_respected"], "warnings": []}
            if answer_text.startswith(repaired.strip()):
                return {"validation_status": "pass", "errors": [], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro), mock.patch(
                "generate_answer.validate_answer", side_effect=_validate_candidate_then_repair
            ):
                result = run_generation(
                    query="Fais une synthèse biologique courte du report 12. Limite-toi à 3 à 5 lignes, mentionne uniquement les anomalies majeures, les résultats dans la référence et une conclusion prudente, sans diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        self.assertEqual(str(result.get("final_answer_source") or ""), "llm_writer_repaired")
        self.assertTrue(bool(result.get("llm_repair_attempted")))
        self.assertEqual(str(result.get("llm_repair_status") or ""), "passed")
        self.assertEqual(str(result.get("fallback_reason") or ""), "")
        self.assertEqual(list(result.get("llm_candidate_validation_errors") or []), ["output_format_not_respected"])
        self.assertEqual(str(result.get("llm_candidate_rejected_reason") or ""), "")
        self.assertTrue(str(result.get("llm_repaired_answer") or "").startswith(repaired))

    def test_doc_scoped_biological_summary_repair_failure_exposes_reason_before_deterministic_fallback(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        candidate = (
            "Anormaux : Bilirubine Directe = 6 mg/l (réf 0.00 - 5.00, au-dessus).\n"
            "Résultats dans la référence uniquement : aucun résultat strictement dans la référence parmi les éléments sélectionnés.\n"
            "Conclusion technique : Analytes anormaux et normaux identifiés, sans diagnostic."
        )
        repaired = "Conclusion technique : sans diagnostic."
        call_count = {"n": 0}

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            call_count["n"] += 1
            answer = candidate if call_count["n"] == 1 else repaired
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": answer,
                "llm_candidate_answer": answer,
                "llm_error": None,
                "use_micro_prompt": True,
            }

        original_validate = __import__("generate_answer").validate_answer

        def _validate_fail_both(*args, **kwargs):
            answer_text = str(kwargs.get("answer_text") or "").strip()
            if answer_text == candidate.strip():
                return {"validation_status": "fail", "errors": ["output_format_not_respected"], "warnings": []}
            if answer_text.startswith(repaired.strip()):
                return {"validation_status": "fail", "errors": ["missing_professional_intro"], "warnings": ["missing_conclusion"]}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro), mock.patch(
                "generate_answer.validate_answer", side_effect=_validate_fail_both
            ):
                result = run_generation(
                    query="Fais une synthèse biologique courte du report 12. Limite-toi à 3 à 5 lignes, mentionne uniquement les anomalies majeures, les résultats dans la référence et une conclusion prudente, sans diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        self.assertEqual(str(result.get("final_answer_source") or ""), "deterministic_renderer")
        self.assertTrue(bool(result.get("llm_repair_attempted")))
        self.assertEqual(str(result.get("llm_repair_status") or ""), "failed_validation")
        self.assertEqual(str(result.get("fallback_reason") or ""), "llm_validation_failed_after_repair")
        self.assertEqual(str(result.get("renderer_used") or ""), "deterministic_doc_scoped_biological_summary_fallback")
        self.assertEqual(
            str(result.get("llm_candidate_rejected_reason") or ""),
            "validation_error:output_format_not_respected",
        )

    def test_biological_summary_doctor_note_reference_ranges_includes_range_section(self) -> None:
        ga = __import__("generate_answer")
        evidences = [
            {
                "doc_id": "report_12",
                "page": 1,
                "analyte": "Créatinine",
                "current_value": "23",
                "unit": "mg/l",
                "reference": "4 - 9 mg/l",
                "technical_status_code": "above_reference",
            },
            {
                "doc_id": "report_12",
                "page": 2,
                "analyte": "Magnésium",
                "current_value": "20",
                "unit": "mg/l",
                "reference": "15 - 22 mg/l",
                "technical_status_code": "within_reference",
            },
        ]
        rendered = ga._build_doc_scoped_biological_summary_answer(
            evidences,
            max_lines=7,
            no_diagnosis=True,
            render_profile="doctor_note_reference_ranges",
        )
        low = rendered.lower()
        self.assertIn("plages et statuts documentés", low)
        self.assertIn("créatinine", low)
        self.assertIn("magnésium", low)
        self.assertIn("= 23 mg/l", low)

    def test_reference_ranges_summary_llm_pack_contains_structured_categories(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding(
            "Tu peux faire une note sur les différentes plages physiologiques et références selon sexe/âge dans report 12 ?"
        )
        pack = {
            "evidences": [
                {
                    "doc_id": "report_12",
                    "page": 1,
                    "analyte": "Créatinine",
                    "current_value": "23",
                    "unit": "mg/l",
                    "reference": "Homme: 7.2 - 12.5 ; Femme: 5.7 - 11.1",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                },
                {
                    "doc_id": "report_12",
                    "page": 2,
                    "analyte": "CKMB",
                    "current_value": "40",
                    "unit": "UI/L",
                    "reference": "< 25",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                },
                {
                    "doc_id": "report_12",
                    "page": 2,
                    "analyte": "Cholestérol total",
                    "current_value": "1.60",
                    "unit": "g/L",
                    "reference": "Souhaitable < 2.0 ; Modéré 2.0-2.4 ; Élevé > 2.4",
                    "technical_status_code": "within_reference",
                    "technical_status": "dans la référence",
                },
            ],
            "evidence_all_summary": [
                {
                    "doc_id": "report_12",
                    "page": 1,
                    "analyte": "Créatinine",
                    "current_value": "23",
                    "unit": "mg/l",
                    "reference": "Homme: 7.2 - 12.5 ; Femme: 5.7 - 11.1",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                },
                {
                    "doc_id": "report_12",
                    "page": 2,
                    "analyte": "CKMB",
                    "current_value": "40",
                    "unit": "UI/L",
                    "reference": "< 25",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                },
                {
                    "doc_id": "report_12",
                    "page": 2,
                    "analyte": "Cholestérol total",
                    "current_value": "1.60",
                    "unit": "g/L",
                    "reference": "Souhaitable < 2.0 ; Modéré 2.0-2.4 ; Élevé > 2.4",
                    "technical_status_code": "within_reference",
                    "technical_status": "dans la référence",
                },
            ],
            "requested_doc_ids": ["report_12"],
        }
        llm_pack, _ = ga._build_llm_evidence_pack(
            query_understanding=qu,
            structured_pack=pack,
            selected_route="reference_ranges_summary",
        )
        facts = dict(llm_pack.get("reference_ranges_summary_facts") or {})
        counts = dict(facts.get("category_counts") or {})
        self.assertGreaterEqual(int(counts.get("ranges_by_sex") or 0), 1)
        self.assertGreaterEqual(int(counts.get("threshold_ranges") or 0), 1)
        self.assertGreaterEqual(int(counts.get("interpretive_categories") or 0), 1)
        self.assertEqual(
            str((llm_pack.get("summary_selection_debug") or {}).get("summary_selection_strategy") or ""),
            "reference_ranges_summary_structured_categories",
        )

    def test_reference_ranges_summary_route_specific_fallback_answer(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding(
            "Tu peux faire une note sur les différentes plages physiologiques dans report 12 ?"
        )
        answer = ga._build_route_specific_short_fallback_answer(
            selected_route="reference_ranges_summary",
            query_understanding=qu,
            displayed_evidences=[
                {
                    "doc_id": "report_12",
                    "page": 1,
                    "analyte": "Bilirubine Directe",
                    "current_value": "6",
                    "unit": "mg/l",
                    "reference": "0.00 - 5.00",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                }
            ],
            evidence_all_summary=None,
            default_answer="fallback",
        )
        self.assertIn("Note sur les valeurs physiologiques", answer)
        self.assertIn("Source :", answer)
        self.assertNotIn("Paramètres hors référence notables", answer)
        self.assertNotIn("Points biologiques notables", answer)

    def test_reference_ranges_summary_facts_filters_ocr_noisy_analytes(self) -> None:
        ga = __import__("generate_answer")
        facts = ga._build_reference_ranges_summary_facts(
            [
                {
                    "doc_id": "report_12",
                    "page": 1,
                    "analyte": "associés < 1 g/L après un infarctus du myocarde",
                    "current_value": "0.23",
                    "unit": "g/l",
                    "reference": "cérébral ou chez les diabétiques à haut risque.",
                    "technical_status_code": "unknown",
                },
                {
                    "doc_id": "report_12",
                    "page": 2,
                    "analyte": "Créatinine",
                    "current_value": "23",
                    "unit": "mg/l",
                    "reference": "Femme : 5,7 - 11,1 mg/l",
                    "technical_status_code": "above_reference",
                },
            ]
        )
        total = int(facts.get("total_reference_items") or 0)
        self.assertEqual(total, 1)
        all_items = []
        for key in [
            "ranges_min_max",
            "ranges_by_sex",
            "ranges_by_age",
            "threshold_ranges",
            "interpretive_categories",
            "unclassified",
        ]:
            all_items.extend(list(facts.get(key) or []))
        analytes = {str(item.get("analyte") or "") for item in all_items}
        self.assertIn("Créatinine", analytes)
        self.assertNotIn("associés < 1 g/L après un infarctus du myocarde", analytes)

    def test_reference_ranges_summary_postprocess_forces_conclusion(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding(
            "tu peux faire une note pour les differents plages qui exist dans les valeurs phisiologique dans report 12"
        )
        meta = ga._postprocess_reference_ranges_summary_answer(
            answer_text="**Types de références physiologiques présentes dans le document**\n* min-max ...",
            displayed_evidences=[
                {
                    "doc_id": "report_12",
                    "page": 1,
                    "analyte": "AMMONIUM",
                    "current_value": "20",
                    "unit": "µg/dl",
                    "reference": "31,00 - 123,00",
                    "technical_status_code": "below_reference",
                }
            ],
            evidence_all_summary=None,
            query_understanding=qu,
        )
        answer = str(meta.get("answer") or "")
        self.assertIn("Note sur les valeurs physiologiques", answer)
        self.assertIn("Conclusion technique :", answer)
        self.assertIn("Source :", answer)
        self.assertEqual(str(meta.get("answer_source") or ""), "deterministic_renderer")
        self.assertEqual(str(meta.get("renderer_used") or ""), "reference_ranges_deterministic_fallback")

    def test_reference_ranges_summary_fallback_has_no_report12_must_include_bias(self) -> None:
        ga = __import__("generate_answer")
        answer = ga._build_reference_ranges_deterministic_fallback(
            [
                {
                    "doc_id": "report_99",
                    "page": 1,
                    "analyte": "AnA",
                    "reference": "10 - 20",
                },
                {
                    "doc_id": "report_99",
                    "page": 1,
                    "analyte": "AnB",
                    "reference": "Homme : 10 - 20 ; Femme : 9 - 18",
                },
                {
                    "doc_id": "report_99",
                    "page": 1,
                    "analyte": "AnC",
                    "reference": "Adulte : 4 - 10 ; Enfant : 3 - 8",
                },
                {
                    "doc_id": "report_99",
                    "page": 2,
                    "analyte": "AnD",
                    "reference": "< 25",
                },
                {
                    "doc_id": "report_99",
                    "page": 2,
                    "analyte": "AnE",
                    "reference": "Taux souhaitable: < 2.0 ; Modéré 2.0-2.4 ; Élevé > 2.4",
                },
            ],
            max_lines=7,
            no_diagnosis=True,
        )
        n = ga.norm_text(answer)
        self.assertIn("ana", n)
        self.assertIn("anb", n)
        self.assertIn("anc", n)
        self.assertIn("and", n)
        self.assertIn("ane", n)
        for forbidden in ["crp", "albumine", "reserve alcaline", "magnesium plasmatique", "lipase", "ckmb", "aslo"]:
            self.assertNotIn(forbidden, n)

    def test_reference_ranges_summary_postprocess_rejects_list_like_llm_with_deterministic_source(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding(
            "tu peux faire une note pour les differents plages qui exist dans les valeurs phisiologique dans report 12"
        )
        meta = ga._postprocess_reference_ranges_summary_answer(
            answer_text=(
                "Note sur les valeurs physiologiques — report_12.\n"
                "Plages min-max : Albumine, ASAT.\n"
                "Références selon âge/sexe : Créatinine, GGT.\n"
                "Seuils et catégories interprétatives : Lipase, CKMB.\n"
                "Source : report_12, pages 1-3."
            ),
            displayed_evidences=[
                {"doc_id": "report_12", "page": 1, "analyte": "Albumine", "reference": "35 - 50"},
                {"doc_id": "report_12", "page": 2, "analyte": "Lipase", "reference": "< 60"},
            ],
            evidence_all_summary=None,
            query_understanding=qu,
            prefer_llm_text=True,
        )
        self.assertEqual(str(meta.get("answer_source") or ""), "deterministic_renderer")
        self.assertEqual(str(meta.get("renderer_used") or ""), "reference_ranges_deterministic_fallback")
        self.assertEqual(str(meta.get("fallback_reason") or ""), "llm_writer_too_deterministic_or_list_like")

    def test_reference_ranges_summary_postprocess_accepts_valid_llm_narrative_source(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding(
            "tu peux faire une note pour les differents plages qui exist dans les valeurs phisiologique dans report 12"
        )
        narrative = (
            "Note sur les valeurs physiologiques — report_12.\n"
            "Le rapport contient plusieurs formats de valeurs physiologiques : plages min-max, seuils numériques, références selon l’âge, selon le sexe et catégories interprétatives.\n"
            "Les plages min-max concernent notamment l’albumine et l’ASAT.\n"
            "Certaines références varient selon le profil patient, notamment pour la créatinine et la GGT.\n"
            "D'autres paramètres utilisent des seuils ou catégories, notamment la lipase et la CK-MB.\n"
            "Ces références servent à structurer une lecture technique du rapport.\n"
            "Note descriptive uniquement, sans diagnostic médical.\n"
            "Source : report_12, pages 1-3."
        )
        meta = ga._postprocess_reference_ranges_summary_answer(
            answer_text=narrative,
            displayed_evidences=[
                {"doc_id": "report_12", "page": 1, "analyte": "Albumine", "reference": "35 - 50"},
                {"doc_id": "report_12", "page": 2, "analyte": "Lipase", "reference": "< 60"},
            ],
            evidence_all_summary=None,
            query_understanding=qu,
            prefer_llm_text=True,
        )
        self.assertEqual(str(meta.get("answer_source") or ""), "llm_writer")
        self.assertIsNone(meta.get("renderer_used"))
        self.assertIsNone(meta.get("fallback_reason"))

    def test_reference_ranges_summary_postprocess_multidoc_requires_doc_mentions_in_body(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding(
            "fais une note des valeurs physiologiques dans report 12 et report 24"
        )
        narrative_missing_body_docs = (
            "Note sur les valeurs physiologiques — report_12, report_24.\n"
            "Le rapport contient plusieurs formats de valeurs physiologiques : plages min-max, seuils numériques, références selon l’âge, selon le sexe et catégories interprétatives.\n"
            "Les plages min-max concernent notamment l’albumine et l’ASAT.\n"
            "Certaines références varient selon le profil patient, notamment pour la créatinine et la GGT.\n"
            "D'autres paramètres utilisent des seuils ou catégories, notamment la lipase et la CK-MB.\n"
            "Ces références servent à structurer une lecture technique du rapport.\n"
            "Note descriptive uniquement, sans diagnostic médical.\n"
            "Source : report_12, report_24, pages 1-3."
        )
        meta = ga._postprocess_reference_ranges_summary_answer(
            answer_text=narrative_missing_body_docs,
            displayed_evidences=[
                {"doc_id": "report_12", "page": 1, "analyte": "Albumine", "reference": "35 - 50"},
                {"doc_id": "report_24", "page": 1, "analyte": "ACIDE URIQUE", "reference": "26-60"},
            ],
            evidence_all_summary=None,
            query_understanding=qu,
            prefer_llm_text=True,
        )
        self.assertEqual(str(meta.get("answer_source") or ""), "deterministic_renderer")
        self.assertEqual(str(meta.get("fallback_reason") or ""), "llm_writer_multidoc_coverage_missing")

    def test_reference_ranges_summary_postprocess_multidoc_accepts_when_docs_in_body(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding(
            "fais une note des valeurs physiologiques dans report 12 et report 24"
        )
        narrative_with_body_docs = (
            "Note sur les valeurs physiologiques — report_12, report_24.\n"
            "Le rapport contient plusieurs formats de valeurs physiologiques dans report_12 et report_24 : plages min-max, seuils numériques, références selon l’âge, selon le sexe et catégories interprétatives.\n"
            "Dans report_12, les plages min-max concernent notamment l’albumine et l’ASAT.\n"
            "Dans report_24, certaines références varient selon le profil patient, notamment pour l’acide urique et le calcium.\n"
            "D'autres paramètres utilisent des seuils ou catégories interprétatives, comme la réserve alcaline et la CRP.\n"
            "Ces références servent à structurer une lecture technique du rapport.\n"
            "Note descriptive uniquement, sans diagnostic médical.\n"
            "Source : report_12, report_24, pages 1-3."
        )
        meta = ga._postprocess_reference_ranges_summary_answer(
            answer_text=narrative_with_body_docs,
            displayed_evidences=[
                {"doc_id": "report_12", "page": 1, "analyte": "Albumine", "reference": "35 - 50"},
                {"doc_id": "report_24", "page": 1, "analyte": "ACIDE URIQUE", "reference": "26-60"},
            ],
            evidence_all_summary=None,
            query_understanding=qu,
            prefer_llm_text=True,
        )
        self.assertEqual(str(meta.get("answer_source") or ""), "llm_writer")
        self.assertIsNone(meta.get("fallback_reason"))

    def test_reference_ranges_summary_style_guard_production(self) -> None:
        ga = __import__("generate_answer")
        answer = ga._build_reference_ranges_summary_answer(
            [
                {
                    "doc_id": "report_12",
                    "page": 1,
                    "analyte": "Albumine",
                    "current_value": "40",
                    "unit": "g/l",
                    "reference": "35 à 50 g/l",
                },
                {
                    "doc_id": "report_12",
                    "page": 1,
                    "analyte": "ACIDE URIQUE",
                    "current_value": "23",
                    "unit": "mg/l",
                    "reference": "Homme : 35 - 72 mg/l Femme: 26-60 mg/l",
                },
                {
                    "doc_id": "report_12",
                    "page": 1,
                    "analyte": "MAGNESIUM PLASMATIQUE",
                    "current_value": "20",
                    "unit": "mg/l",
                    "reference": "Nouveau-né : 15 à 22 mg/l Enfant : 17 à 23 mg/l Adulte : 16 à 26 mg/l",
                },
                {
                    "doc_id": "report_12",
                    "page": 2,
                    "analyte": "Lipase",
                    "current_value": "14",
                    "unit": "UI/l",
                    "reference": "<60 UI/l",
                },
                {
                    "doc_id": "report_12",
                    "page": 2,
                    "analyte": "CKMB (CPKMB)",
                    "current_value": "40",
                    "unit": "UI/L",
                    "reference": "<25 UI/L",
                },
                {
                    "doc_id": "report_12",
                    "page": 2,
                    "analyte": "Cholestérol total",
                    "current_value": "1.60",
                    "unit": "g/l",
                    "reference": "Adulte Taux souhaitable: < 2 g/l Taux modéré: 2 - 2,39 g/l Taux élévé: > 2,40 g/l",
                },
            ],
            max_lines=7,
            no_diagnosis=True,
        )
        first_line = str(answer.splitlines()[0] if answer.splitlines() else "")
        self.assertFalse(first_line.lower().startswith("plages min-max :"))
        self.assertNotIn("\nPlages min-max :", answer)
        self.assertNotIn("\nRéférences selon âge/sexe :", answer)
        self.assertNotIn("\nSeuils et catégories interprétatives :", answer)
        self.assertIn("Le rapport contient plusieurs formats de valeurs physiologiques", answer)
        n = str(answer).lower()
        self.assertIn("plages min-max", n)
        self.assertIn("seuil", n)
        self.assertIn("âge", answer)
        self.assertIn("sexe", answer)
        self.assertIn("catégories interprétatives", answer)
        self.assertIn("sans diagnostic médical", n)

    def test_reference_ranges_summary_llm_style_guard_rejects_list_like_answer(self) -> None:
        ga = __import__("generate_answer")
        list_like = (
            "Note sur les valeurs physiologiques — report_12.\n"
            "Le rapport contient plusieurs formats de valeurs physiologiques.\n"
            "Note: Références selon âge/sexe : Créatinine, GGT.\n"
            "Note: Seuils et catégories interprétatives : Lipase, CKMB.\n"
            "Source : report_12, pages 1-3."
        )
        self.assertFalse(ga._is_reference_ranges_narrative_answer(list_like))

    def test_reference_ranges_summary_llm_style_guard_accepts_narrative_answer(self) -> None:
        ga = __import__("generate_answer")
        narrative = (
            "Note sur les valeurs physiologiques — report_12.\n"
            "Le rapport contient plusieurs formats de valeurs physiologiques : plages min-max, seuils numériques, références selon l’âge, selon le sexe et catégories interprétatives.\n"
            "Les plages min-max concernent notamment l’albumine, l’ammonium et le phosphore.\n"
            "Certaines références varient selon le profil patient, notamment pour la créatinine et la GGT.\n"
            "D'autres paramètres utilisent des seuils ou catégories, notamment la lipase, la CK-MB et les triglycérides.\n"
            "Ces références servent à structurer une lecture technique du rapport.\n"
            "Conclusion technique : note descriptive uniquement, sans diagnostic médical.\n"
            "Source : report_12, pages 1-3."
        )
        self.assertTrue(ga._is_reference_ranges_narrative_answer(narrative))

    def test_reference_ranges_summary_report1_mentions_age_sex_profiles(self) -> None:
        ga = __import__("generate_answer")
        result = run_generation(
            query="tu peux faire une note pour les differents plages qui exist dans les valeurs phisiologique dans report 1",
            mode="keyword",
            top_k=30,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "reference_ranges_summary")
        answer_n = ga.norm_text(str(result.get("answer") or ""))
        self.assertNotIn("aucun exemple exploitable", answer_n)
        self.assertTrue(any(token in answer_n for token in ["profil patient", "homme", "femme", "amh"]))

    def test_report10_unmatched_directional_claim_triggers_quality_gate_fallback(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        candidate = (
            "Note médicale — report_10.\n"
            "Le document contient plusieurs résultats biologiques exploitables.\n"
            "Insuline est au-dessus de la référence selon les lignes analysées.\n"
            "Conclusion technique : note descriptive uniquement, sans diagnostic médical.\n"
            "Source : report_10, pages 1-3."
        )

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": candidate,
                "llm_candidate_answer": candidate,
                "llm_error": None,
                "use_micro_prompt": True,
            }

        original_validate = __import__("generate_answer").validate_answer

        def _validate_pass_candidate(*args, **kwargs):
            if str(kwargs.get("answer_text") or "").strip() == candidate.strip():
                return {"validation_status": "pass", "errors": [], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro), mock.patch(
                "generate_answer.validate_answer", side_effect=_validate_pass_candidate
            ):
                result = run_generation(
                    query="tu peux faire une note medcin de report 10",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        self.assertEqual(str(result.get("final_answer_source") or ""), "deterministic_renderer")
        self.assertFalse(bool(result.get("llm_writer_accepted")))
        quality_gate = dict(result.get("quality_gate") or {})
        reasons = [str(r) for r in list(quality_gate.get("reasons") or [])]
        self.assertIn("directional_claim_unmatched_analyte", reasons)

    def test_summary_quality_gate_rejects_anomaly_contradiction(self) -> None:
        ga = __import__("generate_answer")
        gate = ga._evaluate_summary_quality_gate(
            answer=(
                "Résumé technique\n"
                "Anomalies détectées · 1\n"
                "• aucune anomalie objectivée dans les lignes exploitables."
            ),
            selected_route="doc_scoped_biological_summary",
            displayed_evidences=[
                {"analyte": "CRP", "technical_status_code": "above_reference"},
            ],
        )
        self.assertFalse(bool(gate.get("pass")))
        self.assertIn("contradictory_anomaly_claim", [str(r) for r in list(gate.get("reasons") or [])])

    def test_summary_quality_gate_uses_inferred_status_for_directional_claims(self) -> None:
        ga = __import__("generate_answer")
        gate = ga._evaluate_summary_quality_gate(
            answer="Créatinine = 23 mg/l (réf 4 - 9 mg/l, au-dessus de la référence).",
            selected_route="doc_scoped_biological_summary",
            displayed_evidences=[
                {
                    "analyte": "Créatinine",
                    "current_value": "23",
                    "reference": "4 - 9 mg/l",
                    "interpretation_status": "needs_clinical_context",
                }
            ],
        )
        self.assertTrue(bool(gate.get("pass")))
        self.assertNotIn(
            "directional_claim_on_ambiguous_status:Créatinine",
            [str(r) for r in list(gate.get("reasons") or [])],
        )

    def test_summary_quality_gate_matches_directional_claims_on_canonical_analyte_aliases(self) -> None:
        ga = __import__("generate_answer")
        candidate = (
            "Anormaux : Bilirubine Directe = 6 mg/l (réf 0.00 - 5.00, au-dessus); "
            "Créatinine = 23 mg/l (réf 4 - 9, au-dessus); "
            "LDH = 250 UI/L (réf 125,00 - 243,00, au-dessus); "
            "CKMB (CPKMB) = 40 UI/L (réf < 25, au-dessus); "
            "APOLIPOPROTÉINE A1 (APO A1) = 2.3 g/L (réf 1,1 - 1,6, au-dessus); "
            "et 4 autre(s) anomalie(s).\n"
            "Résultats dans la référence uniquement : aucun résultat strictement dans la référence parmi les éléments sélectionnés.\n"
            "Conclusion technique : Analytes anormaux incluent ACIDE URIQUE, AMMONIUM, Bilirubine Directe, Créatinine et GGT, sans diagnostic."
        )
        gate = ga._evaluate_summary_quality_gate(
            answer=candidate,
            selected_route="doc_scoped_biological_summary",
            displayed_evidences=[
                {"analyte": "Bilirubine Directe", "current_value": "6", "reference": "0.00 - 5.00", "interpretation_status": "needs_clinical_context"},
                {"analyte": "Créatinine", "current_value": "23", "reference": "4 - 9", "interpretation_status": "needs_clinical_context"},
                {"analyte": "LDH", "current_value": "250", "reference": "125,00 - 243,00", "interpretation_status": "needs_clinical_context"},
                {"analyte": "CK-MB (CPKMB)", "current_value": "40", "reference": "0 - 25", "interpretation_status": "needs_clinical_context"},
                {"analyte": "APOLIPOPROTÉINE A1 (APO A1)", "current_value": "2.3", "reference": "1,1 - 1,6", "interpretation_status": "needs_clinical_context"},
            ],
        )
        self.assertTrue(bool(gate.get("pass")))
        self.assertNotIn(
            "directional_claim_unmatched_analyte",
            [str(r) for r in list(gate.get("reasons") or [])],
        )
        self.assertNotIn(
            "directional_claim_on_ambiguous_status:CK-MB (CPKMB)",
            [str(r) for r in list(gate.get("reasons") or [])],
        )

    def test_doc_scoped_biological_summary_keeps_llm_on_soft_quality_warnings(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        candidate = (
            "Note technique pour report_12 : 20-24 ans — 20-24 ans\n"
            "Source : report_12, pages 1-3"
        )

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": candidate,
                "llm_candidate_answer": candidate,
                "llm_error": None,
                "use_micro_prompt": True,
            }

        original_validate = __import__("generate_answer").validate_answer

        def _validate_pass_candidate(*args, **kwargs):
            if str(kwargs.get("answer_text") or "").strip() == candidate.strip():
                return {"validation_status": "pass", "errors": [], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro), mock.patch(
                "generate_answer.validate_answer", side_effect=_validate_pass_candidate
            ):
                result = run_generation(
                    query="Fais une synthèse médico-biologique du report 12 en 6 lignes maximum, en séparant les anomalies et les résultats normaux, sans diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        self.assertEqual(str(result.get("final_answer_source") or ""), "llm_writer")
        self.assertIsNone(result.get("renderer_used"))
        self.assertIsNone(result.get("fallback_reason"))
        quality_gate = dict(result.get("quality_gate") or {})
        self.assertTrue(bool(quality_gate.get("accepted_with_warnings")))
        self.assertTrue(bool(quality_gate.get("soft_warning_only")))
        self.assertTrue(bool(result.get("llm_writer_accepted")))
        self.assertEqual(str(result.get("generation_mode") or ""), "hybrid_structured_llm_writer")

    def test_doc_scoped_biological_summary_keeps_llm_when_directional_aliases_match(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        candidate = (
            "Anormaux : Bilirubine Directe = 6 mg/l (réf 0.00 - 5.00, au-dessus); "
            "Créatinine = 23 mg/l (réf 4 - 9, au-dessus); "
            "LDH = 250 UI/L (réf 125,00 - 243,00, au-dessus); "
            "CKMB (CPKMB) = 40 UI/L (réf < 25, au-dessus); "
            "APOLIPOPROTÉINE A1 (APO A1) = 2.3 g/L (réf 1,1 - 1,6, au-dessus); "
            "et 4 autre(s) anomalie(s).\n"
            "Résultats dans la référence uniquement : aucun résultat strictement dans la référence parmi les éléments sélectionnés.\n"
            "Conclusion technique : Analytes anormaux incluent ACIDE URIQUE, AMMONIUM, Bilirubine Directe, Créatinine et GGT, sans diagnostic."
        )

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": candidate,
                "llm_candidate_answer": candidate,
                "llm_error": None,
                "use_micro_prompt": True,
            }

        original_validate = __import__("generate_answer").validate_answer

        def _validate_pass_candidate(*args, **kwargs):
            if str(kwargs.get("answer_text") or "").strip() == candidate.strip():
                return {"validation_status": "pass", "errors": [], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro), mock.patch(
                "generate_answer.validate_answer", side_effect=_validate_pass_candidate
            ):
                result = run_generation(
                    query="Fais une synthèse médico-biologique du report 12 en 6 lignes maximum, en séparant les anomalies et les résultats dans la référence uniquement. Ne donne pas de diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        self.assertEqual(str(result.get("final_answer_source") or ""), "llm_writer")
        self.assertTrue(bool(result.get("llm_writer_accepted")))
        self.assertIsNone(result.get("fallback_reason"))
        self.assertEqual(str(result.get("generation_mode") or ""), "hybrid_structured_llm_writer")

    def test_summary_readability_normalizer_splits_concatenated_checkmarks(self) -> None:
        ga = __import__("generate_answer")
        raw = "✓ AMPHÉTAMINE QUALITATIF✓ BENZODIAZÉPINE QUALITATIF✓ COCAÏNE QUALITATIF."
        before = ga._evaluate_summary_quality_gate(
            answer=raw,
            selected_route="doc_scoped_biological_summary",
            displayed_evidences=[],
        )
        normalized = ga._normalize_summary_readability(raw)
        after = ga._evaluate_summary_quality_gate(
            answer=normalized,
            selected_route="doc_scoped_biological_summary",
            displayed_evidences=[],
        )
        self.assertIn("readability_concatenated_tokens", [str(r) for r in list(before.get("reasons") or [])])
        self.assertNotIn("readability_concatenated_tokens", [str(r) for r in list(after.get("reasons") or [])])
        self.assertIn(" ; ", normalized)

    def test_llm_evidence_pack_keeps_doc_scope_fields_for_summary(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding("Fais une note médecin courte pour report 12, sans diagnostic.")
        pack = {
            "evidences": [
                {
                    "doc_id": "report_12",
                    "page": 2,
                    "analyte": "Créatinine",
                    "current_value": "23",
                    "unit": "mg/l",
                    "reference": "4 - 9",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                }
            ],
            "evidence_all_summary": [
                {
                    "doc_id": "report_12",
                    "page": 2,
                    "analyte": "Créatinine",
                    "current_value": "23",
                    "unit": "mg/l",
                    "reference": "4 - 9",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                }
            ],
            "requested_doc_ids": ["report_12"],
        }
        llm_pack, _ = ga._build_llm_evidence_pack(
            query_understanding=qu,
            structured_pack=pack,
            selected_route="doc_scoped_biological_summary",
        )
        first = dict((llm_pack.get("evidences") or [])[0])
        self.assertEqual(str(first.get("doc_id") or ""), "report_12")
        self.assertEqual(int(first.get("page") or 0), 2)

    def test_llm_summary_pack_keeps_within_rows_when_available(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding("Résume report 24 en 5 lignes max, strictement technique")
        evidence_all = [
            {
                "analyte": f"A{i}",
                "technical_status_code": "above_reference",
                "technical_status": "au-dessus de la référence",
                "current_value": "10",
                "unit": "u",
                "reference": "0-5",
                "source": "src",
            }
            for i in range(1, 7)
        ]
        evidence_all.append(
            {
                "analyte": "Calcium",
                "technical_status_code": "within_reference",
                "technical_status": "dans la référence",
                "current_value": "92",
                "unit": "mg/l",
                "reference": "80-100",
                "source": "src",
            }
        )
        llm_pack, _ = ga._build_llm_evidence_pack(
            query_understanding=qu,
            structured_pack={"evidences": evidence_all, "evidence_all_summary": evidence_all},
            selected_route="doc_scoped_biological_summary",
        )
        self.assertGreaterEqual(int(llm_pack.get("llm_within_rows_count") or 0), 1)
        compact_statuses = [str(ev.get("status") or "") for ev in list(llm_pack.get("evidences") or [])]
        self.assertTrue(any("dans la référence" in s.lower() or "dans la reference" in s.lower() for s in compact_statuses))

    def test_priority_is_deterministic_preferred_by_default(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        if "MEDICAL_RAG_FORCE_LLM_WRITER" in os.environ:
            os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
        try:
            result = run_generation(
                query="Dans le report 10, explique les anomalies biologiques les plus importantes par priorité technique, avec une justification courte pour chaque anomalie. Ne pose pas de diagnostic.",
                mode="keyword",
                top_k=30,
                index_dir="data/indexes",
            )
        finally:
            if old_force is not None:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("selected_route") or ""), "doc_scoped_priority_anomalies")
        self.assertEqual(str(debug.get("generation_strategy") or ""), "deterministic_preferred")
        self.assertIn(debug.get("llm_expected"), {False, 0})
        self.assertIn(debug.get("llm_writer_attempted"), {False, 0})
        self.assertEqual(str(debug.get("llm_skipped_reason") or ""), "priority_deterministic_structure_preferred")
        self.assertNotEqual(str(debug.get("fallback_reason") or ""), "llm_repair_failed")
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")

    def test_guarded_interpretation_uses_llm_expected_strategy(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        if "MEDICAL_RAG_FORCE_LLM_WRITER" in os.environ:
            os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
        try:
            result = run_generation(
                query="Le bilan thyroïdien du report 16 est-il compatible avec une hyperthyroïdie primaire ? Explique prudemment à partir de TSH, T3, T4 et anticorps, sans conclure à un diagnostic.",
                mode="keyword",
                top_k=30,
                index_dir="data/indexes",
            )
        finally:
            if old_force is not None:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("selected_route") or ""), "doc_scoped_medical_interpretation_guarded")
        self.assertEqual(str(debug.get("llm_route_class") or ""), "llm_allowed")
        self.assertEqual(str(debug.get("llm_prompt_policy_version") or ""), "v2")
        self.assertEqual(str(debug.get("generation_strategy") or ""), "llm_writer_expected")
        self.assertIn(debug.get("llm_expected"), {True, 1})
        self.assertIn(debug.get("llm_writer_attempted"), {True, 1})
        self.assertIn(debug.get("llm_attempt_rate"), {1.0, 1})
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")

    def test_contract_violation_never_sets_llm_attempt_rate_to_one(self) -> None:
        ga = __import__("generate_answer")
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        original_policy = ga._level2_prompt_policy

        def _policy_without_micro(route: str) -> dict[str, object]:
            policy = dict(original_policy(route))
            if str(route or "").strip().lower() == "doc_scoped_biological_summary":
                policy["use_micro_prompt"] = False
            return policy

        def _contract_violation_writer(**_kwargs: object) -> dict[str, object]:
            return {
                "mode": "writer_contract_violation_fallback",
                "answer": (
                    "Synthèse technique indisponible en rédaction assistée. "
                    "Réponse déterministe de repli utilisée à partir des faits disponibles."
                ),
                "llm_error": "writer_evidence_contract_violation",
                "contract_violation": ["scope_incoherent", "results_locked_empty"],
                "llm_prompt_policy_version": "v2",
            }

        try:
            with mock.patch("generate_answer._level2_prompt_policy", side_effect=_policy_without_micro):
                with mock.patch("generate_answer.compose_professional_answer", side_effect=_contract_violation_writer):
                    result = run_generation(
                        query="Fais une synthèse médico-biologique du report 12 en 6 lignes maximum, en séparant les anomalies et les résultats rassurants. Ne donne pas de diagnostic.",
                        mode="keyword",
                        top_k=30,
                        index_dir="data/indexes",
                    )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("selected_route") or ""), "doc_scoped_biological_summary")
        self.assertEqual(int(debug.get("contract_violation_count") or 0), 2)
        self.assertEqual(list(debug.get("contract_violation") or []), ["scope_incoherent", "results_locked_empty"])
        self.assertIn(debug.get("llm_writer_attempted"), {False, 0})
        self.assertEqual(float(debug.get("llm_attempt_rate") or 0.0), 0.0)
        self.assertEqual(float(debug.get("llm_accept_rate") or 0.0), 0.0)
        self.assertEqual(float(debug.get("fallback_after_llm_rate") or 0.0), 0.0)
        self.assertEqual(float((debug.get("stage_timings_ms") or {}).get("llm_writer_ms") or 0.0), 0.0)
        self.assertEqual(str(debug.get("llm_prompt_policy_version") or ""), "v2")

    def test_benchmark_does_not_penalize_deterministic_preferred_routes(self) -> None:
        response = {
            "answer": "Anormaux : test.\nRésultats dans la référence uniquement : aucun.\nConclusion technique : synthèse descriptive limitée aux données disponibles, sans diagnostic.",
            "generation_mode": "deterministic_doc_scoped_biological_summary",
            "generation_writer": "professional_fallback",
            "validation_status": "pass",
            "quality_report": {"final_status": "pass"},
            "displayed_evidences": [{"analyte": "CRP"}],
            "sources": [{"label": "report"}],
            "debug": {
                "generation_strategy": "deterministic_preferred",
                "llm_expected": False,
                "llm_writer_attempted": False,
                "llm_model_override_applied": True,
                "llm_model_requested": "llama3.2:latest",
                "llm_model_effective": "llama3.2:latest",
                "ollama_model": "llama3.2:latest",
                "validation": {"errors": [], "warnings": []},
                "stage_timings_ms": {"llm_writer_ms": 0.0},
            },
        }
        row = _extract_response_fields(
            model="llama3.2:latest",
            question_id="Q1",
            question="dummy",
            response=response,
        )
        self.assertEqual(row["generation_strategy"], "deterministic_preferred")
        self.assertFalse(bool(row["llm_expected"]))
        self.assertFalse(bool(row["llm_writer_attempted"]))
        self.assertGreaterEqual(int(row["score"]), 9)

    def test_level2_timeout_falls_back_with_llm_timeout_reason(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {"mode": "llm_writer_error_fallback", "answer": "", "llm_error": "Ollama timeout", "use_micro_prompt": True}

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro):
                result = run_generation(
                    query="Résume le report 12 en quelques lignes, avec une partie anomalies et une partie résultats normaux, sans conclusion diagnostique.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        self.assertTrue(str(result.get("generation_mode") or "").startswith("deterministic_"))
        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("fallback_reason") or ""), "llm_timeout")
        self.assertIn(debug.get("retry_used"), {False, 0})
        self.assertIsNone(debug.get("llm_candidate_validation_status"))
        self.assertTrue(debug.get("llm_candidate_validation_errors") in (None, []))
        self.assertEqual(str(debug.get("fallback_renderer_used") or ""), "deterministic_biological_summary_short")
        self.assertFalse(bool(result.get("llm_writer_accepted")))
        self.assertEqual(str(result.get("final_answer_source") or ""), "deterministic_renderer")

    def test_level2_candidate_valid_kept_as_final_answer(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        candidate = (
            "Anormaux : bilirubine directe élevée, créatinine élevée.\n"
            "Résultats dans la référence uniquement : Aucun résultat strictement dans la référence parmi les éléments sélectionnés.\n"
            "Conclusion technique : synthèse descriptive, sans diagnostic."
        )

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": candidate,
                "llm_candidate_answer": candidate,
                "llm_error": None,
                "prompt_chars": 1200,
                "llm_prompt_tokens_estimate": 300,
                "use_micro_prompt": True,
                "llm_call_skipped_due_prompt_budget": False,
            }

        original_validate = __import__("generate_answer").validate_answer

        def _validate_pass_candidate(*args, **kwargs):
            if str(kwargs.get("answer_text") or "").strip() == candidate.strip():
                return {"validation_status": "pass", "errors": [], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro), mock.patch(
                "generate_answer.validate_answer", side_effect=_validate_pass_candidate
            ):
                result = run_generation(
                    query="Fais une synthèse médico-biologique du report 12 en 6 lignes maximum, en séparant les anomalies et les résultats dans la référence uniquement. Ne donne pas de diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        self.assertEqual(str(result.get("generation_mode") or ""), "hybrid_structured_llm_writer")
        self.assertEqual(str(result.get("answer") or "").strip(), candidate.strip())
        debug = dict(result.get("debug") or {})
        self.assertIsNone(debug.get("fallback_reason"))
        self.assertIsNone(debug.get("generation_mode_before_fallback"))
        self.assertEqual(str(debug.get("llm_candidate_validation_status") or ""), "pass")

    def test_reference_ranges_pipeline_marks_deterministic_source_when_llm_is_list_like(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        candidate = (
            "Note sur les valeurs physiologiques — report_12.\n"
            "Plages min-max : Albumine, ASAT.\n"
            "Références selon âge/sexe : Créatinine, GGT.\n"
            "Seuils et catégories interprétatives : Lipase, CKMB.\n"
            "Source : report_12, pages 1-3."
        )

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": candidate,
                "llm_candidate_answer": candidate,
                "llm_error": None,
                "use_micro_prompt": True,
            }

        original_validate = __import__("generate_answer").validate_answer

        def _validate_pass_candidate(*args, **kwargs):
            if str(kwargs.get("answer_text") or "").strip() == candidate.strip():
                return {"validation_status": "pass", "errors": [], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro), mock.patch(
                "generate_answer.validate_answer", side_effect=_validate_pass_candidate
            ):
                result = run_generation(
                    query="tu peux faire une note pour les differents plages qui exist dans les valeurs phisiologique dans report 12",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        self.assertEqual(str(result.get("final_answer_source") or ""), "deterministic_renderer")
        self.assertEqual(str(result.get("renderer_used") or ""), "reference_ranges_deterministic_fallback")
        self.assertEqual(str(result.get("fallback_reason") or ""), "llm_writer_too_deterministic_or_list_like")
        self.assertFalse(bool(result.get("llm_writer_accepted")))

    def test_reference_ranges_pipeline_keeps_llm_source_when_narrative_is_valid(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        candidate = (
            "Note sur les valeurs physiologiques — report_12.\n"
            "Le rapport contient plusieurs formats de valeurs physiologiques : plages min-max, seuils numériques, références selon l’âge, selon le sexe et catégories interprétatives.\n"
            "Les plages min-max concernent notamment l’albumine et l’ASAT.\n"
            "Certaines références varient selon le profil patient, notamment pour la créatinine et la GGT.\n"
            "D'autres paramètres utilisent des seuils ou catégories, notamment la lipase et la CK-MB.\n"
            "Ces références servent à structurer une lecture technique du rapport.\n"
            "Note descriptive uniquement, sans diagnostic médical.\n"
            "Source : report_12, pages 1-3."
        )

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": candidate,
                "llm_candidate_answer": candidate,
                "llm_error": None,
                "use_micro_prompt": True,
            }

        original_validate = __import__("generate_answer").validate_answer

        def _validate_pass_candidate(*args, **kwargs):
            if str(kwargs.get("answer_text") or "").strip() == candidate.strip():
                return {"validation_status": "pass", "errors": [], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro), mock.patch(
                "generate_answer.validate_answer", side_effect=_validate_pass_candidate
            ):
                result = run_generation(
                    query="tu peux faire une note pour les differents plages qui exist dans les valeurs phisiologique dans report 12",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        self.assertEqual(str(result.get("final_answer_source") or ""), "llm_writer")
        self.assertIsNone(result.get("renderer_used"))
        self.assertIsNone(result.get("fallback_reason"))
        self.assertTrue(bool(result.get("llm_writer_accepted")))

    def test_level2_postprocess_exception_sets_explicit_fallback_reason(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": "Anormaux : test.\nRésultats dans la référence uniquement : aucun.\nConclusion technique : test.",
                "llm_candidate_answer": "Anormaux : test.",
                "llm_error": None,
                "use_micro_prompt": True,
            }

        def _raise_postprocess(_answer: str) -> bool:
            raise RuntimeError("forced postprocess failure")

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro), mock.patch(
                "generate_answer._contains_internal_reasoning_leak", side_effect=_raise_postprocess
            ):
                result = run_generation(
                    query="Fais une synthèse médico-biologique du report 12 en 6 lignes maximum, en séparant les anomalies et les résultats dans la référence uniquement. Ne donne pas de diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("fallback_reason") or ""), "llm_postprocess_exception")
        self.assertEqual(str(debug.get("llm_postprocess_error_type") or ""), "RuntimeError")
        self.assertEqual(str(debug.get("fallback_renderer_used") or ""), "deterministic_biological_summary_short")

    def test_level2_biological_summary_micro_prompt_budget(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"

        def _fake_generate(self, prompt: str, **kwargs):  # type: ignore[no-untyped-def]
            self.last_call_debug = {"prompt_chars": len(prompt)}
            return "Anormaux : test.\nRésultats dans la référence uniquement : aucun.\nConclusion technique : test."

        try:
            with mock.patch("generate_answer.LLMClient.generate", new=_fake_generate):
                result = run_generation(
                    query="Fais une synthèse médico-biologique du report 12 en 6 lignes maximum, en séparant les anomalies et les résultats dans la référence uniquement. Ne donne pas de diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        debug = dict(result.get("debug") or {})
        self.assertTrue(bool(debug.get("use_micro_prompt")))
        self.assertLessEqual(int(debug.get("prompt_chars") or 0), 3500)

    def test_priority_llm_enforces_backend_sections_and_no_collapsed_reference(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"

        def _fake_generate(self, prompt: str, **kwargs):  # type: ignore[no-untyped-def]
            self.last_call_debug = {"prompt_chars": len(prompt)}
            return (
                "Priorité élevée : Albumine 8 g/l.\n"
                "Priorité modérée/faible : Triglycérides 8 g/l, référence 1,50 - 1,50 ; ...\n"
                "Conclusion technique : ..."
            )

        try:
            with mock.patch("generate_answer.LLMClient.generate", new=_fake_generate):
                result = run_generation(
                    query="Dans le report 10, explique les anomalies biologiques les plus importantes par priorité technique, avec une justification courte pour chaque anomalie. Ne pose pas de diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        answer = str(result.get("answer") or "")
        answer_n = answer.lower()
        self.assertNotIn("...", answer)
        self.assertNotIn("1,50 - 1,50", answer)
        self.assertIn("trigly", answer_n)
        self.assertTrue(("priorité élevée" in answer_n) or ("| high |" in answer_n))
        self.assertRegex(answer_n, r"(trigly[^\n]*\|\s*high\s*\|)|(\|\s*high\s*\|[^\n]*trigly)")
        self.assertNotEqual(str(result.get("validation", {}).get("validation_status") or ""), "fail")

    def test_priority_postprocess_exception_is_controlled(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"

        def _fake_generate(self, prompt: str, **kwargs):  # type: ignore[no-untyped-def]
            self.last_call_debug = {"prompt_chars": len(prompt)}
            return "Priorité élevée : Albumine 8 g/l.\nPriorité modérée/faible : CRP 7 mg/l.\nConclusion technique : test."

        try:
            with mock.patch("generate_answer.LLMClient.generate", new=_fake_generate), mock.patch(
                "generate_answer._priority_answer_needs_enforcement", side_effect=NameError("_canonical_analyte_key is not defined")
            ):
                result = run_generation(
                    query="Dans le report 10, explique les anomalies biologiques les plus importantes par priorité technique, avec une justification courte pour chaque anomalie. Ne pose pas de diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        debug = dict(result.get("debug") or {})
        self.assertNotEqual(str(debug.get("fallback_reason") or ""), "_canonical_analyte_key is not defined")
        self.assertIn(str(debug.get("fallback_reason") or ""), {"priority_postprocess_exception", "llm_validation_failed", "llm_writer_quality_or_format_fallback"})
        self.assertNotIn("_canonical_analyte_key is not defined", str(debug.get("llm_raw_error_message") or ""))

    def test_biological_summary_replaces_weak_conclusion(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": (
                    "Anormaux : CRP (au-dessus), Réserve Alcaline (en dessous).\n"
                    "Résultats dans la référence uniquement : ACIDE URIQUE 40 mg/l.\n"
                    "Conclusion technique : Le nombre de résultats anormaux est de 2."
                ),
                "llm_candidate_answer": "x",
                "llm_error": None,
                "use_micro_prompt": True,
            }

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro):
                result = run_generation(
                    query="Résume le report 24 comme une note courte pour un médecin, en restant strictement descriptif et sans diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        answer = str(result.get("answer") or "")
        self.assertNotIn("Le nombre de résultats anormaux est de", answer)
        self.assertIn(
            "Conclusion technique : synthèse descriptive limitée aux données disponibles, sans diagnostic.",
            answer,
        )

    def test_final_postprocess_fixed_warnings_tracks_missing_conclusion_fix(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        candidate = "Faits techniques observés : TSHus haute, T4 libre haute, T3 libre haute."

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": candidate,
                "llm_candidate_answer": candidate,
                "llm_error": None,
                "use_micro_prompt": True,
            }

        original_validate = __import__("generate_answer").validate_answer

        def _fake_validate(*args, **kwargs):
            answer_text = str(kwargs.get("answer_text") or "")
            if answer_text.strip() == candidate.strip():
                return {"validation_status": "warning", "errors": [], "warnings": ["missing_conclusion"]}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro), mock.patch(
                "generate_answer.validate_answer", side_effect=_fake_validate
            ):
                result = run_generation(
                    query="Le bilan thyroïdien du report 16 est-il compatible avec une hyperthyroïdie primaire ? Explique prudemment à partir de TSH, T3, T4 et anticorps, sans conclure à un diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        debug = dict(result.get("debug") or {})
        self.assertIn("missing_conclusion", list(debug.get("llm_candidate_validation_warnings") or []))
        self.assertIn("missing_conclusion", list(debug.get("final_postprocess_fixed_warnings") or []))
        self.assertNotIn("missing_conclusion", list(((result.get("validation") or {}).get("warnings")) or []))

    def test_thyroid_guarded_deduplicates_discordance_and_clinical_recommendations(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"

        def _fake_micro(**kwargs: object) -> dict[str, object]:
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": (
                    "Faits techniques observés : TSHus haute, T4 libre haute, T3 libre haute.\n"
                    "Ce profil est biologiquement discordant pour une hyperthyroïdie primaire.\n"
                    "Il est essentiel de réaliser des examens complémentaires.\n"
                    "Ce profil est biologiquement discordant pour une hyperthyroïdie primaire."
                ),
                "llm_candidate_answer": "x",
                "llm_error": None,
                "use_micro_prompt": True,
            }

        try:
            with mock.patch("generate_answer._compose_level2_micro_prompt_answer", side_effect=_fake_micro):
                result = run_generation(
                    query="Le bilan thyroïdien du report 16 est-il compatible avec une hyperthyroïdie primaire ? Explique prudemment à partir de TSH, T3, T4 et anticorps, sans conclure à un diagnostic.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force

        answer = str(result.get("answer") or "")
        self.assertNotIn("examens complémentaires", answer.lower())
        self.assertLessEqual(answer.lower().count("discordant pour une hyperthyroïdie primaire"), 1)
        self.assertIn("Conclusion technique :", answer)

    def test_hard_gate_value_changed_forces_deterministic_fallback(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        original_validate = __import__("generate_answer").validate_answer

        call_count = {"n": 0}

        def _fake_validate(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"validation_status": "fail", "errors": ["value_changed"], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer.validate_answer", side_effect=_fake_validate):
                result = run_generation(
                    query="Résume le report 12 en quelques lignes, avec une partie anomalies et une partie résultats normaux, sans conclusion diagnostique.",
                    mode="keyword",
                    top_k=30,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        self.assertTrue(str(result.get("generation_mode") or "").startswith("deterministic_"))
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
        dbg = dict(result.get("debug") or {})
        self.assertTrue(bool(dbg.get("hard_gate_triggered")))
        self.assertIn("value_changed", list(dbg.get("hard_gate_errors") or []))

    def test_hard_gate_unit_mismatch(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        original_validate = __import__("generate_answer").validate_answer

        call_count = {"n": 0}

        def _fake_validate(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"validation_status": "fail", "errors": ["unit_mismatch"], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer.validate_answer", side_effect=_fake_validate):
                result = run_generation(
                    query="Résume uniquement les anomalies biologiques du report (16), sans poser de diagnostic.",
                    mode="keyword",
                    top_k=20,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        self.assertTrue(str(result.get("generation_mode") or "").startswith("deterministic_"))
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})
        dbg = dict(result.get("debug") or {})
        self.assertIn("unit_mismatch", list(dbg.get("hard_gate_errors") or []))

    def test_hard_gate_internal_fields_visible(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        original_validate = __import__("generate_answer").validate_answer

        call_count = {"n": 0}

        def _fake_validate(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"validation_status": "fail", "errors": ["forbidden_internal_field", "chunk_id_visible"], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer.validate_answer", side_effect=_fake_validate):
                result = run_generation(
                    query="Résume uniquement les anomalies biologiques du report (16), sans poser de diagnostic.",
                    mode="keyword",
                    top_k=20,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        self.assertNotIn("chunk_id=", str(result.get("answer") or "").lower())
        self.assertTrue(str(result.get("generation_mode") or "").startswith("deterministic_"))

    def test_hard_gate_source_mismatch(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        original_validate = __import__("generate_answer").validate_answer
        call_count = {"n": 0}

        def _fake_validate(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"validation_status": "fail", "errors": ["source_mismatch"], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer.validate_answer", side_effect=_fake_validate):
                result = run_generation(
                    query="Résume uniquement les anomalies biologiques du report (16), sans poser de diagnostic.",
                    mode="keyword",
                    top_k=20,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        self.assertTrue(str(result.get("generation_mode") or "").startswith("deterministic_"))
        dbg = dict(result.get("debug") or {})
        self.assertIn("source_mismatch", list(dbg.get("hard_gate_errors") or []))

    def test_hard_gate_raw_internal_source(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        original_validate = __import__("generate_answer").validate_answer
        call_count = {"n": 0}

        def _fake_validate(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"validation_status": "fail", "errors": ["raw_internal_source"], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer.validate_answer", side_effect=_fake_validate):
                result = run_generation(
                    query="Résume uniquement les anomalies biologiques du report (16), sans poser de diagnostic.",
                    mode="keyword",
                    top_k=20,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        self.assertTrue(str(result.get("generation_mode") or "").startswith("deterministic_"))
        dbg = dict(result.get("debug") or {})
        self.assertIn("raw_internal_source", list(dbg.get("hard_gate_errors") or []))

    def test_hard_gate_pii_exposure(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        original_validate = __import__("generate_answer").validate_answer
        call_count = {"n": 0}

        def _fake_validate(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"validation_status": "fail", "errors": ["pii_exposure"], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer.validate_answer", side_effect=_fake_validate):
                result = run_generation(
                    query="Résume uniquement les anomalies biologiques du report (16), sans poser de diagnostic.",
                    mode="keyword",
                    top_k=20,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        self.assertTrue(str(result.get("generation_mode") or "").startswith("deterministic_"))
        dbg = dict(result.get("debug") or {})
        self.assertIn("pii_exposure", list(dbg.get("hard_gate_errors") or []))

    def test_small_talk_plus_medical_not_general_conversation(self) -> None:
        result = run_generation(
            query="Bonjour, peux-tu résumer le report 16 ?",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertNotEqual(str((result.get("debug") or {}).get("selected_route") or ""), "general_conversation")
        self.assertIn("report_16", str(result.get("answer") or "").lower())

    def test_treatment_recommendation_hard_gate_refusal(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        original_validate = __import__("generate_answer").validate_answer
        call_count = {"n": 0}

        def _fake_validate(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"validation_status": "fail", "errors": ["treatment_recommendation"], "warnings": []}
            return original_validate(*args, **kwargs)

        try:
            with mock.patch("generate_answer.validate_answer", side_effect=_fake_validate):
                result = run_generation(
                    query="Quel traitement recommandes-tu pour les anomalies du report 16 ?",
                    mode="keyword",
                    top_k=20,
                    index_dir="data/indexes",
                )
        finally:
            if old_force is None:
                os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)
            else:
                os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = old_force
        answer = str(result.get("answer") or "")
        self.assertTrue(answer.startswith("Je ne peux pas recommander de traitement à partir de ces résultats seuls."))
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})

    def test_response_transform_validation_fail_forces_deterministic_fallback(self) -> None:
        first = run_generation(
            query="Dans report 19, compare l’insuline et la T4 libre avec leurs résultats antérieurs. Retourne la réponse sous forme de tableau.",
            mode="keyword",
            top_k=5,
            index_dir="data/indexes",
        )
        previous_pack = first.get("structured_evidence_pack") or {}
        self.assertTrue(previous_pack.get("evidences"))

        ga = __import__("generate_answer")
        original_validate = ga.validate_answer
        call_count = {"n": 0}

        def _fake_validate(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return {"validation_status": "fail", "errors": ["source_mismatch"], "warnings": []}
            return original_validate(*args, **kwargs)

        with mock.patch("generate_answer.validate_answer", side_effect=_fake_validate):
            transformed = run_generation(
                query="Convertis la réponse précédente en style paragraphe médical pro.",
                mode="keyword",
                top_k=5,
                index_dir="data/indexes",
                search_engine=_FailIfCalledSearchEngine(),
                previous_structured_evidence_pack=previous_pack,
            )

        self.assertEqual(str(transformed.get("generation_mode") or ""), "deterministic_response_transform_professional")
        self.assertGreaterEqual(call_count["n"], 2)
        dbg = dict(transformed.get("debug") or {})
        self.assertEqual(str(dbg.get("fallback_reason") or ""), "llm_validation_failed")

    def test_abnormal_results_without_scope_uses_deterministic_clarification(self) -> None:
        result = run_generation(
            query="les résultats anormaux",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_no_evidence_response")
        self.assertNotEqual(str(result.get("generation_mode") or ""), "llm")
        self.assertEqual(len(result.get("sources") or []), 0)
        answer = str(result.get("answer") or "").lower()
        self.assertIn("information insuffisante", answer)
        self.assertIn("précisez un rapport", answer)
        dbg = dict(result.get("debug") or {})
        self.assertEqual(str(dbg.get("selected_route") or ""), "cohort_search")
        self.assertEqual(str(dbg.get("route_reason") or ""), "abnormal_results_without_scope_requires_clarification")
        self.assertEqual(str(dbg.get("answerability_status") or ""), "ambiguous")

    def test_abnormal_variants_without_scope_use_same_deterministic_clarification(self) -> None:
        for q in ["anomalies ?", "hors norme ?"]:
            with self.subTest(query=q):
                result = run_generation(
                    query=q,
                    mode="keyword",
                    top_k=20,
                    index_dir="data/indexes",
                )
                self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_no_evidence_response")
                self.assertEqual(len(result.get("sources") or []), 0)
                dbg = dict(result.get("debug") or {})
                self.assertEqual(str(dbg.get("selected_route") or ""), "cohort_search")
                self.assertEqual(str(dbg.get("route_reason") or ""), "abnormal_results_without_scope_requires_clarification")

    def test_factual_sources_block_is_appended_even_without_displayed_rows(self) -> None:
        ga = __import__("generate_answer")
        answer, sources = ga._ensure_sources_in_factual_answer(
            answer="Comparaison technique disponible.",
            generation_mode="deterministic_multi_doc_comparison",
            selected_route="multi_doc_comparison",
            displayed_evidences=[],
            source_citations=[
                {
                    "label": "report (10).pdf — page 1, ligne 2",
                    "doc_id": "report_10",
                    "url": "/viewer/pdf?doc_id=report_10&page=1",
                }
            ],
        )
        self.assertIn("Sources :", answer)
        self.assertGreater(len(list(sources or [])), 0)

    def test_multi_doc_comparison_without_rows_has_doc_scope_sources(self) -> None:
        result = run_generation(
            query="compare report 10 et 12 vite fait",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_multi_doc_comparison")
        sources = list(result.get("sources") or [])
        self.assertGreater(len(sources), 0)
        source_doc_ids = {str(s.get("doc_id") or "").strip().lower() for s in sources}
        self.assertIn("report_10", source_doc_ids)
        self.assertIn("report_12", source_doc_ids)

    def test_multi_doc_comparison_never_uses_placeholder_report_labels(self) -> None:
        result = run_generation(
            query="compare report 10 et 12 vite fait",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        answer = str(result.get("answer") or "")
        self.assertNotIn("report_a", answer)
        self.assertNotIn("report_b", answer)

    def test_global_abnormal_with_analyte_not_forced_to_scope_clarification(self) -> None:
        result = run_generation(
            query="quels sont les rapports qui ont créatinine supérieur a 2 ??",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertNotEqual(str(result.get("generation_mode") or ""), "deterministic_no_evidence_response")
        self.assertIn(str((result.get("debug") or {}).get("selected_route") or ""), {"cohort_search", "global_analyte_abnormal_search"})

    def test_multi_doc_reference_ranges_query_forces_reference_ranges_summary_route(self) -> None:
        result = run_generation(
            query="fais une note des valeurs physiologiques dans report 12 et report 24",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "reference_ranges_summary")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_reference_ranges_summary")

    def test_reference_range_lookup_validator_accepts_multi_profile_with_viewer_url(self) -> None:
        av = __import__("answer_validator")
        answer_text = (
            "Pour AMH, plusieurs sous-profils valides existent pour la demande.\n"
            "Sous-profils disponibles :\n"
            "- Homme — âge non précisé : 4,35–5,35 ng/ml\n"
            "- Femme — 20-24 ans : 3,55–4,33 ng/ml\n"
            "- Femme — 25-29 ans : 3,03–3,87 ng/ml\n"
            "Documents contenant cet analyte : [report 1](/viewer/pdf?doc_id=report_1&page=1)."
        )
        validation = av.validate_answer(
            query="dans le meme document donne moi la plage de AMH",
            answer_text=answer_text,
            evidence_pack=[],
            displayed_evidences=[],
            source_citations=[{"doc_id": "report_1", "label": "report (1).pdf — page 1, ligne 4", "url": "/viewer/pdf?doc_id=report_1&page=1"}],
            generation_mode="deterministic_reference_range_lookup",
            retrieval_status="answerable",
            query_intents={"reference_range_lookup": True},
        )
        errors = {str(e).strip() for e in list(validation.get("errors") or [])}
        self.assertNotIn("reference_range_missing_main_fact", errors)
        self.assertNotIn("reference_range_internal_source_leak", errors)

    def test_response_transform_pack_includes_rows_when_evidences_missing(self) -> None:
        ga = __import__("generate_answer")
        qu_mod = __import__("query_understanding")
        qu = qu_mod.parse_query_understanding("mets ça en tableau")
        previous_pack = {
            "intent": "doc_scoped_summary",
            "requested_doc_ids": ["report_12", "report_24"],
            "evidences": [],
            "rows": [
                {"doc_id": "report_12", "analyte": "Albumine", "value_raw": "40", "unit": "g/l", "reference_range": "35 à 50 g/l"},
                {"doc_id": "report_24", "analyte": "APO A1", "value_raw": "2.3", "unit": "g/l", "reference_range": "1,1 - 1,6"},
            ],
        }
        transformed = ga._build_response_transform_pack(
            query="mets ça en tableau",
            query_understanding=qu,
            previous_pack=previous_pack,
        )
        docs = {str(r.get("doc_id") or "").strip().lower() for r in list(transformed.get("results") or [])}
        self.assertIn("report_12", docs)
        self.assertIn("report_24", docs)

    def test_response_transform_blocks_value_changed_before_final_render(self) -> None:
        first = run_generation(
            query="fais une note des valeurs physiologiques dans report 12 et report 24",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        previous_pack = first.get("structured_evidence_pack") or {}
        self.assertTrue(bool(previous_pack))

        def _always_fail_validate(*args, **kwargs):
            return {"validation_status": "fail", "errors": ["value_changed"], "warnings": []}

        with mock.patch("generate_answer.validate_answer", side_effect=_always_fail_validate):
            transformed = run_generation(
                query="Convertis la réponse précédente en style paragraphe médical pro.",
                mode="keyword",
                top_k=5,
                index_dir="data/indexes",
                search_engine=_FailIfCalledSearchEngine(),
                previous_structured_evidence_pack=previous_pack,
            )
        answer = str(transformed.get("answer") or "").lower()
        self.assertIn("transformation demandée est bloquée", answer)
        self.assertEqual(str((transformed.get("validation") or {}).get("validation_status") or ""), "fail")
        self.assertIn("transform_blocked_value_changed", list((transformed.get("validation") or {}).get("errors") or []))


if __name__ == "__main__":
    unittest.main()
