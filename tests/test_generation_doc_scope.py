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
    def test_general_conversation_bonjour_fast_path_no_retrieval(self) -> None:
        result = run_generation(
            query="Bonjour.",
            mode="keyword",
            top_k=20,
            index_dir="data/indexes",
        )
        self.assertEqual(str((result.get("debug") or {}).get("selected_route") or ""), "general_conversation")
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_general_conversation")
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")

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
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
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
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
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
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
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
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
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
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
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
        try:
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
        self.assertEqual(str(result.get("generation_mode") or ""), "deterministic_doc_scoped_biological_summary")
        answer = str(result.get("answer") or "").lower()
        self.assertIn("anorm", answer)
        self.assertIn("normaux", str(result.get("answer") or "").lower())
        self.assertNotIn("anormaux : aucune anomalie objectivée", answer)
        self.assertIn(str(result.get("validation", {}).get("validation_status") or ""), {"pass", "warning"})
        stage = dict((result.get("debug") or {}).get("stage_timings_ms") or {})
        self.assertIn(stage.get("llm_writer_ms"), {0, 0.0})

    def test_report24_short_note_summary_keeps_abnormal_when_present(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"
        try:
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
        self.assertEqual(str(debug.get("policy_level") or ""), "deterministic_preferred")
        self.assertEqual(str(debug.get("generation_strategy") or ""), "deterministic_preferred")
        self.assertIn(debug.get("llm_expected"), {False, 0})
        self.assertGreater(int(debug.get("llm_prompt_tokens_estimate") or 0), 0)
        self.assertLessEqual(int(debug.get("llm_evidence_rows_count") or 0), 6)
        self.assertTrue(bool(debug.get("use_micro_prompt")))
        self.assertLessEqual(int(debug.get("prompt_hard_limit_chars") or 0), 3500)

    def test_biological_summary_is_deterministic_preferred_by_default(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        if "MEDICAL_RAG_FORCE_LLM_WRITER" in os.environ:
            os.environ.pop("MEDICAL_RAG_FORCE_LLM_WRITER", None)

        def _feature_enabled(name: str, default: bool = False) -> bool:
            if str(name) == "LLM_SUMMARY_WRITER_ENABLED":
                return False
            if str(name) in {"LLM_GLOBAL_ENABLED", "LLM_REWRITE_ENABLED", "LLM_FALLBACK_NON_CRITICAL_ONLY"}:
                return True
            return default
        try:
            with mock.patch("generate_answer._is_feature_enabled", side_effect=_feature_enabled):
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
        self.assertEqual(str(debug.get("generation_strategy") or ""), "deterministic_preferred")
        self.assertIn(debug.get("llm_expected"), {False, 0})
        self.assertIn(debug.get("llm_writer_attempted"), {False, 0})
        self.assertEqual(str(debug.get("llm_skipped_reason") or ""), "biological_summary_deterministic_preferred")
        self.assertEqual(str(result.get("validation", {}).get("validation_status") or ""), "pass")
        self.assertTrue(str(result.get("generation_mode") or "").startswith("deterministic_"))

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
                "technical_status_code": "above_reference",
            },
            {
                "doc_id": "report_24",
                "page": 1,
                "row": 2,
                "analyte": "Phosphore",
                "analyte_label": "Phosphore",
                "display_name": "Phosphore",
                "source_analyte": "Phosphore",
                "technical_status_code": "within_reference",
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


if __name__ == "__main__":
    unittest.main()
