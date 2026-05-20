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
        self.assertEqual(len(result.get("sources") or []), 0)
        self.assertEqual(len(result.get("displayed_evidences") or []), 0)
        stage = dict((result.get("debug") or {}).get("stage_timings_ms") or {})
        self.assertIn(stage.get("llm_writer_ms"), {0, 0.0})
        self.assertIn(stage.get("retrieval_ms"), {0, 0.0})
        answer = str(result.get("answer") or "").lower()
        self.assertNotIn("report_", answer)
        self.assertNotIn("tsh", answer)

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

        def _fake_compose(**kwargs: object) -> dict[str, object]:
            captured["mode"] = kwargs.get("mode")
            captured["timeout"] = kwargs.get("timeout")
            captured["max_tokens"] = kwargs.get("max_tokens")
            return {
                "mode": "hybrid_structured_llm_writer",
                "answer": "Anormaux : test. Normaux / rassurants : test.",
                "llm_error": None,
            }

        try:
            with mock.patch("generate_answer.compose_professional_answer", side_effect=_fake_compose):
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

        self.assertEqual(captured.get("mode"), "hybrid_structured_llm_writer")
        self.assertEqual(int(captured.get("timeout") or 0), 60)
        self.assertEqual(int(captured.get("max_tokens") or 0), 220)
        debug = dict(result.get("debug") or {})
        self.assertEqual(str(debug.get("policy_level") or ""), "hybrid_controlled")

    def test_level2_timeout_falls_back_with_llm_timeout_reason(self) -> None:
        old_force = os.environ.get("MEDICAL_RAG_FORCE_LLM_WRITER")
        os.environ["MEDICAL_RAG_FORCE_LLM_WRITER"] = "1"

        def _fake_compose(**kwargs: object) -> dict[str, object]:
            mode = str(kwargs.get("mode") or "")
            if mode == "fallback":
                return {"mode": "deterministic_doc_scoped_biological_summary", "answer": "Fallback déterministe."}
            return {"mode": "llm_writer_error_fallback", "answer": "", "llm_error": "Ollama timeout"}

        try:
            with mock.patch("generate_answer.compose_professional_answer", side_effect=_fake_compose):
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


if __name__ == "__main__":
    unittest.main()
