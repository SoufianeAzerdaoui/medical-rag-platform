from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
