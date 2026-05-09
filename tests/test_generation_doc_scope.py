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


if __name__ == "__main__":
    unittest.main()
