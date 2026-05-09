from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from fastapi import HTTPException
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
GENERATION_ROOT = SCRIPTS_ROOT / "generation"
for root in (SCRIPTS_ROOT, GENERATION_ROOT, PROJECT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from backend_api import get_pdf
from citation_builder import build_source_citations
from generate_answer import run_generation
from retrieval.models import RetrievalResult, SearchResponse


class _FakeSearchEngine:
    def __init__(self, response: SearchResponse) -> None:
        self._response = response

    def search(self, **_: object) -> SearchResponse:
        return self._response

    def close(self) -> None:
        return None


class _FakeLLMClient:
    def generate(self, **_: object) -> str:
        return "Réponse : Résultat confirmé à partir des évidences."


def _mk_result(*, chunk_id: str, doc_id: str, value_raw: str = "4,90") -> RetrievalResult:
    metadata = {
        "analyte": "INSULINE",
        "analyte_norm": "insuline",
        "value_raw": value_raw,
        "value_numeric": value_raw.replace(",", "."),
        "unit": "uU/mL",
        "reference_range": "4 à 20 µIU/mL",
        "interpretation_status": "within_reference",
        "previous_result_present": 1,
        "previous_result_value_raw": "2,00",
        "source_kind": "chu_text_fallback",
        "row_index": 2,
        "page_number": 1,
        "source_pdf": f"docs/report ({doc_id.split('_')[-1]}).pdf",
    }
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_type="lab_result",
        document_type="lab_report",
        source_pdf=metadata["source_pdf"],
        page_number=1,
        text=f"INSULINE {value_raw}",
        text_preview=f"INSULINE {value_raw}",
        metadata=metadata,
        retrieval_mode="keyword",
        match_reason=["fake"],
    )


class TestSourceCitationsBackend(unittest.TestCase):
    def test_source_citation_builder_known_doc(self) -> None:
        evidences = [
            {
                "doc_id": "report_19",
                "page_number": 1,
                "row_index": 2,
                "source_pdf": "docs/report (19).pdf",
            }
        ]
        sources = build_source_citations(evidences)
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertEqual(source["doc_id"], "report_19")
        self.assertEqual(source["page"], 1)
        self.assertEqual(source["row"], 2)
        self.assertIn("report", (source.get("label") or "").lower())
        self.assertEqual(source["url"], "/api/documents/report_19/pdf?page=1")
        self.assertNotIn("/home/", json.dumps(source))

    def test_source_citation_builder_unknown_doc(self) -> None:
        evidences = [
            {
                "doc_id": "unknown_999",
                "page_number": 1,
                "row_index": 2,
            }
        ]
        sources = build_source_citations(evidences)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["doc_id"], "unknown_999")
        self.assertIsNone(sources[0]["url"])
        self.assertIsNone(sources[0]["viewer_url"])

    def test_path_traversal_doc_id_rejected(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            get_pdf("../../etc/passwd", page=1)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_generation_contains_structured_sources(self) -> None:
        keep = _mk_result(chunk_id="chk_report_19_insuline", doc_id="report_19")
        response = SearchResponse(
            query="x",
            mode="keyword",
            filters={},
            top_results=[keep],
            context_chunks=[keep],
            sources=[{"doc_id": "report_19", "chunk_id": "chk_report_19_insuline"}],
            answerability={"status": "answerable", "reason": "test"},
        )

        result = run_generation(
            query="Dans report 19, donne-moi la valeur de l'insuline.",
            mode="keyword",
            top_k=5,
            index_dir="data/indexes",
            search_engine=_FakeSearchEngine(response),
            llm_client=_FakeLLMClient(),
        )

        sources = result.get("sources") or []
        self.assertTrue(sources)
        first = sources[0]
        self.assertEqual(first.get("doc_id"), "report_19")
        self.assertIn("/api/documents/report_19/pdf", str(first.get("url") or ""))

    def test_normal_mode_sources_do_not_expose_chunk_id(self) -> None:
        keep = _mk_result(chunk_id="chk_report_19_insuline_very_long_identifier", doc_id="report_19")
        response = SearchResponse(
            query="x",
            mode="keyword",
            filters={},
            top_results=[keep],
            context_chunks=[keep],
            sources=[{"doc_id": "report_19", "chunk_id": "chk_report_19_insuline_very_long_identifier"}],
            answerability={"status": "answerable", "reason": "test"},
        )

        result = run_generation(
            query="Dans report 19, donne-moi la valeur de l'insuline.",
            mode="keyword",
            top_k=5,
            index_dir="data/indexes",
            search_engine=_FakeSearchEngine(response),
            llm_client=_FakeLLMClient(),
        )

        for src in result.get("sources") or []:
            self.assertNotIn("chunk_id=", str(src.get("label") or "").lower())
        self.assertNotIn("chunk_id=", str(result.get("answer") or "").lower())


if __name__ == "__main__":
    unittest.main()
