from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
GENERATION_ROOT = SCRIPTS_ROOT / "generation"
for root in (SCRIPTS_ROOT, GENERATION_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from answer_validator import validate_answer
from professional_answer_composer import (
    choose_presentation_format,
    compose_professional_answer,
    deduplicate_sources,
    format_result_count,
    format_source_label,
    render_professional_fallback,
)
from query_understanding import parse_query_understanding


class TestProfessionalAnswerComposer(unittest.TestCase):
    def _cohort_pack(self, n: int = 1) -> dict:
        rows = []
        for i in range(1, n + 1):
            rows.append(
                {
                    "doc_id": "report_16",
                    "patient_token": f"PAT_00000{i}",
                    "page": 1,
                    "row": i,
                    "analyte": "TSHus",
                    "analyte_norm": "tshus",
                    "current_value": "55,00",
                    "unit": "mUI/L",
                    "reference": "0,35 à 4,94 mUI/l",
                    "previous_result": "",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                    "variation": "non comparable",
                }
            )
        return {
            "question": "Quels patients ont une TSHus au-dessus de la référence ?",
            "intent": "cohort_search",
            "requested_doc_ids": [],
            "requested_analytes": ["tshus", "tsh"],
            "technical_condition": "above_reference",
            "output_format": "table",
            "answer_style": "standard",
            "requested_table_columns": [],
            "evidences": rows,
            "missing_items": [],
        }

    def _sources(self, rows: list[int]) -> list[dict]:
        return [
            {
                "doc_id": "report_16",
                "filename": "report (16).pdf",
                "page": 1,
                "row": r,
                "label": "report (16).pdf — page 1",
                "url": "/api/documents/report_16/pdf?page=1",
                "viewer_url": "/viewer/pdf?doc_id=report_16&page=1",
            }
            for r in rows
        ]

    def test_01_no_internal_aliases_in_intro(self) -> None:
        qu = parse_query_understanding(
            "Quels patients ont une TSHus au-dessus de la référence ? Retourne uniquement les résultats TSHus."
        )
        composed = render_professional_fallback(
            evidence_pack=self._cohort_pack(1),
            query_understanding=qu,
            user_question=qu.requested_analytes[0],
            source_citations=self._sources([1]),
        )
        answer = str(composed.get("answer") or "")
        self.assertIn("TSHus", answer)
        self.assertNotIn("tshus, tsh", answer.lower())
        self.assertNotIn("TRAK", answer)
        self.assertIn("Un seul résultat correspondant a été retrouvé.", answer)

    def test_02_singular_result_count(self) -> None:
        self.assertEqual(format_result_count(1), "Un seul résultat correspondant a été retrouvé.")
        self.assertNotIn("résultat(s)", format_result_count(1))

    def test_03_plural_result_count(self) -> None:
        self.assertEqual(format_result_count(3), "3 résultats correspondants ont été retrouvés.")
        self.assertNotIn("résultat(s)", format_result_count(3))

    def test_04_source_label_is_clean(self) -> None:
        src = {
            "doc_id": "report_16",
            "filename": "report (16).pdf",
            "page": 1,
            "row": 1,
            "url": "/api/documents/report_16/pdf?page=1",
        }
        label = format_source_label(src)
        self.assertIn("page 1, ligne 1", label)
        self.assertNotIn("page 1row 1", label)

    def test_05_sources_are_grouped(self) -> None:
        grouped = deduplicate_sources(self._sources([1, 2, 3, 4, 5, 6]))
        self.assertEqual(len(grouped), 1)
        self.assertIn("lignes 1–6", str(grouped[0].get("label") or ""))

    def test_06_no_repeated_cold_conclusion(self) -> None:
        qu = parse_query_understanding("Dans report 16, donne les résultats TSHus sous forme tableau.")
        pack = self._cohort_pack(1)
        pack["intent"] = "doc_scoped_results"
        pack["requested_doc_ids"] = ["report_16"]
        composed = render_professional_fallback(
            evidence_pack=pack,
            query_understanding=qu,
            user_question="Dans report 16, donne les résultats TSHus sous forme tableau.",
            source_citations=self._sources([1]),
        )
        answer = str(composed.get("answer") or "")
        self.assertNotIn("Les résultats ci-dessus sont strictement extraits des données indexées.", answer)

    def test_07_auto_table_for_homogeneous_results(self) -> None:
        qu = parse_query_understanding("Dans report 16, liste les analytes demandés.")
        pack = self._cohort_pack(5)
        pack["intent"] = "doc_scoped_results"
        pack["requested_doc_ids"] = ["report_16"]
        fmt = choose_presentation_format(qu, pack)
        self.assertEqual(fmt, "table")
        composed = render_professional_fallback(
            evidence_pack=pack,
            query_understanding=qu,
            user_question="Dans report 16, liste les analytes demandés.",
            source_citations=self._sources([1, 2, 3, 4, 5]),
        )
        answer = str(composed.get("answer") or "")
        self.assertIn("| Analyte |", answer)

    def test_08_yes_no_is_strict(self) -> None:
        qu = parse_query_understanding(
            "Dans report 16, est-ce que l’ACTH est hors référence ? Réponds uniquement yes ou no, avec la valeur, la référence et la source."
        )
        pack = {
            "question": "Q",
            "intent": "doc_scoped_results",
            "requested_doc_ids": ["report_16"],
            "requested_analytes": ["acth"],
            "output_format": "yes_no",
            "answer_style": "yes_no",
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
                }
            ],
            "missing_items": [],
        }
        composed = render_professional_fallback(
            evidence_pack=pack,
            query_understanding=qu,
            user_question="Q",
            source_citations=self._sources([1]),
        )
        answer = str(composed.get("answer") or "").strip().lower()
        self.assertTrue(answer.startswith("non") or answer.startswith("no"))
        self.assertNotIn("| --- |", answer)

    def test_09_json_strict_only(self) -> None:
        qu = parse_query_understanding("Convertis la réponse précédente en JSON strict.")
        pack = self._cohort_pack(1)
        pack["intent"] = "response_transform"
        pack["output_format"] = "json"
        composed = compose_professional_answer(
            user_question="Convertis la réponse précédente en JSON strict.",
            query_understanding=qu,
            evidence_pack=pack,
            mode="fallback",
            source_citations=self._sources([1]),
        )
        answer = str(composed.get("answer") or "").strip()
        self.assertTrue(answer.startswith("{") and answer.endswith("}"))
        parsed = json.loads(answer)
        self.assertIn("results", parsed)
        self.assertNotIn("Sources :", answer)

    def test_10_no_hallucination_against_validator(self) -> None:
        qu = parse_query_understanding("Dans report 16, donne la valeur ACTH.")
        pack = {
            "question": "Dans report 16, donne la valeur ACTH.",
            "intent": "doc_scoped_results",
            "requested_doc_ids": ["report_16"],
            "requested_analytes": ["acth"],
            "output_format": "table",
            "answer_style": "standard",
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
                    "page": 1,
                    "row": 1,
                }
            ],
            "missing_items": [],
        }
        composed = render_professional_fallback(
            evidence_pack=pack,
            query_understanding=qu,
            user_question=pack["question"],
            source_citations=self._sources([1]),
        )
        answer = str(composed.get("answer") or "")
        displayed = [
            {
                "doc_id": "report_16",
                "chunk_id": "test_chunk_1",
                "analyte_norm": "acth",
                "analyte": "ACTH",
                "value_raw": "23,00",
                "reference_range": "4,70 - 48,80 pg/ml",
                "unit": "pg/ml",
                "previous_result": "",
                "patient_token": "PAT_000001",
                "interpretation_status": "within_reference",
                "page_number": 1,
                "row_index": 1,
                "text_excerpt": "ACTH 23,00 pg/ml",
            }
        ]
        validation = validate_answer(
            query=pack["question"],
            answer_text=answer,
            evidence_pack=displayed,
            displayed_evidences=displayed,
            source_citations=self._sources([1]),
            generation_mode="deterministic_professional_fallback",
            output_format_requested="table",
            answer_style_requested="standard",
            query_intents={"is_structured_query": True, "doc_scoped_results": True},
            requested_analytes=["acth"],
            found_requested_analytes=["acth"],
            found_requested_analyte_norms=["acth"],
            missing_requested_analytes=[],
        )
        self.assertNotIn("unsupported_value", validation.get("errors") or [])
        self.assertNotIn("unsupported_analyte", validation.get("errors") or [])
        self.assertNotIn("forbidden_internal_field", validation.get("errors") or [])


if __name__ == "__main__":
    unittest.main()
