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
from llm_client import LLMClientError
from professional_answer_composer import (
    choose_presentation_format,
    compose_professional_answer,
    deduplicate_sources,
    format_result_count,
    format_source_label,
    render_professional_fallback,
)
from query_understanding import parse_query_understanding


class _FakeLLMClient:
    def __init__(self, response: str, raise_error: bool = False) -> None:
        self.response = response
        self.raise_error = raise_error

    def generate(self, **_: object) -> str:
        if self.raise_error:
            raise LLMClientError("llm_unavailable")
        return self.response


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
        self.assertIn("Conclusion technique :", answer)

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

    def test_11_intro_mentions_numeric_criterion(self) -> None:
        qu = parse_query_understanding(
            "Liste-moi tous les patients qui ont ACTH avec une valeur de 23,00 ou plus. Retourne un tableau avec source cliquable."
        )
        pack = {
            "question": "Q",
            "intent": "cohort_search",
            "requested_doc_ids": [],
            "requested_analytes": ["acth"],
            "requested_value": "23,00",
            "technical_condition": None,
            "output_format": "table",
            "answer_style": "standard",
            "requested_table_columns": [],
            "evidences": [
                {
                    "doc_id": "report_16",
                    "patient_token": "PAT_000002",
                    "page": 1,
                    "row": 1,
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
            user_question=pack["question"],
            source_citations=self._sources([1]),
        )
        answer = str(composed.get("answer") or "").lower()
        self.assertIn("23,00", answer)
        self.assertTrue("supérieure ou égale" in answer or "ou plus" in answer)

    def test_12_clickable_source_column_when_requested(self) -> None:
        qu = parse_query_understanding(
            "Dans report 16, liste les résultats hors référence sous forme de tableau avec source cliquable."
        )
        pack = {
            "question": "Q",
            "intent": "doc_scoped_results",
            "requested_doc_ids": ["report_16"],
            "requested_analytes": ["tshus"],
            "output_format": "table",
            "answer_style": "standard",
            "requested_table_columns": [],
            "evidences": [
                {
                    "doc_id": "report_16",
                    "patient_token": "PAT_000002",
                    "page": 1,
                    "row": 4,
                    "analyte": "TSHus",
                    "analyte_norm": "tshus",
                    "current_value": "55,00",
                    "unit": "mUI/L",
                    "reference": "0,35 à 4,94 mUI/l",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                    "source_label": "report (16).pdf — page 1, ligne 4",
                    "source_url": "/api/documents/report_16/pdf?page=1",
                    "viewer_url": "/viewer/pdf?doc_id=report_16&page=1",
                }
            ],
            "missing_items": [],
        }
        composed = render_professional_fallback(
            evidence_pack=pack,
            query_understanding=qu,
            user_question=pack["question"],
            source_citations=self._sources([4]),
        )
        answer = str(composed.get("answer") or "")
        self.assertIn("| Source |", answer)
        self.assertIn("[report (16).pdf — page 1, ligne 4](/api/documents/report_16/pdf?page=1)", answer)

    def test_13_source_label_repairs_page_row_glue(self) -> None:
        src = {
            "doc_id": "report_16",
            "label": "report (16).pdf — page 1row 1",
            "page": 1,
            "row": 1,
        }
        label = format_source_label(src)
        self.assertEqual(label, "report (16).pdf — page 1, ligne 1")

    def test_14_professional_no_mechanical_placeholders(self) -> None:
        qu = parse_query_understanding(
            "Liste-moi tous les patients qui ont ACTH avec une valeur de 23,00 ou plus. Retourne un tableau avec source cliquable."
        )
        pack = {
            "question": "Q",
            "intent": "cohort_search",
            "requested_doc_ids": [],
            "requested_analytes": ["acth"],
            "requested_value": "23,00",
            "technical_condition": None,
            "output_format": "table",
            "answer_style": "standard",
            "requested_table_columns": [],
            "evidences": [
                {
                    "doc_id": "report_16",
                    "patient_token": "PAT_000002",
                    "page": 1,
                    "row": 1,
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
        answer = str(
            render_professional_fallback(
                evidence_pack=pack,
                query_understanding=qu,
                user_question=pack["question"],
                source_citations=self._sources([1]),
            ).get("answer")
            or ""
        )
        self.assertNotIn("résultat(s)", answer)
        self.assertNotIn("correspondant(s)", answer)
        self.assertNotIn("tshus, tsh", answer.lower())

    def test_15_llm_writer_path_and_fallback(self) -> None:
        qu = parse_query_understanding("Dans report 16, liste les résultats ACTH sous forme de tableau avec source cliquable.")
        pack = {
            "question": "Q",
            "intent": "doc_scoped_results",
            "requested_doc_ids": ["report_16"],
            "requested_analytes": ["acth"],
            "output_format": "table",
            "answer_style": "standard",
            "requested_table_columns": [],
            "evidences": [
                {
                    "doc_id": "report_16",
                    "patient_token": "PAT_000002",
                    "filename": "report (16).pdf",
                    "page": 1,
                    "row": 1,
                    "analyte": "ACTH",
                    "analyte_norm": "acth",
                    "current_value": "23,00",
                    "unit": "pg/ml",
                    "reference": "4,70 - 48,80 pg/ml",
                    "technical_status_code": "within_reference",
                    "technical_status": "dans la référence",
                    "source_label": "report (16).pdf — page 1, ligne 1",
                    "source_url": "/api/documents/report_16/pdf?page=1",
                    "viewer_url": "/viewer/pdf?doc_id=report_16&page=1",
                }
            ],
            "missing_items": [],
        }
        llm_ok = compose_professional_answer(
            user_question=pack["question"],
            query_understanding=qu,
            evidence_pack=pack,
            mode="auto",
            source_citations=self._sources([1]),
            llm_client=_FakeLLMClient(
                "J’ai recherché les résultats ACTH dans report_16.\n\n| Analyte | Valeur actuelle | Référence | Statut | Source |\n| --- | --- | --- | --- | --- |\n| ACTH | 23,00 pg/ml | 4,70 - 48,80 pg/ml | dans la référence | [report (16).pdf — page 1, ligne 1](/api/documents/report_16/pdf?page=1) |\n\nConclusion technique : le résultat retrouvé respecte le critère demandé."
            ),
        )
        self.assertEqual(str(llm_ok.get("mode") or ""), "llm_professional_writer")
        llm_fail = compose_professional_answer(
            user_question=pack["question"],
            query_understanding=qu,
            evidence_pack=pack,
            mode="auto",
            source_citations=self._sources([1]),
            llm_client=_FakeLLMClient("", raise_error=True),
        )
        self.assertEqual(str(llm_fail.get("mode") or ""), "llm_writer_error_fallback")
        self.assertIn("Conclusion technique :", str(llm_fail.get("answer") or ""))

    def test_16_safety_no_diagnosis(self) -> None:
        qu = parse_query_understanding(
            "Avec ACE, PSA TOTALE et CA 15-3, peut-on conclure à un cancer ?"
        )
        pack = {
            "question": "Q",
            "intent": "diagnostic_safety_question",
            "requested_doc_ids": ["report_31"],
            "requested_analytes": ["ace", "psa_totale", "ca_15_3"],
            "output_format": "list",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": "report_31",
                    "page": 1,
                    "row": 2,
                    "analyte": "ACE",
                    "analyte_norm": "ace",
                    "current_value": "8,00",
                    "unit": "ng/ml",
                    "reference": "0,00 - 5,00 ng/ml",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                }
            ],
            "missing_items": [],
        }
        answer = str(
            render_professional_fallback(
                evidence_pack=pack,
                query_understanding=qu,
                user_question=pack["question"],
                source_citations=[],
            ).get("answer")
            or ""
        ).lower()
        self.assertIn("on ne peut pas conclure", answer)
        self.assertIn("conclusion technique", answer)

    def test_17_validator_strict_json_violation(self) -> None:
        validation = validate_answer(
            query="Convertis la réponse précédente en JSON strict.",
            answer_text='Réponse:\n{"ok": true}',
            evidence_pack=[],
            displayed_evidences=[],
            source_citations=[],
            generation_mode="llm_professional_writer",
            output_format_requested="json",
            answer_style_requested="standard",
            query_intents={"is_structured_query": True},
        )
        self.assertIn("strict_json_violation", validation.get("errors") or [])

    def test_18_chart_request_not_silent_table(self) -> None:
        qu = parse_query_understanding("Dans report 16, liste les résultats hors référence sous forme Arithmetic Line-Graph.")
        pack = {
            "question": "Q",
            "intent": "doc_scoped_results",
            "requested_doc_ids": ["report_16"],
            "requested_analytes": ["tshus", "insuline"],
            "output_format": "chart",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": "report_16",
                    "patient_token": "PAT_000002",
                    "page": 1,
                    "row": 4,
                    "analyte": "TSHus",
                    "analyte_norm": "tshus",
                    "current_value": "55,00",
                    "unit": "mUI/L",
                    "reference": "0,35 à 4,94 mUI/l",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                },
                {
                    "doc_id": "report_16",
                    "patient_token": "PAT_000002",
                    "page": 1,
                    "row": 2,
                    "analyte": "INSULINE",
                    "analyte_norm": "insuline",
                    "current_value": "2,00",
                    "unit": "uU/mL",
                    "reference": "4 à 20 µIU/mL",
                    "technical_status_code": "below_reference",
                    "technical_status": "en dessous de la référence",
                },
            ],
            "missing_items": [],
        }
        composed = render_professional_fallback(
            evidence_pack=pack,
            query_understanding=qu,
            user_question=pack["question"],
            source_citations=self._sources([2, 4]),
        )
        answer = str(composed.get("answer") or "")
        self.assertIn("rendu graphique n’est pas encore disponible dans l’interface", answer)
        self.assertIn("données structurées", answer)
        self.assertIn("barres", answer.lower())
        self.assertIn("| Analyte |", answer)

    def test_19_unknown_format_no_silent_fallback(self) -> None:
        qu = parse_query_understanding(
            "Dans report 16, affiche les résultats hors référence sous forme bio-clinical matrix radar comparative."
        )
        pack = {
            "question": "Q",
            "intent": "doc_scoped_results",
            "requested_doc_ids": ["report_16"],
            "requested_analytes": ["tshus"],
            "output_format": "unknown",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": "report_16",
                    "patient_token": "PAT_000002",
                    "page": 1,
                    "row": 4,
                    "analyte": "TSHus",
                    "analyte_norm": "tshus",
                    "current_value": "55,00",
                    "unit": "mUI/L",
                    "reference": "0,35 à 4,94 mUI/l",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                }
            ],
            "missing_items": [],
        }
        composed = render_professional_fallback(
            evidence_pack=pack,
            query_understanding=qu,
            user_question=pack["question"],
            source_citations=self._sources([4]),
        )
        answer = str(composed.get("answer") or "").lower()
        self.assertIn("bio-clinical matrix radar comparative", answer)
        self.assertTrue(
            ("non support" in answer)
            or ("format alternatif" in answer)
            or ("composant graphique" in answer)
            or ("rendu graphique n’est pas encore disponible dans l’interface" in answer)
        )

    def test_20_source_grouping_lines_range(self) -> None:
        qu = parse_query_understanding("Dans report 16, liste les résultats hors référence sous forme de tableau avec source cliquable.")
        pack = {
            "question": "Q",
            "intent": "doc_scoped_results",
            "requested_doc_ids": ["report_16"],
            "requested_analytes": ["insuline", "t4_libre", "tshus", "t3_libre", "anti_tg"],
            "output_format": "table",
            "answer_style": "standard",
            "evidences": [],
            "missing_items": [],
        }
        sources = [
            {"doc_id": "report_16", "filename": "report (16).pdf", "page": 1, "row": 2, "url": "/api/documents/report_16/pdf?page=1"},
            {"doc_id": "report_16", "filename": "report (16).pdf", "page": 1, "row": 3, "url": "/api/documents/report_16/pdf?page=1"},
            {"doc_id": "report_16", "filename": "report (16).pdf", "page": 1, "row": 4, "url": "/api/documents/report_16/pdf?page=1"},
            {"doc_id": "report_16", "filename": "report (16).pdf", "page": 1, "row": 5, "url": "/api/documents/report_16/pdf?page=1"},
            {"doc_id": "report_16", "filename": "report (16).pdf", "page": 1, "row": 6, "url": "/api/documents/report_16/pdf?page=1"},
        ]
        composed = render_professional_fallback(
            evidence_pack=pack,
            query_understanding=qu,
            user_question=pack["question"],
            source_citations=sources,
        )
        answer = str(composed.get("answer") or "")
        self.assertIn("report (16).pdf — page 1, lignes 2–6", answer)
        self.assertNotIn("ligne 2ligne 2", answer)


if __name__ == "__main__":
    unittest.main()
