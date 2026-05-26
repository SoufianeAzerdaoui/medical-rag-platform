from __future__ import annotations

import json
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

from answer_validator import validate_answer
from llm_client import LLMClientError
from professional_answer_composer import (
    LOGGER,
    PROFESSIONAL_WRITER_SYSTEM_PROMPT,
    build_writer_evidence_pack,
    choose_presentation_format,
    compose_professional_answer,
    deduplicate_sources,
    format_result_count,
    format_source_label,
    render_professional_fallback,
    _validate_writer_evidence_pack_contract,
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
        self.assertNotIn("Un seul résultat correspondant a été retrouvé.", answer)

    def test_02_singular_result_count(self) -> None:
        self.assertEqual(format_result_count(1), "Une valeur exploitable a été retrouvée.")
        self.assertNotIn("résultat(s)", format_result_count(1))

    def test_03_plural_result_count(self) -> None:
        self.assertEqual(format_result_count(3), "3 valeurs exploitables ont été retrouvées.")
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
            user_question="Dans report 16, est-ce que l’ACTH est hors référence ? Réponds uniquement yes ou no.",
            source_citations=self._sources([1]),
        )
        answer = str(composed.get("answer") or "").strip().lower()
        self.assertTrue(answer.startswith("non") or answer.startswith("no"))
        self.assertNotIn("| --- |", answer)

    def test_08b_no_yesno_prefix_when_not_explicit_yesno(self) -> None:
        qu = parse_query_understanding("Montre-moi ACTH du dernier rapport.")
        pack = {
            "question": "Montre-moi ACTH du dernier rapport.",
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
            user_question=pack["question"],
            source_citations=self._sources([1]),
        )
        answer = str(composed.get("answer") or "")
        self.assertNotIn("Non —", answer)
        self.assertIn("ACTH", answer)

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

    def test_14b_prompt_has_no_mechanical_contradiction(self) -> None:
        self.assertNotIn("N’écris jamais résultat", PROFESSIONAL_WRITER_SYSTEM_PROMPT)
        self.assertNotIn("Un seul résultat correspondant a été retrouvé.", PROFESSIONAL_WRITER_SYSTEM_PROMPT)
        self.assertIn("Évite les formulations mécaniques", PROFESSIONAL_WRITER_SYSTEM_PROMPT)

    def test_14c_prompt_enforces_no_selection_or_recalculation(self) -> None:
        self.assertIn("Tu ne sélectionnes jamais des lignes toi-même", PROFESSIONAL_WRITER_SYSTEM_PROMPT)
        self.assertIn("tu ne recalcules jamais une valeur", PROFESSIONAL_WRITER_SYSTEM_PROMPT.lower())
        self.assertIn("Tu reformules uniquement les lignes déjà fournies", PROFESSIONAL_WRITER_SYSTEM_PROMPT)

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

    def test_15b_clickable_requested_without_url_shows_text_notice(self) -> None:
        qu = parse_query_understanding(
            "Quelle est la norme AMH pour une femme de 30-34 ans ? avec source cliquable"
        )
        pack = {
            "question": "Q",
            "intent": "reference_range_lookup",
            "requested_doc_ids": ["report_1"],
            "requested_analytes": ["amh"],
            "output_format": "list",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": "report_1",
                    "filename": "report (1).pdf",
                    "page": 1,
                    "row": 4,
                    "analyte": "AMH",
                    "analyte_norm": "amh",
                    "current_value": "8",
                    "unit": "ng/ml",
                    "reference": "3,03-3,87 ng/ml",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                    "source_label": "report (1).pdf — page 1, ligne 4",
                }
            ],
            "missing_items": [],
        }
        composed = render_professional_fallback(
            evidence_pack=pack,
            query_understanding=qu,
            user_question=pack["question"],
            source_citations=[
                {
                    "doc_id": "report_1",
                    "filename": "report (1).pdf",
                    "page": 1,
                    "row": 4,
                    "label": "report (1).pdf — page 1, ligne 4",
                }
            ],
        )
        answer = str(composed.get("answer") or "")
        self.assertIn("Sources :", answer)
        self.assertIn("report (1).pdf — page 1, ligne 4", answer)
        self.assertIn("Source disponible uniquement en texte", answer)

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

    def test_17b_validator_structured_first_tolerates_summary_count_style(self) -> None:
        answer = (
            "Les anomalies techniques ci-dessous sont organisées par section du rapport.\n\n"
            "6 valeurs exploitables ont été retrouvées.\n\n"
            "| Analyte | Valeur actuelle | Référence | Statut | Document |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| INSULINE | 2,00 uU/mL | 4 à 20 µIU/mL | en dessous de la référence | report_16 |\n"
        )
        displayed = [
            {
                "doc_id": "report_16",
                "chunk_id": "test_chunk_insuline",
                "analyte_norm": "insuline",
                "analyte": "INSULINE",
                "value_raw": "2,00",
                "current_value": "2,00",
                "reference_range": "4 à 20 µIU/mL",
                "unit": "uU/mL",
                "interpretation_status": "below_reference",
                "page_number": 1,
                "row_index": 2,
                "text_excerpt": "INSULINE 2,00 uU/mL 4 à 20 µIU/mL",
            }
        ]
        validation = validate_answer(
            query="Résume uniquement les anomalies biologiques du report (16), sans poser de diagnostic.",
            answer_text=answer,
            evidence_pack=displayed,
            displayed_evidences=displayed,
            source_citations=self._sources([2]),
            generation_mode="llm_professional_writer",
            output_format_requested="table",
            answer_style_requested="standard",
            query_intents={"is_structured_query": True, "doc_scoped_summary": True},
        )
        self.assertNotIn("unsupported_value", validation.get("errors") or [])

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
        self.assertIn("Vous avez demandé une", answer)
        self.assertIn("unités biologiques sont différentes", answer)
        self.assertIn("écart normalisé", answer)
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
            or ("n’est pas disponible" in answer)
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

    def test_21_writer_pack_contains_visualization_facts(self) -> None:
        qu = parse_query_understanding("Donne les résultats sous forme radar chart.")
        pack = {
            "question": "Q",
            "intent": "doc_scoped_results",
            "requested_doc_ids": ["report_16"],
            "requested_analytes": ["tshus"],
            "output_format": "chart",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": "report_16",
                    "analyte": "TSHus",
                    "analyte_norm": "tshus",
                    "current_value": "55,00",
                    "unit": "mUI/L",
                    "reference": "0,35 à 4,94 mUI/l",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                }
            ],
            "visualization_facts": {
                "requested_type": "radar",
                "requested_label": "graphique radar",
                "rendered_type": "bar",
                "rendered_label": "graphique en barres",
                "supported": False,
                "suitable": True,
                "fallback_used": True,
                "fallback_reason": "Le graphique radar n’est pas encore disponible dans l’interface.",
                "recommendation_reason": "Le graphique en barres permet une comparaison plus lisible.",
            },
        }
        writer_pack = build_writer_evidence_pack(
            user_question="Donne les résultats sous forme radar chart.",
            query_understanding=qu,
            evidence_pack=pack,
            source_citations=[],
        )
        facts = writer_pack.get("visualization_facts") or {}
        self.assertEqual(facts.get("requested_type"), "radar")
        self.assertEqual(facts.get("rendered_type"), "bar")
        self.assertTrue(facts.get("fallback_used"))

    def test_21b_writer_pack_results_are_locked_fact_rows(self) -> None:
        qu = parse_query_understanding("Dans report 16, donne la valeur ACTH.")
        pack = {
            "question": "Q",
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
                    "reference": "<b>4,70 - 48,80 pg/ml</b>",
                    "technical_status": "dans la référence",
                    "source_label": "<ul><li>report (16).pdf — page 1, ligne 1</li></ul>",
                }
            ],
        }
        writer_pack = build_writer_evidence_pack(
            user_question="Dans report 16, donne la valeur ACTH.",
            query_understanding=qu,
            evidence_pack=pack,
            source_citations=[],
        )
        results = list(writer_pack.get("results") or [])
        self.assertEqual(len(results), 1)
        row = dict(results[0])
        self.assertEqual(
            sorted(row.keys()),
            ["analyte", "analyte_norm", "reference", "source_label", "status", "unit", "value"],
        )
        self.assertEqual(row["analyte"], "ACTH")
        self.assertEqual(row["analyte_norm"], "acth")
        self.assertEqual(row["value"], "23,00")
        self.assertEqual(row["unit"], "pg/ml")
        self.assertEqual(row["reference"], "4,70 - 48,80 pg/ml")
        self.assertEqual(row["source_label"], "report (16).pdf — page 1, ligne 1")
        self.assertNotIn("<", "".join(str(v) for v in row.values()))

    def test_21c_writer_pack_evidence_contract_scope_and_sources_are_normalized(self) -> None:
        qu = parse_query_understanding("Listez tous les rapports qui ont la créatinine supérieure à 10.")
        pack = {
            "question": "Q",
            "intent": "cohort_search",
            "requested_doc_ids": [],
            "requested_analytes": ["créat"],
            "output_format": "table",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": "report_12",
                    "analyte": "Créatinine",
                    "analyte_norm": "creatinine",
                    "current_value": "23",
                    "unit": "mg/l",
                    "reference": "7,2 - 12,5 mg/l",
                    "technical_status": "au-dessus de la référence",
                    "source_label": "<b>report (12).pdf — page 1, ligne 13</b>",
                    "viewer_url": "/viewer/pdf?doc_id=report_12&page=1",
                }
            ],
        }
        writer_pack = build_writer_evidence_pack(
            user_question="Listez tous les rapports qui ont la créatinine supérieure à 10.",
            query_understanding=qu,
            evidence_pack=pack,
            source_citations=[
                {
                    "doc_id": "report_12",
                    "filename": "report (12).pdf",
                    "page": 1,
                    "row": 13,
                    "viewer_url": "/viewer/pdf?doc_id=report_12&page=1",
                }
            ],
        )
        contract = dict(writer_pack.get("evidence_contract") or {})
        scope = dict(writer_pack.get("scope") or {})
        sources = list(writer_pack.get("sources") or [])
        constraints = dict(writer_pack.get("constraints") or {})
        self.assertEqual(contract.get("contract_version"), "v1")
        self.assertTrue(bool(contract.get("rows_filtered")))
        self.assertTrue(bool(contract.get("rows_fact_locked")))
        self.assertTrue(bool(contract.get("sources_normalized")))
        self.assertTrue(bool(contract.get("sources_deduplicated")))
        self.assertEqual(constraints.get("requested_analytes"), ["creatinine"])
        self.assertEqual(scope.get("requested_analytes"), ["creatinine"])
        self.assertEqual(scope.get("effective_analytes"), ["creatinine"])
        self.assertTrue(bool(scope.get("scope_coherent")))
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].get("label"), "report (12).pdf — page 1, ligne 13")
        self.assertEqual(sources[0].get("doc_id"), "report_12")
        self.assertTrue(bool(sources[0].get("viewer_url")))

    def test_21d_writer_pack_filters_out_scope_incoherent_rows(self) -> None:
        qu = parse_query_understanding("Dans report 16, donne la valeur ACTH.")
        pack = {
            "question": "Q",
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
                    "technical_status": "dans la référence",
                },
                {
                    "doc_id": "report_12",
                    "analyte": "Créatinine",
                    "analyte_norm": "creatinine",
                    "current_value": "23",
                    "unit": "mg/l",
                    "reference": "7,2 - 12,5 mg/l",
                    "technical_status": "au-dessus de la référence",
                },
            ],
        }
        writer_pack = build_writer_evidence_pack(
            user_question="Dans report 16, donne la valeur ACTH.",
            query_understanding=qu,
            evidence_pack=pack,
            source_citations=[],
        )
        results = list(writer_pack.get("results_locked") or [])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get("analyte_norm"), "acth")

    def test_21e_writer_pack_contract_validator_rejects_incoherent_payload(self) -> None:
        bad_pack = {
            "results_locked": [
                {
                    "analyte": "ACTH",
                    "analyte_norm": "",
                    "value": "23,00",
                    "unit": "pg/ml",
                    "reference": "4,70 - 48,80 pg/ml",
                    "status": "dans la référence",
                    "source_label": "doc_id=report_16",
                }
            ],
            "sources": [{"label": ""}],
            "scope": {"scope_coherent": False},
            "evidence_contract": {"contract_version": "v0", "rows_fact_locked": False, "sources_normalized": False},
            "constraints": {"requested_analytes": ["acth"]},
        }
        ok, errors = _validate_writer_evidence_pack_contract(bad_pack)
        self.assertFalse(ok)
        self.assertIn("results_locked_row_empty_analyte_norm:0", errors)
        self.assertIn("results_locked_row_bad_source_label:0", errors)
        self.assertIn("scope_incoherent", errors)
        self.assertIn("evidence_contract_version_invalid", errors)

    def test_21f_contract_violation_forces_fallback_and_logs(self) -> None:
        qu = parse_query_understanding("Dans report 16, donne la valeur ACTH.")
        bad_pack = {
            "question": "Q",
            "intent": "doc_scoped_results",
            "requested_doc_ids": ["report_16"],
            "requested_analytes": ["acth"],
            "output_format": "table",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": "report_16",
                    "analyte": "",
                    "analyte_norm": "",
                    "current_value": "23,00",
                    "unit": "pg/ml",
                    "reference": "4,70 - 48,80 pg/ml",
                    "technical_status": "dans la référence",
                    "source_label": "doc_id=report_16",
                }
            ],
        }
        with mock.patch.object(LOGGER, "warning") as warning_mock:
            composed = compose_professional_answer(
                user_question="Dans report 16, donne la valeur ACTH.",
                query_understanding=qu,
                evidence_pack=bad_pack,
                mode="llm_professional_writer",
                source_citations=[],
                llm_client=_FakeLLMClient("unused"),
            )
        self.assertEqual(str(composed.get("mode") or ""), "writer_contract_violation_fallback")
        self.assertEqual(str(composed.get("llm_error") or ""), "writer_evidence_contract_violation")
        self.assertTrue(list(composed.get("contract_violation") or []))
        warning_mock.assert_called()
        logged = " ".join(str(arg) for arg in warning_mock.call_args.args)
        self.assertIn("contract_violation", logged)

    def test_22_multi_doc_comparison_identical_is_specific(self) -> None:
        qu = parse_query_understanding("Compare le glucose entre report 10 et report 12.")
        pack = {
            "question": "Compare le glucose entre report 10 et report 12.",
            "intent": "multi_doc_comparison",
            "requested_doc_ids": ["report_10", "report_12"],
            "requested_analytes": ["glucose"],
            "output_format": "list",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": "comparison_report_10_report_12",
                    "analyte": "Glucose",
                    "analyte_norm": "glucose",
                    "current_value": "report_10 = 1 | report_12 = 1",
                    "unit": "g/L",
                    "technical_status": "comparaison effectuée",
                    "comparison_status": "identical",
                    "doc_a": "report 10",
                    "doc_b": "report 12",
                    "value_a": "1",
                    "value_b": "1",
                }
            ],
            "missing_items": [],
        }
        composed = render_professional_fallback(
            evidence_pack=pack,
            query_understanding=qu,
            user_question=pack["question"],
            source_citations=[
                {"doc_id": "report_10", "filename": "report (10).pdf", "page": 2, "row": 18},
                {"doc_id": "report_12", "filename": "report (12).pdf", "page": 1, "row": 9},
            ],
        )
        answer = str(composed.get("answer") or "").lower()
        self.assertIn("aucun écart numérique", answer)
        self.assertNotIn("présents dans un rapport et absents", answer)

    def test_23_multi_doc_presence_diff_keeps_presence_wording(self) -> None:
        qu = parse_query_understanding(
            "Compare report 12 et report 11 et indique quels analytes sont présents dans un rapport mais absents dans l’autre."
        )
        pack = {
            "question": "Q",
            "intent": "multi_doc_presence_diff",
            "requested_doc_ids": ["report_11", "report_12"],
            "requested_analytes": [],
            "output_format": "table",
            "answer_style": "standard",
            "evidences": [
                {
                    "analyte": "AMH",
                    "analyte_norm": "amh",
                    "present_in": "report_12",
                    "absent_in": "report_11",
                }
            ],
            "missing_items": [],
        }
        composed = render_professional_fallback(
            evidence_pack=pack,
            query_understanding=qu,
            user_question=pack["question"],
            source_citations=[],
        )
        answer = str(composed.get("answer") or "").lower()
        self.assertIn("présence/absence", answer)

    def test_24_no_naive_hardcoded_phrases(self) -> None:
        qu = parse_query_understanding("Compare le glucose entre report 10 et report 12.")
        pack = {
            "question": "Q",
            "intent": "multi_doc_comparison",
            "requested_doc_ids": ["report_10", "report_12"],
            "requested_analytes": ["glucose"],
            "output_format": "list",
            "answer_style": "standard",
            "evidences": [
                {
                    "analyte": "Glucose",
                    "analyte_norm": "glucose",
                    "current_value": "report_10 = 1 | report_12 = 1",
                    "unit": "g/L",
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
        )
        self.assertNotIn("J’ai vérifié les données retrouvées", answer)
        self.assertNotIn("correspond strictement aux données extraites", answer)

    def test_25_llm_modified_value_falls_back(self) -> None:
        qu = parse_query_understanding("Compare le glucose entre report 10 et report 12.")
        pack = {
            "question": "Q",
            "intent": "multi_doc_comparison",
            "requested_doc_ids": ["report_10", "report_12"],
            "requested_analytes": ["glucose"],
            "output_format": "list",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": "comparison_report_10_report_12",
                    "filename": "report (10).pdf",
                    "page": 2,
                    "row": 18,
                    "analyte": "Glucose",
                    "analyte_norm": "glucose",
                    "current_value": "report_10 = 1 | report_12 = 1",
                    "unit": "g/L",
                    "source_label": "report (10).pdf — page 2, ligne 18",
                }
            ],
            "missing_items": [],
        }
        out = compose_professional_answer(
            user_question=pack["question"],
            query_understanding=qu,
            evidence_pack=pack,
            mode="auto",
            source_citations=[{"doc_id": "report_10", "filename": "report (10).pdf", "page": 2, "row": 18}],
            llm_client=_FakeLLMClient(
                "Le glucose est plus élevé dans report 12 : 2 g/L contre 1 g/L.\n\nSources : report (10).pdf — page 2, ligne 18"
            ),
        )
        self.assertEqual(str(out.get("mode") or ""), "llm_writer_quality_fallback")

    def test_26_multi_doc_comparison_increased_delta(self) -> None:
        qu = parse_query_understanding("Compare le glucose entre report 10 et report 12.")
        pack = {
            "question": "Q",
            "intent": "multi_doc_comparison",
            "requested_doc_ids": ["report_10", "report_12"],
            "requested_analytes": ["glucose"],
            "output_format": "list",
            "answer_style": "standard",
            "evidences": [
                {
                    "analyte": "Glucose",
                    "analyte_norm": "glucose",
                    "doc_a": "report 10",
                    "doc_b": "report 12",
                    "value_a_raw": "0,8",
                    "value_b_raw": "1,0",
                    "unit_a": "g/L",
                    "unit_b": "g/L",
                    "delta_abs": 0.2,
                    "delta_unit": "g/L",
                    "comparison_status": "increased",
                    "reference_summary": "Plusieurs plages selon l’âge/profil",
                }
            ],
            "missing_items": [],
        }
        ans = str(
            render_professional_fallback(
                evidence_pack=pack,
                query_understanding=qu,
                user_question=pack["question"],
                source_citations=[],
            ).get("answer")
            or ""
        ).lower()
        self.assertIn("augmentation", ans)
        self.assertIn("+0.2 g/l", ans)

    def test_27_multi_doc_comparison_decreased_delta(self) -> None:
        qu = parse_query_understanding("Compare le glucose entre report 10 et report 12.")
        pack = {
            "question": "Q",
            "intent": "multi_doc_comparison",
            "requested_doc_ids": ["report_10", "report_12"],
            "requested_analytes": ["glucose"],
            "output_format": "list",
            "answer_style": "standard",
            "evidences": [
                {
                    "analyte": "Glucose",
                    "analyte_norm": "glucose",
                    "doc_a": "report 10",
                    "doc_b": "report 12",
                    "value_a_raw": "1,2",
                    "value_b_raw": "1,0",
                    "unit_a": "g/L",
                    "unit_b": "g/L",
                    "delta_abs": -0.2,
                    "delta_unit": "g/L",
                    "comparison_status": "decreased",
                    "reference_summary": "Plusieurs plages selon l’âge/profil",
                }
            ],
            "missing_items": [],
        }
        ans = str(
            render_professional_fallback(
                evidence_pack=pack,
                query_understanding=qu,
                user_question=pack["question"],
                source_citations=[],
            ).get("answer")
            or ""
        ).lower()
        self.assertIn("diminution", ans)
        self.assertIn("-0.2 g/l", ans)

    def test_28_multi_doc_comparison_non_comparable_units(self) -> None:
        qu = parse_query_understanding("Compare le glucose entre report 10 et report 12.")
        pack = {
            "question": "Q",
            "intent": "multi_doc_comparison",
            "requested_doc_ids": ["report_10", "report_12"],
            "requested_analytes": ["glucose"],
            "output_format": "list",
            "answer_style": "standard",
            "evidences": [
                {
                    "analyte": "Glucose",
                    "analyte_norm": "glucose",
                    "doc_a": "report 10",
                    "doc_b": "report 12",
                    "value_a_raw": "90",
                    "value_b_raw": "1",
                    "unit_a": "mg/dL",
                    "unit_b": "g/L",
                    "comparison_status": "non_comparable",
                    "reference_summary": "Plusieurs plages selon l’âge/profil",
                }
            ],
            "missing_items": [],
        }
        ans = str(
            render_professional_fallback(
                evidence_pack=pack,
                query_understanding=qu,
                user_question=pack["question"],
                source_citations=[],
            ).get("answer")
            or ""
        ).lower()
        self.assertIn("non comparable", ans)

    def test_29_hybrid_mode_uses_llm_writer_with_locked_facts(self) -> None:
        qu = parse_query_understanding("Dans report 19, quels résultats sont hors référence ?")
        pack = {
            "question": "Dans report 19, quels résultats sont hors référence ?",
            "intent": "doc_scoped_abnormal_results",
            "requested_doc_ids": ["report_19"],
            "requested_analytes": [],
            "output_format": "list",
            "answer_style": "standard",
            "evidences": [
                {
                    "doc_id": "report_19",
                    "analyte": "INSULINE",
                    "analyte_norm": "insuline",
                    "current_value": "23,00",
                    "unit": "uU/mL",
                    "reference": "4 à 20 µIU/mL",
                    "technical_status_code": "above_reference",
                    "technical_status": "au-dessus de la référence",
                    "page": 1,
                    "row": 1,
                }
            ],
            "missing_items": [],
        }
        llm_answer = "Pour l’insuline, la valeur mesurée est 23,00 uU/mL (référence : 4 à 20 µIU/mL), statut : au-dessus de la référence. Source : report (16).pdf — page 1, ligne 1."
        composed = compose_professional_answer(
            user_question=pack["question"],
            query_understanding=qu,
            evidence_pack=pack,
            mode="hybrid_structured_llm_writer",
            source_citations=self._sources([1]),
            llm_client=_FakeLLMClient(llm_answer),
        )
        self.assertEqual(str(composed.get("mode") or ""), "hybrid_structured_llm_writer")
        self.assertIn("23,00", str(composed.get("answer") or ""))


if __name__ == "__main__":
    unittest.main()
