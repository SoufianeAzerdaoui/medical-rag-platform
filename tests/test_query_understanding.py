from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_SCRIPT_ROOT))

from query_understanding import (
    detect_exact_analytes,
    detect_query_intents,
    detect_requested_doc_ids,
    match_analyte,
    parse_query_understanding,
)


class TestQueryUnderstanding(unittest.TestCase):
    def test_detect_requested_doc_ids_single(self) -> None:
        query = "Dans report 31, quels résultats d’immunoanalyse nécessitent une attention technique ?"
        self.assertEqual(detect_requested_doc_ids(query), ["report_31"])

    def test_detect_requested_doc_ids_multiple(self) -> None:
        query = "Compare report 12 et report (11) sur CRP et CKMB."
        self.assertEqual(detect_requested_doc_ids(query), ["report_12", "report_11"])

    def test_detect_requested_doc_ids_deduplicated(self) -> None:
        query = "Dans report_21 puis report 21, vérifie le lithium."
        self.assertEqual(detect_requested_doc_ids(query), ["report_21"])

    def test_detect_requested_doc_ids_rapport_document(self) -> None:
        query = "Dans le rapport 31 et le document 12, compare CKMB."
        self.assertEqual(detect_requested_doc_ids(query), ["report_31", "report_12"])

    def test_detect_exact_analytes_extended_aliases(self) -> None:
        query = "Compare l’insuline, la T4 libre, CKMB, triglycérides et microalbuminurie."
        analytes = detect_exact_analytes(query)
        self.assertIn("insuline", analytes)
        self.assertIn("t4_libre", analytes)
        self.assertIn("ckmb", analytes)
        self.assertIn("triglycerides", analytes)
        self.assertIn("microalbuminurie", analytes)

    def test_detect_query_intents(self) -> None:
        query = "Compare report 12 et report 11 sur CRP, CKMB, triglycérides."
        doc_ids = detect_requested_doc_ids(query)
        analytes = detect_exact_analytes(query)
        intents = detect_query_intents(query, requested_doc_ids=doc_ids, analytes=analytes)
        self.assertTrue(intents.get("multi_doc_comparison"))
        self.assertTrue(intents.get("multi_analyte_results"))
        self.assertTrue(intents.get("is_structured_query"))

    def test_parse_query_understanding_output_format_table(self) -> None:
        qu = parse_query_understanding(
            "Dans report 14, liste les tests toxicologiques urinaires avec antérieurs sous forme tableau, colonnes : analyte, valeur actuelle, référence, statut, résultat antérieur, variation."
        )
        self.assertEqual(qu.requested_doc_ids, ["report_14"])
        self.assertEqual(qu.output_format, "table")
        self.assertEqual(
            qu.requested_table_columns,
            ["analyte", "valeur_actuelle", "reference", "statut", "resultat_anterieur", "variation"],
        )
        self.assertTrue(qu.requires_previous_results)

    def test_parse_query_understanding_yes_no_and_global_patient_lookup(self) -> None:
        qu_yes_no = parse_query_understanding(
            "Dans report 16, est-ce que l’ACTH est hors référence ? Réponds uniquement oui/non."
        )
        self.assertEqual(qu_yes_no.output_format, "yes_no")
        self.assertEqual(qu_yes_no.answer_style, "yes_no")
        self.assertEqual(qu_yes_no.intent, "doc_scoped_results")
        self.assertEqual(qu_yes_no.language, "fr")

        qu_global = parse_query_understanding(
            "retour a moi tous les patients qui ont ACIDE VALPOROIQUE (DEPAKINE) est 030"
        )
        self.assertEqual(qu_global.intent, "cohort_search")
        self.assertEqual(qu_global.requested_value, "030")
        self.assertTrue(qu_global.patient_query)
        self.assertTrue(qu_global.requires_global_search)

    def test_parse_query_understanding_yes_no_english_markers(self) -> None:
        qu = parse_query_understanding(
            "Dans report 16, est-ce que l’ACTH est hors référence ? Réponds uniquement yes ou no, avec la valeur, la référence et la source."
        )
        self.assertEqual(qu.output_format, "yes_no")
        self.assertEqual(qu.answer_style, "yes_no")

    def test_cohort_search_with_technical_condition(self) -> None:
        qu = parse_query_understanding(
            "Liste tous les patients qui ont une INSULINE en dessous de la référence. Donne patient, report, valeur, référence et source."
        )
        self.assertEqual(qu.intent, "cohort_search")
        self.assertTrue(qu.requires_global_search)
        self.assertEqual(qu.technical_condition, "below_reference")
        self.assertIn("insuline", qu.requested_analytes)

    def test_multi_doc_presence_diff_intent(self) -> None:
        qu = parse_query_understanding(
            "Compare report 12 et report 11 et indique quels analytes sont présents dans un rapport mais absents dans l’autre."
        )
        self.assertEqual(qu.intent, "multi_doc_presence_diff")
        self.assertEqual(qu.requested_doc_ids, ["report_12", "report_11"])

    def test_response_transform_intent(self) -> None:
        qu = parse_query_understanding("Convertis la réponse précédente en JSON strict.")
        self.assertEqual(qu.intent, "response_transform")
        self.assertEqual(qu.output_format, "json")
        self.assertTrue(qu.is_response_transform)

    def test_tshus_does_not_match_trak(self) -> None:
        self.assertFalse(match_analyte("ANTICORPS ANTI RECEPTEUR DE LA TSH (TRAK)", "tshus"))
        self.assertTrue(match_analyte("TSHus", "tshus"))

    def test_strict_operator_detection(self) -> None:
        qu = parse_query_understanding("Liste les patients avec ACTH strictement supérieure à 23,00.")
        self.assertEqual(qu.comparison_operator, ">")
        self.assertEqual(qu.requested_value, "23,00")

    def test_small_talk_intent(self) -> None:
        qu = parse_query_understanding("bonjour")
        self.assertEqual(qu.intent, "small_talk")
        self.assertTrue(qu.is_small_talk)

    def test_identity_intent(self) -> None:
        qu = parse_query_understanding("t es qui")
        self.assertEqual(qu.intent, "identity_question")
        self.assertTrue(qu.intents.get("general_conversation"))
        self.assertFalse(qu.is_small_talk)

    def test_capability_intent(self) -> None:
        qu = parse_query_understanding("tu peux faire quoi")
        self.assertEqual(qu.intent, "capability_question")
        self.assertTrue(qu.intents.get("general_conversation"))
        self.assertFalse(qu.is_small_talk)

    def test_help_intent(self) -> None:
        qu = parse_query_understanding("help")
        self.assertEqual(qu.intent, "help_question")
        self.assertTrue(qu.intents.get("general_conversation"))
        self.assertFalse(qu.is_small_talk)

    def test_excluded_analytes_detection(self) -> None:
        qu = parse_query_understanding(
            "Quels patients ont une TSHus au-dessus de la référence ? N’inclus pas TRAK, Anti-TG, ni anticorps anti-récepteur de la TSH."
        )
        self.assertIn("trak", qu.excluded_analytes)


if __name__ == "__main__":
    unittest.main()
