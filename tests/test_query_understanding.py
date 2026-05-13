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

    def test_gte_operator_detection(self) -> None:
        qu = parse_query_understanding("Liste-moi tous les patients qui ont ACTH avec une valeur supérieure ou égale à 23,00.")
        self.assertEqual(qu.comparison_operator, ">=")
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

    def test_chart_presentation_intent_line(self) -> None:
        qu = parse_query_understanding("Dans report 16, liste les résultats hors référence sous forme Arithmetic Line-Graph.")
        self.assertEqual(qu.output_format, "chart")
        self.assertEqual(qu.presentation_intent.requested_output, "chart")
        self.assertEqual(qu.presentation_intent.chart_type, "line")
        self.assertTrue(qu.presentation_intent.user_requested_visualization)
        self.assertFalse(qu.presentation_intent.unsupported_format)
        self.assertIn("Arithmetic", str(qu.raw_format_phrase or ""))

    def test_chart_presentation_intent_bar(self) -> None:
        qu = parse_query_understanding("Dans report 16, affiche les résultats hors référence sous forme de graphique en barres.")
        self.assertEqual(qu.output_format, "chart")
        self.assertEqual(qu.presentation_intent.chart_type, "bar")
        self.assertTrue(qu.presentation_intent.user_requested_visualization)

    def test_unknown_or_complex_format_is_preserved(self) -> None:
        qu = parse_query_understanding(
            "Dans report 16, affiche les résultats hors référence sous forme bio-clinical matrix radar comparative."
        )
        self.assertEqual(qu.output_format, "chart")
        self.assertTrue(bool(qu.raw_format_phrase))
        self.assertTrue(bool(qu.unhandled_instructions))
        self.assertEqual(qu.presentation_intent.chart_type, "radar")

    def test_chart_presentation_intent_heatmap(self) -> None:
        qu = parse_query_understanding("Affiche les résultats sous forme heatmap comparative.")
        self.assertEqual(qu.output_format, "chart")
        self.assertEqual(qu.presentation_intent.chart_type, "heatmap")
        self.assertTrue(qu.presentation_intent.user_requested_visualization)

    def test_chart_presentation_intent_unknown_type(self) -> None:
        qu = parse_query_understanding("Donne les résultats sous forme bio-clinical matrix hyper-radar mode.")
        self.assertEqual(qu.output_format, "chart")
        self.assertTrue(qu.presentation_intent.user_requested_visualization)
        self.assertTrue(bool(qu.raw_format_phrase))

    def test_ok_alone_is_small_talk(self) -> None:
        qu = parse_query_understanding("ok")
        self.assertEqual(qu.intent, "small_talk")

    def test_ok_transform_followup_is_response_transform(self) -> None:
        qu = parse_query_understanding("ok donne moi le résultat en JSON strict")
        self.assertEqual(qu.intent, "response_transform")
        self.assertEqual(qu.response_strategy, "transform_previous_response")

    def test_excluded_analytes_detection(self) -> None:
        qu = parse_query_understanding(
            "Quels patients ont une TSHus au-dessus de la référence ? N’inclus pas TRAK, Anti-TG, ni anticorps anti-récepteur de la TSH."
        )
        self.assertIn("trak", qu.excluded_analytes)

    def test_visualization_recommendation_intent(self) -> None:
        qu = parse_query_understanding(
            "ok si ces donnees ne sont pas des valeurs transformables, recommande-moi une visualisation qui correspond a ce type de donnees"
        )
        self.assertEqual(qu.intent, "visualization_recommendation")
        self.assertTrue(bool(qu.intents.get("visualization_recommendation")))

    def test_inventory_visualization_render_intent(self) -> None:
        qu = parse_query_understanding("ok affiche a moi avec des cartes patient avec le nombre de rapports associés")
        self.assertEqual(qu.intent, "inventory_visualization_render")
        self.assertTrue(bool(qu.intents.get("inventory_visualization_render")))
        self.assertEqual(qu.inventory_view_type, "patient_cards")

    def test_inventory_view_type_accordion(self) -> None:
        qu = parse_query_understanding("ok affiche ça dans liste accordéon pour ouvrir les rapports de chaque patient")
        self.assertEqual(qu.inventory_view_type, "report_accordion")

    def test_inventory_view_type_filterable_table(self) -> None:
        qu = parse_query_understanding("ok affiche ça dans une table filtrable par patient, date ou nom de fichier")
        self.assertEqual(qu.inventory_view_type, "filterable_table")

    def test_multi_analytes_latest_report_detection(self) -> None:
        qu = parse_query_understanding("montre les résultats ACTH, TSHus, T4 libre, T3 libre et ANTI-TG du dernier rapport")
        self.assertTrue(qu.latest_report)
        self.assertIn("acth", qu.requested_analytes)
        self.assertIn("tshus", qu.requested_analytes)
        self.assertIn("t4_libre", qu.requested_analytes)
        self.assertIn("t3_libre", qu.requested_analytes)
        self.assertIn("anti_tg", qu.requested_analytes)

    def test_date_scope_extraction(self) -> None:
        qu = parse_query_understanding("montre les résultats du rapport du 20/06/2025")
        self.assertEqual(qu.requested_date_iso, "2025-06-20")

    def test_plural_gte_operator_detection(self) -> None:
        qu = parse_query_understanding("trouve les résultats ACTH supérieurs ou égaux à 23")
        self.assertEqual(qu.comparison_operator, ">=")
        self.assertEqual(qu.requested_value, "23")

    def test_qualitative_context_detection(self) -> None:
        qu = parse_query_understanding("montre le commentaire sur la troponine")
        self.assertEqual(qu.requested_context_type, "medical_qualitative_comment")

    def test_qualitative_comment_render_intent(self) -> None:
        qu = parse_query_understanding("ok affiche ce commentaire dans un bloc commentaire sourcé")
        self.assertEqual(qu.intent, "qualitative_comment_render")
        self.assertTrue(bool(qu.intents.get("qualitative_comment_render")))
        self.assertEqual(qu.qualitative_view_type, "sourced_comment_block")

    def test_qualitative_view_type_text_table(self) -> None:
        qu = parse_query_understanding("ok affiche ce commentaire dans un tableau texte : sujet, commentaire, source")
        self.assertEqual(qu.qualitative_view_type, "text_table")

    def test_qualitative_view_type_interpretive_note(self) -> None:
        qu = parse_query_understanding("ok affiche ce commentaire dans un encadré de note interprétative")
        self.assertEqual(qu.qualitative_view_type, "interpretive_note")

    def test_clickable_source_detection_extended_markers(self) -> None:
        qu = parse_query_understanding("affiche ce commentaire dans un tableau texte avec source cliquable et ouvrir PDF")
        self.assertTrue(qu.source_clickable_requested)


if __name__ == "__main__":
    unittest.main()
