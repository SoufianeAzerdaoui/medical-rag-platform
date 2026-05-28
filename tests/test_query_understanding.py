from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_SCRIPT_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_SCRIPT_ROOT))

from query_understanding import (
    build_intent_arbitration_debug,
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

    def test_response_transform_paragraph_not_misdetected_as_chart(self) -> None:
        qu = parse_query_understanding("Convertis la réponse précédente en style paragraphe médical pro.")
        self.assertEqual(qu.intent, "response_transform")
        self.assertEqual(qu.output_format, "paragraph")
        self.assertFalse(bool(getattr(qu.presentation_intent, "user_requested_visualization", False)))

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

    def test_cholesterol_total_alias_detection(self) -> None:
        analytes = detect_exact_analytes("donne la plage de cholestérol total")
        self.assertIn("cholesterol_total", analytes)

    def test_ok_suffix_does_not_force_small_talk_for_medical_request(self) -> None:
        qu = parse_query_understanding(
            "donne la plage de Cholestérol total, juste Cholestérol total, avec les sources disponibles, liste aussi les autres documents ok"
        )
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertIn("cholesterol_total", qu.requested_analytes)
        self.assertFalse(qu.is_small_talk)

    def test_reference_ranges_summary_intent_for_doc_scoped_request(self) -> None:
        qu = parse_query_understanding(
            "Tu peux faire une note sur les différentes plages physiologiques et références selon sexe/âge dans report 12 ?"
        )
        self.assertEqual(qu.intent, "reference_ranges_summary")
        self.assertEqual(qu.requested_doc_ids, ["report_12"])

    def test_reference_ranges_summary_intent_with_noisy_french_typo(self) -> None:
        qu = parse_query_understanding(
            "tu peux faire une note pour les differents plages qui exist dans les valeurs phisiologique dans report 12"
        )
        self.assertEqual(qu.intent, "reference_ranges_summary")
        self.assertEqual(qu.requested_doc_ids, ["report_12"])

    def test_ok_transform_followup_is_response_transform(self) -> None:
        qu = parse_query_understanding("ok donne moi le résultat en JSON strict")
        self.assertEqual(qu.intent, "response_transform")
        self.assertEqual(qu.response_strategy, "transform_previous_response")

    def test_direct_analyte_result_request_is_not_transform(self) -> None:
        qu = parse_query_understanding("donne moi le resultat de AMH")
        self.assertEqual(qu.intent, "doc_scoped_results")
        self.assertIn("amh", qu.requested_analytes)
        self.assertFalse(qu.is_response_transform)

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

    def test_qualitative_comment_without_analyte_intent(self) -> None:
        qu = parse_query_understanding("montre le commentaire")
        self.assertEqual(qu.intent, "comment_without_measured_value")
        self.assertTrue(bool(qu.intents.get("comment_without_measured_value")))

    def test_list_all_comments_intent(self) -> None:
        qu = parse_query_understanding("liste moi tous les commentaires existants")
        self.assertEqual(qu.intent, "comment_without_measured_value")
        self.assertTrue(bool(qu.intents.get("comment_without_measured_value")))

    def test_latest_report_comment_intent(self) -> None:
        qu = parse_query_understanding("liste moi le commentaire du dernier rapport")
        self.assertEqual(qu.intent, "comment_without_measured_value")
        self.assertTrue(qu.latest_report)

    def test_single_comment_intent(self) -> None:
        qu = parse_query_understanding("liste une seule commentaire")
        self.assertEqual(qu.intent, "comment_without_measured_value")

    def test_single_comment_implicit_intent(self) -> None:
        qu = parse_query_understanding("liste une commentaire")
        self.assertEqual(qu.intent, "comment_without_measured_value")

    def test_qualitative_comment_render_intent(self) -> None:
        qu = parse_query_understanding("ok affiche ce commentaire dans un bloc commentaire sourcé")
        self.assertEqual(qu.intent, "qualitative_comment_render")
        self.assertTrue(bool(qu.intents.get("qualitative_comment_render")))
        self.assertEqual(qu.qualitative_view_type, "sourced_comment_block")

    def test_qualitative_view_type_text_table(self) -> None:
        qu = parse_query_understanding("ok affiche ce commentaire dans un tableau texte : sujet, commentaire, source")
        self.assertEqual(qu.qualitative_view_type, "text_table")

    def test_source_followup_intent(self) -> None:
        qu = parse_query_understanding("d'où vient ce commentaire ?")
        self.assertEqual(qu.intent, "source_followup")
        self.assertTrue(bool(qu.intents.get("source_followup")))

    def test_context_summary_render_intent_and_points_digit(self) -> None:
        qu = parse_query_understanding("résume ce commentaire en 3 points")
        self.assertEqual(qu.intent, "context_summary_render")
        self.assertTrue(bool(qu.intents.get("context_summary_render")))
        self.assertEqual(qu.requested_summary_points, 3)

    def test_context_summary_render_points_word(self) -> None:
        qu = parse_query_understanding("fais une synthèse de ça en cinq points")
        self.assertEqual(qu.intent, "context_summary_render")
        self.assertEqual(qu.requested_summary_points, 5)

    def test_context_summary_render_points_two_and_six(self) -> None:
        qu2 = parse_query_understanding("résume ça en 2 points")
        qu6 = parse_query_understanding("résume ça en 6 points")
        self.assertEqual(qu2.intent, "context_summary_render")
        self.assertEqual(qu6.intent, "context_summary_render")
        self.assertEqual(qu2.requested_summary_points, 2)
        self.assertEqual(qu6.requested_summary_points, 6)

    def test_same_action_for_subject_detects_analyte(self) -> None:
        qu = parse_query_understanding("fais la même chose pour TSHus")
        self.assertIn("tshus", qu.requested_analytes)

    def test_qualitative_view_type_interpretive_note(self) -> None:
        qu = parse_query_understanding("ok affiche ce commentaire dans un encadré de note interprétative")
        self.assertEqual(qu.qualitative_view_type, "interpretive_note")

    def test_clickable_source_detection_extended_markers(self) -> None:
        qu = parse_query_understanding("affiche ce commentaire dans un tableau texte avec source cliquable et ouvrir PDF")
        self.assertTrue(qu.source_clickable_requested)

    def test_reference_range_lookup_male_over_60(self) -> None:
        qu = parse_query_understanding("Dans le rapport d'immunoanalyse du 19/07/2024, quelle est la plage normale de la PTH intacte ?")
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertIn("pth", "".join(qu.requested_analytes))
        self.assertEqual(qu.requested_date_iso, "2024-07-19")
        self.assertEqual(qu.requested_report_type, "immunoanalyse")

    def test_reference_range_lookup_infant_population(self) -> None:
        qu = parse_query_understanding("valeurs physiologiques calcium nourrisson")
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertEqual((qu.requested_reference_profile or {}).get("population"), "infant")

    def test_reference_range_lookup_female(self) -> None:
        qu = parse_query_understanding("Dans le rapport d'immunoanalyse du 19/07/2024, quelle est la norme AMH pour une femme de 25–29 ans ?")
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertIn("amh", qu.requested_analytes)
        self.assertEqual((qu.requested_reference_profile or {}).get("sex"), "female")
        self.assertEqual((qu.requested_reference_profile or {}).get("age_min"), 25.0)
        self.assertEqual((qu.requested_reference_profile or {}).get("age_max"), 29.0)
        self.assertEqual((qu.requested_reference_profile or {}).get("age_unit"), "years")

    def test_reference_range_lookup_use_patient_profile(self) -> None:
        qu = parse_query_understanding("calcium pour ce patient")
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertTrue(qu.use_patient_profile)

    def test_reference_range_lookup_request_all_ranges(self) -> None:
        qu = parse_query_understanding("donne toutes les plages du calcium")
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertTrue(qu.request_all_reference_ranges)

    def test_reference_range_lookup_report_type_biochimie(self) -> None:
        qu = parse_query_understanding("Dans le rapport de biochimie, quelle est la plage normale du Calcium pour Homme > 60 ans ?")
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertEqual(qu.requested_report_type, "biochimie")

    def test_reference_range_lookup_cycled_female_population_without_age(self) -> None:
        qu = parse_query_understanding("et la plage de AMH pour femme cyclée J2-J4")
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertIn("amh", qu.requested_analytes)
        profile = qu.requested_reference_profile or {}
        self.assertEqual(profile.get("sex"), "female")
        self.assertEqual(profile.get("population"), "cycled_female_j2_j4")
        self.assertIsNone(profile.get("age_min"))
        self.assertIsNone(profile.get("age_max"))

    def test_reference_range_lookup_haptoglobine_femme(self) -> None:
        qu = parse_query_understanding("Quelle est la plage normale de l’haptoglobine chez la femme ?")
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertIn("haptoglobine", qu.requested_analytes)
        profile = qu.requested_reference_profile or {}
        self.assertEqual(profile.get("sex"), "female")
        self.assertIsNone(profile.get("age_min"))

    def test_reference_range_lookup_haptoglobine_femme_over_60(self) -> None:
        qu = parse_query_understanding("Quelle est la plage normale de l’haptoglobine chez la femme de plus de 60 ans ?")
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertIn("haptoglobine", qu.requested_analytes)
        profile = qu.requested_reference_profile or {}
        self.assertEqual(profile.get("sex"), "female")
        self.assertEqual(profile.get("age_operator"), ">")
        self.assertEqual(profile.get("age"), 60.0)
        self.assertEqual(profile.get("age_unit"), "years")

    def test_reference_range_lookup_pal_homme(self) -> None:
        qu = parse_query_understanding("Quelle est la norme de la phosphatase alcaline chez l’homme ?")
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertIn("phosphatase_alcaline", qu.requested_analytes)
        profile = qu.requested_reference_profile or {}
        self.assertEqual(profile.get("sex"), "male")

    def test_reference_range_lookup_pal_homme_12_15(self) -> None:
        qu = parse_query_understanding("Quelle est la norme de la phosphatase alcaline chez l’homme de 12 à 15 ans ?")
        self.assertEqual(qu.intent, "reference_range_lookup")
        self.assertIn("phosphatase_alcaline", qu.requested_analytes)
        profile = qu.requested_reference_profile or {}
        self.assertEqual(profile.get("sex"), "male")
        self.assertEqual(profile.get("age_min"), 12.0)
        self.assertEqual(profile.get("age_max"), 15.0)
        self.assertEqual(profile.get("age_unit"), "years")

    def test_multi_doc_comparison_glycemie_glucose_detected(self) -> None:
        qu = parse_query_understanding("Compare les résultats de la Glycémie (Glucose) entre le report 10 et le report 12.")
        self.assertEqual(qu.intent, "doc_pair_comparison")
        self.assertIn("report_10", qu.requested_doc_ids)
        self.assertIn("report_12", qu.requested_doc_ids)
        self.assertIn("glucose", qu.requested_analytes)

    def test_doc_scoped_out_of_range_not_misrouted_to_comment(self) -> None:
        qu = parse_query_understanding(
            "Dans report (19), quels résultats sont hors référence ? Donne paramètre, valeur, référence, statut technique above_reference/below_reference. Ne donne aucune interprétation médicale."
        )
        self.assertEqual(qu.requested_doc_ids, ["report_19"])
        self.assertEqual(qu.technical_condition, "out_of_reference")
        self.assertNotEqual(qu.intent, "comment_without_measured_value")
        self.assertEqual(qu.intent, "doc_scoped_abnormal_results")

    def test_doc_scoped_summary_keeps_primary_intent_with_safety_constraint(self) -> None:
        qu = parse_query_understanding(
            "Résume uniquement les anomalies biologiques du report (16), sans poser de diagnostic."
        )
        self.assertEqual(qu.requested_doc_ids, ["report_16"])
        self.assertEqual(qu.intent, "doc_scoped_abnormal_results")
        self.assertEqual(qu.safety_intent, "no_diagnosis_constraint")

    def test_note_medecin_without_diagnostic_stays_numeric_context(self) -> None:
        qu = parse_query_understanding("Fais une note médecin courte pour report 12, sans diagnostic.")
        self.assertEqual(qu.requested_doc_ids, ["report_12"])
        self.assertEqual(qu.safety_intent, "no_diagnosis_constraint")
        self.assertEqual(qu.requested_context_type, "biological_numeric_results")
        self.assertEqual(qu.answer_style, "doctor_note")

    def test_pure_diagnostic_question_without_data_scope(self) -> None:
        qu = parse_query_understanding("Peut-on conclure à un cancer avec ces marqueurs ?")
        self.assertEqual(qu.intent, "diagnostic_safety_question")

    def test_intent_arbitration_debug_for_doc_summary_with_safety(self) -> None:
        qu = parse_query_understanding("Résume uniquement les anomalies biologiques du report (16), sans poser de diagnostic.")
        arb = build_intent_arbitration_debug(qu)
        self.assertEqual(arb.get("winner"), "doc_scoped_abnormal_results")
        self.assertIn("doc_scoped_abnormal_results", list(arb.get("candidate_intents") or []))
        self.assertEqual(arb.get("safety_intent"), "no_diagnosis_constraint")
        self.assertTrue(str(arb.get("reason") or "").strip())

    def test_global_analyte_abnormal_search_priority(self) -> None:
        qu = parse_query_understanding(
            "Dans tous les rapports disponibles, quels documents contiennent une insuline hors référence ? Donne le document, la valeur, la référence et le statut."
        )
        self.assertEqual(qu.intent, "global_analyte_abnormal_search")
        self.assertIn("insuline", qu.requested_analytes)
        self.assertEqual(qu.technical_condition, "out_of_reference")

    def test_doc_scoped_medical_interpretation_guarded(self) -> None:
        qu = parse_query_understanding("Est-ce que le report (16) permet de conclure à une hyperthyroïdie ?")
        self.assertEqual(qu.intent, "doc_scoped_medical_interpretation_guarded")
        self.assertEqual(qu.requested_doc_ids, ["report_16"])

    def test_multi_doc_comparison_three_docs_not_doc_pair(self) -> None:
        qu = parse_query_understanding(
            "Compare les reports 16, 19 et 31 sur l’insuline et le bilan thyroïdien, en précisant les données manquantes."
        )
        self.assertEqual(qu.intent, "multi_doc_comparison")
        self.assertEqual(qu.requested_doc_ids, ["report_16", "report_19", "report_31"])

    def test_doc_scoped_priority_anomalies_intent(self) -> None:
        qu = parse_query_understanding("Dans report (10), liste les anomalies importantes par ordre de priorité technique.")
        self.assertEqual(qu.intent, "doc_scoped_priority_anomalies")
        self.assertEqual(qu.requested_doc_ids, ["report_10"])

    def test_phase3_intent_candidates_present_and_sorted(self) -> None:
        qu = parse_query_understanding("la créat du report 29 est basse ?")
        self.assertTrue(isinstance(qu.intent_candidates, list))
        self.assertGreaterEqual(len(qu.intent_candidates), 1)
        self.assertLessEqual(len(qu.intent_candidates), 3)
        for i in range(len(qu.intent_candidates) - 1):
            self.assertGreaterEqual(
                float(qu.intent_candidates[i]["confidence"]),
                float(qu.intent_candidates[i + 1]["confidence"]),
            )

    def test_phase3_intent_confidence_range(self) -> None:
        qu = parse_query_understanding("y a quoi d'anormal dans report 24 ?")
        self.assertGreaterEqual(qu.intent_confidence, 0.0)
        self.assertLessEqual(qu.intent_confidence, 1.0)

    def test_phase3_scope_confidence_doc_scoped_high(self) -> None:
        qu = parse_query_understanding("la créat du report 29 est basse ?")
        self.assertGreater(qu.scope_confidence, 0.80)

    def test_phase3_scope_confidence_missing_doc_low(self) -> None:
        qu = parse_query_understanding("les résultats anormaux")
        self.assertLess(qu.scope_confidence, 0.40)

    def test_phase3_ambiguity_flags_list(self) -> None:
        qu = parse_query_understanding("la créat du report 29 est basse ?")
        self.assertTrue(isinstance(qu.ambiguity_flags, list))
        self.assertTrue(all(isinstance(flag, str) for flag in qu.ambiguity_flags))

    def test_phase3_patient_question_sets_safety_flag(self) -> None:
        qu = parse_query_understanding("le patient a quoi ?")
        self.assertIn("unsafe_diagnosis_request", qu.ambiguity_flags)
        self.assertIsNotNone(qu.safety_intent)

    def test_phase3_medical_topics_detected(self) -> None:
        qu = parse_query_understanding("créatinine basse report 29")
        topics = list(qu.medical_topics or [])
        names = [str(t.get("topic") or "") for t in topics]
        self.assertIn("renal", names)
        for topic in topics:
            conf = float(topic.get("confidence") or 0.0)
            self.assertGreaterEqual(conf, 0.30)
            self.assertLessEqual(conf, 1.0)

    def test_phase3_backward_compat_intent_matches_top_candidate(self) -> None:
        qu = parse_query_understanding("la créat du report 29 est basse ?")
        self.assertEqual(qu.intent, str(qu.intent_candidates[0]["intent"]))

    def test_phase3_deterministic_stability(self) -> None:
        query = "la créat du report 29 est basse ?"
        results = [parse_query_understanding(query) for _ in range(5)]
        first = results[0]
        for other in results[1:]:
            self.assertEqual(other.intent_candidates, first.intent_candidates)
            self.assertEqual(other.intent_confidence, first.intent_confidence)
            self.assertEqual(other.scope_confidence, first.scope_confidence)
            self.assertEqual(other.ambiguity_flags, first.ambiguity_flags)

    def test_treatment_request_sets_treatment_safety_intent(self) -> None:
        qu = parse_query_understanding("donne le traitement")
        self.assertEqual(qu.safety_intent, "treatment_refusal")
        self.assertFalse(bool((qu.intents or {}).get("diagnostic_safety_question")))
        self.assertTrue(bool((qu.intents or {}).get("treatment_safety_question")))


if __name__ == "__main__":
    unittest.main()
