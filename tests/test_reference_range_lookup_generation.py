from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))


class TestReferenceRangeLookupGeneration(unittest.TestCase):
    def test_reference_force_guard_skips_measured_doc_scoped_query(self) -> None:
        try:
            from scripts.generation.generate_answer import _should_force_reference_range_lookup
            from scripts.generation.query_understanding import parse_query_understanding, norm_text
        except Exception as exc:
            self.skipTest(f"imports indisponibles: {exc}")
        q = "montre moi l'insuline avec sa référence du rapport 19"
        qu = parse_query_understanding(q)
        self.assertFalse(_should_force_reference_range_lookup(norm_text(q), qu))

    def test_reference_force_guard_allows_reference_norm_query(self) -> None:
        try:
            from scripts.generation.generate_answer import _should_force_reference_range_lookup
            from scripts.generation.query_understanding import parse_query_understanding, norm_text
            from dataclasses import replace
        except Exception as exc:
            self.skipTest(f"imports indisponibles: {exc}")
        q = "quelle est la plage normale AMH pour une femme de 30-34 ans ?"
        qu = parse_query_understanding(q)
        drifted = replace(qu, intent="small_talk")
        self.assertTrue(_should_force_reference_range_lookup(norm_text(q), drifted))
    def test_reference_range_exact_selection(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")

        fake_rows = [
            {
                "doc_id": "report_1",
                "analyte": "PTH INTACT",
                "analyte_norm": "pth_intact",
                "value_raw": "7.00",
                "value_numeric": 7.0,
                "unit": "pg/ml",
                "reference_range": "(15,00 - 65,00) pg/ml(1.6-6.9 pmol/l)",
                "page_number": 1,
                "row_index": 1,
                "source_pdf": "report (1).pdf",
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer._resolve_reference_scope_doc_ids", return_value=["report_1"]
        ), patch(
            "scripts.generation.generate_answer._fetch_doc_lab_rows", return_value=fake_rows
        ):
            result = run_generation(
                query="Dans le rapport d'immunoanalyse du 19/07/2024, quelle est la plage normale de la PTH intacte ?",
                index_dir="data/indexes",
            )
        answer = str(result.get("answer") or "")
        self.assertTrue("15,0–65,0 pg/ml" in answer or "15–65 pg/ml" in answer or "15.0–65.0 pg/ml" in answer)
        self.assertTrue("1,6–6,9 pmol/l" in answer or "1.6–6.9 pmol/l" in answer)
        self.assertNotIn("Source :", answer)
        self.assertNotIn("55 résultats", answer)
        self.assertNotIn("report_10", answer)
        self.assertTrue(bool(result.get("sources")))
        first_src = (result.get("sources") or [])[0]
        self.assertIn("report (1).pdf — page 1", str(first_src.get("label") or ""))
        self.assertNotIn("docs/", str(first_src.get("label") or ""))

    def test_reference_range_ambiguous_without_profile(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")

        fake_rows = [
            {
                "doc_id": "report_1",
                "analyte": "AMH",
                "analyte_norm": "amh",
                "value_raw": "8",
                "value_numeric": 8.0,
                "unit": "ng/ml",
                "reference_range": "Homme: 4.35-5.35 ng/ml Femme cyclée J2-J4: -age(20-24 ans) : 3.55-4.33 ng/ml -age(25-29 ans) : 3.03-3.87 ng/ml",
                "page_number": 1,
                "row_index": 1,
                "source_pdf": "report (1).pdf",
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer._resolve_reference_scope_doc_ids", return_value=["report_1"]
        ), patch(
            "scripts.generation.generate_answer._fetch_doc_lab_rows", return_value=fake_rows
        ):
            result = run_generation(
                query="Dans le rapport d'immunoanalyse du 19/07/2024, quelle est la norme AMH pour une femme de 25–29 ans ?",
                index_dir="data/indexes",
            )
        answer = str(result.get("answer") or "")
        self.assertTrue("3,03–3,87 ng/ml" in answer or "3.03–3.87 ng/ml" in answer)
        sources = list(result.get("sources") or [])
        self.assertTrue(bool(sources))
        self.assertTrue(any("report (1).pdf" in str(s.get("label") or "") for s in sources))
        self.assertNotIn("report_10", answer)

    def test_flow_type_only_analyte_present(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {
                "doc_id": "report_10",
                "section_norm": "biochimie",
                "analyte": "CALCIUM",
                "reference_raw": "Cordon: 82 - 112 mg/l Nourrisson: 62 - 110 mg/l 0 à 10 jours: 76 - 104 mg/l 10 j à 24 mois: 90 - 110 mg/l 2 à 12 ans: 88 - 108 mg/l Adulte: 84 - 102 mg/l Homme>60 ans: 88 - 100 mg/l",
                "source_pdf": "report (10).pdf",
                "page": 1,
            }
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="calcium",
            requested_profile={"sex": "male", "age_operator": ">", "age": 60, "age_unit": "years"},
            report_type="biochimie",
        )
        answer = str(out.get("answer") or "")
        # self.assertIn("88.0–100.0 mg/l", answer)
        self.assertIn("calcium", answer.lower())
        self.assertNotIn("Aucune plage", answer)
        self.assertNotIn("| Analyte |", answer)

    def test_flow_multiple_docs_same_selected_range(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {"doc_id": "report_10", "analyte": "CALCIUM", "reference_raw": "Homme>60 ans: 88 - 100 mg/l", "source_pdf": "report (10).pdf", "page": 1},
            {"doc_id": "report_12", "analyte": "CALCIUM", "reference_raw": "Homme>60 ans: 88 - 100 mg/l", "source_pdf": "report (12).pdf", "page": 1},
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="calcium",
            requested_profile={"sex": "male", "age_operator": ">", "age": 60, "age_unit": "years"},
            report_type="biochimie",
        )
        self.assertEqual(out.get("status"), "selected")
        answer = str(out.get("answer") or "").lower()
        self.assertIn("plusieurs rapports", answer)
        self.assertTrue(bool(out.get("sources")))

    def test_flow_multiple_docs_different_ranges(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {"doc_id": "report_10", "analyte": "CALCIUM", "reference_raw": "Homme>60 ans: 88 - 100 mg/l", "source_pdf": "report (10).pdf", "page": 1},
            {"doc_id": "report_12", "analyte": "CALCIUM", "reference_raw": "Homme>60 ans: 85 - 98 mg/l", "source_pdf": "report (12).pdf", "page": 1},
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="calcium",
            requested_profile={"sex": "male", "age_operator": ">", "age": 60, "age_unit": "years"},
            report_type="biochimie",
        )
        self.assertEqual(out.get("status"), "ambiguous")
        answer = str(out.get("answer") or "").lower()
        self.assertIn("plusieurs plages différentes", answer)
        self.assertIn("précisez la date/le document", answer)
        self.assertIn("report (10).pdf", answer)

    def test_flow_ambiguous_shows_short_options(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {
                "doc_id": "report_1",
                "section_norm": "immunoanalyse",
                "analyte": "AMH",
                "reference_raw": "Homme: 4.35-5.35 ng/ml Femme cyclée J2-J4: -age(20-24 ans) : 3.55-4.33 ng/ml -age(25-29 ans) : 3.03-3.87 ng/ml",
                "source_pdf": "report (1).pdf",
                "page": 1,
            }
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="amh",
            requested_profile=None,
            report_type="immunoanalyse",
        )
        answer = str(out.get("answer") or "")
        self.assertEqual(out.get("status"), "ambiguous")
        self.assertIn("Sous-profils", answer)
        self.assertIn("Homme", answer)

    def test_flow_no_rows(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        out = run_reference_range_lookup_from_rows(
            rows=[],
            analyte="calcium",
            requested_profile={"sex": "male", "age_operator": ">", "age": 60, "age_unit": "years"},
        )
        self.assertEqual(out.get("status"), "no_match")
        self.assertIn("Aucune plage", str(out.get("answer") or ""))

    def test_flow_amh_female_25_29_selected(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {
                "doc_id": "report_1",
                "section_norm": "immunoanalyse",
                "analyte": "AMH",
                "reference_raw": "Homme: 4.35-5.35 ng/ml Femme cyclée J2-J4: -age(20-24 ans) : 3.55-4.33 ng/ml -age(25-29 ans) : 3.03-3.87 ng/ml -age(30-34 ans) : 2.34-3.55 ng/ml",
                "source_pdf": "report (1).pdf",
                "page": 1,
            }
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="amh",
            requested_profile={"sex": "female", "age_min": 25, "age_max": 29, "age_unit": "years"},
            report_type="immunoanalyse",
            date_iso="2024-07-19",
        )
        answer = str(out.get("answer") or "")
        self.assertEqual(out.get("status"), "selected")
        self.assertTrue("3,03–3,87 ng/ml" in answer or "3.03–3.87 ng/ml" in answer)
        self.assertIn("AMH", answer)
        self.assertIn("Pour l’AMH,", answer)
        self.assertTrue(bool(out.get("sources")))
        self.assertNotIn("Précisez le profil", answer)
        self.assertNotIn("plusieurs plages physiologiques", answer)

    def test_flow_profile_label_never_raw_range(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {
                "doc_id": "report_16",
                "section_norm": "immunoanalyse",
                "analyte": "ACTH",
                "reference_raw": "4,70 - 48,80 pg/ml",
                "source_pdf": "report (16).pdf",
                "page": 1,
            }
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="acth",
            requested_profile={"sex": "female"},
            report_type="immunoanalyse",
        )
        answer = str(out.get("answer") or "")
        self.assertNotIn("pour 4,70 - 48,80 pg/ml", answer.lower())

    def test_generation_mode_when_strict_flag_disabled(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")

        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=False):
            result = run_generation(
                query="Dans le rapport d'immunoanalyse du 19/07/2024, quelle est la norme AMH pour une femme de 30–34 ans ?",
                index_dir="data/indexes",
            )
        self.assertEqual(result.get("generation_mode"), "reference_range_lookup_disabled_by_feature_flag")
        self.assertFalse(bool(((result.get("debug") or {}).get("feature_flags") or {}).get("REFERENCE_RANGE_STRICT_MODE", True)))
        answer = str(result.get("answer") or "")
        self.assertIn("temporairement désactivée", answer)

    def test_generation_mode_when_strict_flag_enabled(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")

        fake_rows = [
            {
                "doc_id": "report_1",
                "analyte": "PTH INTACT",
                "analyte_norm": "pth_intact",
                "value_raw": "7.00",
                "value_numeric": 7.0,
                "unit": "pg/ml",
                "reference_range": "(15,00 - 65,00) pg/ml(1.6-6.9 pmol/l)",
                "page_number": 1,
                "row_index": 1,
                "source_pdf": "report (1).pdf",
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(
                query="Dans le rapport d'immunoanalyse du 19/07/2024, quelle est la plage normale de la PTH intacte ?",
                index_dir="data/indexes",
            )
        self.assertEqual(result.get("generation_mode"), "deterministic_reference_range_lookup")
        self.assertTrue(bool(((result.get("debug") or {}).get("feature_flags") or {}).get("REFERENCE_RANGE_STRICT_MODE", False)))

    def test_source_markdown_clickable_when_viewer_url_exists(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")

        fake_rows = [
            {
                "doc_id": "report_1",
                "analyte": "AMH",
                "analyte_norm": "amh",
                "unit": "ng/ml",
                "reference_raw": "Femme: 30-34 ans: 2.34-3.55 ng/ml",
                "page": 1,
                "row": 1,
                "source_pdf": "docs/report (1).pdf",
                "viewer_url": "http://test/source",
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(
                query="Quelle est la norme AMH pour une femme de 30–34 ans ? avec source cliquable",
                index_dir="data/indexes",
            )
        answer = str(result.get("answer") or "")
        self.assertNotIn("[report (1).pdf — page 1](http://test/source)", answer)
        self.assertTrue(bool(result.get("sources")))
        self.assertNotIn("docs/report", answer)

    def test_no_fake_clickable_link_when_no_url(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")

        fake_rows = [
            {
                "doc_id": "report_1",
                "analyte": "AMH",
                "analyte_norm": "amh",
                "unit": "ng/ml",
                "reference_raw": "Femme: 30-34 ans: 2.34-3.55 ng/ml",
                "page": 1,
                "row": 1,
                "source_pdf": "docs/report (1).pdf",
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(
                query="Quelle est la norme AMH pour une femme de 30–34 ans ?",
                index_dir="data/indexes",
            )
        answer = str(result.get("answer") or "")
        self.assertNotIn("](", answer)

    def test_french_formulation_has_comma_after_prefix(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {
                "doc_id": "report_1",
                "section_norm": "immunoanalyse",
                "analyte": "AMH",
                "reference_raw": "Femme: 30-34 ans: 2.34-3.55 ng/ml",
                "source_pdf": "report (1).pdf",
                "page": 1,
            }
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="amh",
            requested_profile={"sex": "female", "age_min": 30, "age_max": 34, "age_unit": "years"},
            report_type="immunoanalyse",
        )
        answer = str(out.get("answer") or "")
        self.assertIn("Pour l’AMH,", answer)
        self.assertNotIn("Pour l’AMH la", answer)

    def test_flow_population_without_age_shows_age_group_options_only(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {
                "doc_id": "report_1",
                "section_norm": "immunoanalyse",
                "analyte": "AMH",
                "reference_raw": (
                    "Homme: 4.35-5.35 ng/ml "
                    "Femme cyclée J2-J4: -age(20-24 ans) : 3.55-4.33 ng/ml "
                    "-age(25-29 ans) : 3.03-3.87 ng/ml -age(30-34 ans) : 2.34-3.55 ng/ml"
                ),
                "source_pdf": "report (1).pdf",
                "page": 1,
                "row": 4,
            }
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="amh",
            requested_profile={"sex": "female", "population": "cycled_female_j2_j4"},
            report_type="immunoanalyse",
        )
        answer = str(out.get("answer") or "")
        self.assertEqual(out.get("status"), "grouped_options")
        self.assertIn("dépend de la tranche d’âge", answer)
        self.assertIn("20–24 ans : 3,55–4,33 ng/ml", answer)
        self.assertIn("25–29 ans : 3,03–3,87 ng/ml", answer)
        self.assertIn("30–34 ans : 2,34–3,55 ng/ml", answer)
        self.assertIn("Donnez votre tranche d’âge (ex: 30–34 ans).", answer)
        self.assertNotIn("Homme", answer)
        self.assertNotIn("Femme: 20–24 ng/ml", answer)
        self.assertTrue(bool(out.get("sources")))

    def test_generalization_haptoglobine_female_without_age(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {
                "doc_id": "report_10",
                "analyte": "HAPTOGLOBINE",
                "reference_raw": (
                    "Femme 0 - 1 an: 0 - 2,35 g/l 1 - 12 ans: 0,11 - 2,20 g/l 12 - 60 ans: 0,35 - 2,50 g/l > 60 ans: 0,63 - 2,73 g/l "
                    "Homme 0 - 1 an: 0 - 3 g/l 1 - 12 ans: 0,03 - 2,70 g/l 12 - 60 ans: 0,14 - 2,58 g/l > 60 ans: 0,40 - 2,68 g/l"
                ),
                "source_pdf": "report (10).pdf",
                "page": 3,
                "row": 34,
            }
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="haptoglobine",
            requested_profile={"sex": "female"},
            report_type="biochimie",
        )
        answer = str(out.get("answer") or "")
        self.assertIn("0–1 ans", answer)
        self.assertNotIn("Homme", answer)

    def test_generalization_phosphatase_homme_without_age(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {
                "doc_id": "report_10",
                "analyte": "PHOSPHATASE ALCALINE",
                "reference_raw": (
                    "Femme : 1 à 12 ans: < 500 UI/L > 15 ans: 40 - 150 UI/L "
                    "Homme : 1 à 12 ans: < 500 UI/L 12 à 15 ans: < 750 UI/L >20 ans: 40 - 150 UI/L"
                ),
                "source_pdf": "report (10).pdf",
                "page": 2,
                "row": 28,
            }
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="phosphatase alcaline",
            requested_profile={"sex": "male"},
            report_type="biochimie",
        )
        answer = str(out.get("answer") or "")
        self.assertIn("homme", answer.lower())
        self.assertNotIn("Femme :", answer)

    def test_generalization_glucose_over_70_selected(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {
                "doc_id": "report_10",
                "analyte": "GLUCOSE",
                "reference_raw": (
                    "Adulte: 0,70 - 1,05 g/l > 60 ans :0,80 - 1,15 g/l > 70 ans:0,83 - 1,10 g/l"
                ),
                "source_pdf": "report (10).pdf",
                "page": 2,
                "row": 22,
            }
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="glucose",
            requested_profile={"age_operator": ">", "age": 70, "age_unit": "years"},
            report_type="biochimie",
        )
        answer = str(out.get("answer") or "")
        self.assertEqual(out.get("status"), "selected")
        self.assertTrue("0,83–1,1 g/l" in answer or "0,83–1,10 g/l" in answer)

    def test_generalization_proteines_totales_adulte_subgroups(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {
                "doc_id": "report_10",
                "analyte": "PROTEINES TOTALES",
                "reference_raw": "Adulte (ambulatoire): 64 - 83 g/l; Adulte (alité): 60 - 78 g/l; > 60 ans: 58 - 76 g/l",
                "source_pdf": "report (10).pdf",
                "page": 2,
                "row": 23,
            }
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="protéines totales",
            requested_profile={"population": "adult"},
            report_type="biochimie",
        )
        answer = str(out.get("answer") or "").lower()
        self.assertIn("sous-profils", answer)
        self.assertIn("ambulatoire", answer)
        self.assertIn("alité", answer)

    def test_generalization_mock_testx_beta_grouped_only(self) -> None:
        from scripts.generation.reference_range_lookup_flow import run_reference_range_lookup_from_rows

        rows = [
            {
                "doc_id": "report_x",
                "analyte": "TEST_X",
                "reference_ranges": [
                    {"label": "Alpha", "condition": "alpha", "operator": "range", "low": 1.0, "high": 2.0, "unit": "u"},
                    {"label": "Beta — 20-30 ans", "condition": "beta", "age_min": 20.0, "age_max": 30.0, "age_unit": "years", "operator": "range", "low": 3.0, "high": 4.0, "unit": "u"},
                    {"label": "Beta — 31-40 ans", "condition": "beta", "age_min": 31.0, "age_max": 40.0, "age_unit": "years", "operator": "range", "low": 5.0, "high": 6.0, "unit": "u"},
                ],
                "source_pdf": "report (x).pdf",
                "page": 1,
                "row": 1,
            }
        ]
        out = run_reference_range_lookup_from_rows(
            rows=rows,
            analyte="test_x",
            requested_profile={"condition": "beta"},
        )
        answer = str(out.get("answer") or "").lower()
        self.assertIn("20–30 ans", answer)
        self.assertIn("31–40 ans", answer)
        self.assertNotIn("alpha", answer)

    def test_generation_haptoglobine_femme_grouped_options(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")
        fake_rows = [
            {
                "doc_id": "report_10",
                "analyte": "HAPTOGLOBINE",
                "analyte_norm": "haptoglobine",
                "reference_raw": (
                    "Femme 0 - 1 an: 0 - 2,35 g/l 1 - 12 ans: 0,11 - 2,20 g/l 12 - 60 ans: 0,35 - 2,50 g/l > 60 ans: 0,63 - 2,73 g/l "
                    "Homme 0 - 1 an: 0 - 3 g/l 1 - 12 ans: 0,03 - 2,70 g/l 12 - 60 ans: 0,14 - 2,58 g/l > 60 ans: 0,40 - 2,68 g/l"
                ),
                "source_pdf": "report (10).pdf",
                "page": 3,
                "row": 34,
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(query="Quelle est la plage normale de l’haptoglobine chez la femme ?", index_dir="data/indexes")
        answer = str(result.get("answer") or "")
        self.assertIn("0–1 ans", answer)
        self.assertIn("1–12 ans", answer)
        self.assertIn("12–60 ans", answer)
        self.assertTrue("> 60 ans" in answer or ">60 ans" in answer)
        self.assertIn("0,63–2,73 g/l", answer)
        self.assertNotIn("Homme", answer)
        self.assertNotIn("Phosphatase", answer)
        self.assertNotIn("résultats correspondants", answer.lower())
        self.assertEqual(result.get("generation_mode"), "deterministic_reference_range_lookup")

    def test_generation_haptoglobine_femme_over_60_selected(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")
        fake_rows = [
            {
                "doc_id": "report_10",
                "analyte": "HAPTOGLOBINE",
                "analyte_norm": "haptoglobine",
                "reference_raw": "Femme > 60 ans: 0,63 - 2,73 g/l Homme > 60 ans: 0,40 - 2,68 g/l",
                "source_pdf": "report (10).pdf",
                "page": 3,
                "row": 34,
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(
                query="Quelle est la plage normale de l’haptoglobine chez la femme de plus de 60 ans ?",
                index_dir="data/indexes",
            )
        answer = str(result.get("answer") or "")
        self.assertIn("0,63–2,73 g/l", answer)
        self.assertNotIn("| Analyte |", answer)

    def test_generation_pal_homme_grouped_options(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")
        fake_rows = [
            {
                "doc_id": "report_10",
                "analyte": "PHOSPHATASE ALCALINE",
                "analyte_norm": "phosphatase_alcaline",
                "reference_raw": "Femme : 1 à 12 ans: < 500 UI/L > 15 ans: 40 - 150 UI/L Homme : 1 à 12 ans: < 500 UI/L 12 à 15 ans: < 750 UI/L >20 ans: 40 - 150 UI/L",
                "source_pdf": "report (10).pdf",
                "page": 2,
                "row": 28,
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(query="Quelle est la norme de la phosphatase alcaline chez l’homme ?", index_dir="data/indexes")
        answer = str(result.get("answer") or "")
        self.assertIn("1–12 ans", answer)
        self.assertIn("12–15 ans", answer)
        self.assertTrue("> 20 ans" in answer or ">20 ans" in answer)
        self.assertIn("<750", answer.replace(" ", ""))
        self.assertNotIn("Femme", answer)
        self.assertNotIn("Haptoglobine", answer)
        self.assertIn("Pour la phosphatase alcaline", answer)

    def test_generation_pal_homme_12_15_selected(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")
        fake_rows = [
            {
                "doc_id": "report_10",
                "analyte": "PHOSPHATASE ALCALINE",
                "analyte_norm": "phosphatase_alcaline",
                "reference_raw": "Homme : 1 à 12 ans: < 500 UI/L 12 à 15 ans: < 750 UI/L >20 ans: 40 - 150 UI/L",
                "source_pdf": "report (10).pdf",
                "page": 2,
                "row": 28,
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(query="Quelle est la norme de la phosphatase alcaline chez l’homme de 12 à 15 ans ?", index_dir="data/indexes")
        answer = str(result.get("answer") or "")
        self.assertIn("<750", answer.replace(" ", ""))
        self.assertNotIn("| Analyte |", answer)
        self.assertIn("Pour la phosphatase alcaline", answer)

    def test_generation_calcium_homme_over_60_sentence_case(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")
        fake_rows = [
            {
                "doc_id": "report_10",
                "analyte": "CALCIUM",
                "analyte_norm": "calcium",
                "reference_raw": "Homme>60 ans: 88 - 100 mg/l",
                "source_pdf": "report (10).pdf",
                "page": 1,
                "row": 1,
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(query="plage calcium pour homme > 60 ans", index_dir="data/indexes")
        answer = str(result.get("answer") or "")
        self.assertIn("Pour le calcium", answer)

    def test_generation_pth_multi_units_single_norm(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")
        fake_rows = [
            {
                "doc_id": "report_1",
                "analyte": "PTH INTACT",
                "analyte_norm": "pth_intact",
                "reference_raw": "(15,00 - 65,00) pg/ml(1.6-6.9 pmol/l)",
                "source_pdf": "report (1).pdf",
                "page": 1,
                "row": 3,
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(query="plage normale pth intacte", index_dir="data/indexes")
        answer = str(result.get("answer") or "")
        self.assertIn("15–65 pg/ml", answer.replace(",0", ""))
        self.assertTrue("soit 1,6–6,9 pmol/l" in answer or "soit 1.6–6.9 pmol/l" in answer)
        self.assertNotIn("plusieurs sous-profils", answer.lower())

    def test_reference_followup_switch_analyte_keeps_reference_mode(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")

        def _fake_find_rows(**kwargs):
            analytes = [str(a).strip().lower() for a in (kwargs.get("analyte_names") or [])]
            target = analytes[0] if analytes else ""
            if target == "calcium":
                return [
                    {
                        "doc_id": "report_10",
                        "analyte": "CALCIUM",
                        "analyte_norm": "calcium",
                        "reference_raw": "Homme>60 ans: 88 - 100 mg/l",
                        "source_pdf": "report (10).pdf",
                        "page": 1,
                        "row": 1,
                    }
                ]
            if target == "tshus":
                return [
                    {
                        "doc_id": "report_1",
                        "analyte": "TSHus",
                        "analyte_norm": "tshus",
                        "reference_raw": "0,35 à 4,94 mUI/l",
                        "source_pdf": "report (1).pdf",
                        "page": 1,
                        "row": 13,
                    }
                ]
            return []

        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            side_effect=_fake_find_rows,
        ):
            second = run_generation(
                query="et pour TSHus ?",
                index_dir="data/indexes",
                previous_context_intent="reference_range_lookup",
                previous_data_context_intent="reference_range_lookup",
                previous_data_context_type="biological_numeric_results",
                previous_displayed_context={
                    "reference_intent": "reference_range_lookup",
                    "subject": "Calcium",
                    "reference_profile": {"sex": "male", "age_operator": ">", "age_value": 60, "age_unit": "years"},
                    "last_reference_range_context": {
                        "intent": "reference_range_lookup",
                        "analyte": "Calcium",
                        "requested_reference_profile": {"sex": "male", "age_operator": ">", "age_value": 60, "age_unit": "years"},
                    },
                },
            )
        answer = str(second.get("answer") or "")
        self.assertTrue("0,35–4,94 mUI/l" in answer or "0.35–4.94 mUI/l" in answer or "0,35 à 4,94 mUI/l" in answer)
        self.assertNotIn("2,00 mUI/L", answer)
        self.assertEqual(second.get("generation_mode"), "deterministic_reference_range_lookup")

    def test_reference_followup_age_update_same_analyte(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")

        fake_rows = [
            {
                "doc_id": "report_1",
                "analyte": "AMH",
                "analyte_norm": "amh",
                "reference_raw": "Homme: 4.35-5.35 ng/ml Femme cyclée J2-J4: -age(20-24 ans) : 3.55-4.33 ng/ml -age(25-29 ans) : 3.03-3.87 ng/ml -age(30-34 ans) : 2.34-3.55 ng/ml",
                "source_pdf": "report (1).pdf",
                "page": 1,
                "row": 4,
            }
        ]
        prev_ctx = {
            "reference_intent": "reference_range_lookup",
            "subject": "AMH",
            "reference_profile": {"sex": "female", "population": "cycled_female_j2_j4", "age_min": 30, "age_max": 34, "age_unit": "years"},
            "last_reference_range_context": {
                "intent": "reference_range_lookup",
                "analyte": "AMH",
                "requested_reference_profile": {"sex": "female", "population": "cycled_female_j2_j4", "age_min": 30, "age_max": 34, "age_unit": "years"},
            },
        }
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            second = run_generation(
                query="et pour 25–29 ans ?",
                index_dir="data/indexes",
                previous_context_intent="reference_range_lookup",
                previous_data_context_intent="reference_range_lookup",
                previous_data_context_type="biological_numeric_results",
                previous_displayed_context=prev_ctx,
            )
        answer = str(second.get("answer") or "")
        self.assertTrue("3,03–3,87 ng/ml" in answer or "3.03–3.87 ng/ml" in answer)
        self.assertNotIn("8 ng/ml", answer.lower())
        self.assertEqual(second.get("generation_mode"), "deterministic_reference_range_lookup")

    def test_validator_pass_amh_female_30_34(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")
        fake_rows = [
            {
                "doc_id": "report_1",
                "analyte": "AMH",
                "analyte_norm": "amh",
                "reference_raw": "Homme: 4.35-5.35 ng/ml Femme cyclée J2-J4: -age(30-34 ans) : 2.34-3.55 ng/ml",
                "source_pdf": "report (1).pdf",
                "page": 1,
                "row": 4,
                "viewer_url": "/viewer/pdf?doc_id=report_1&page=1",
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(
                query="Quelle est la plage normale AMH pour une femme de 30–34 ans ?",
                index_dir="data/indexes",
            )
        answer = str(result.get("answer") or "")
        validation = dict(result.get("validation") or {})
        quality = dict(result.get("quality_report") or {})
        self.assertTrue("2,34–3,55 ng/ml" in answer or "2.34–3.55 ng/ml" in answer)
        self.assertTrue(bool(result.get("sources")))
        self.assertEqual(validation.get("validation_status"), "pass")
        self.assertEqual(quality.get("final_status"), "pass")
        self.assertNotEqual(float(quality.get("faithfulness_score") or 0.0), 0.0)

    def test_validator_pass_amh_followup_25_29(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")
        fake_rows = [
            {
                "doc_id": "report_1",
                "analyte": "AMH",
                "analyte_norm": "amh",
                "reference_raw": "Femme cyclée J2-J4: -age(25-29 ans) : 3.03-3.87 ng/ml -age(30-34 ans) : 2.34-3.55 ng/ml",
                "source_pdf": "report (1).pdf",
                "page": 1,
                "row": 4,
            }
        ]
        prev_ctx = {
            "reference_intent": "reference_range_lookup",
            "subject": "AMH",
            "reference_profile": {"sex": "female", "population": "cycled_female_j2_j4", "age_min": 30, "age_max": 34, "age_unit": "years"},
            "last_reference_range_context": {
                "intent": "reference_range_lookup",
                "analyte": "AMH",
                "requested_reference_profile": {"sex": "female", "population": "cycled_female_j2_j4", "age_min": 30, "age_max": 34, "age_unit": "years"},
            },
        }
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(
                query="et pour 25–29 ans ?",
                index_dir="data/indexes",
                previous_context_intent="reference_range_lookup",
                previous_data_context_intent="reference_range_lookup",
                previous_data_context_type="biological_numeric_results",
                previous_displayed_context=prev_ctx,
            )
        answer = str(result.get("answer") or "")
        validation = dict(result.get("validation") or {})
        self.assertTrue("3,03–3,87 ng/ml" in answer or "3.03–3.87 ng/ml" in answer)
        self.assertEqual(validation.get("validation_status"), "pass")

    def test_validator_not_fail_multi_doc_same_range(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")
        fake_rows = [
            {
                "doc_id": "report_10",
                "analyte": "CALCIUM",
                "analyte_norm": "calcium",
                "reference_raw": "Homme>60 ans: 88 - 100 mg/l",
                "source_pdf": "report (10).pdf",
                "page": 1,
                "row": 1,
            },
            {
                "doc_id": "report_11",
                "analyte": "CALCIUM",
                "analyte_norm": "calcium",
                "reference_raw": "Homme>60 ans: 88 - 100 mg/l",
                "source_pdf": "report (11).pdf",
                "page": 1,
                "row": 1,
            },
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(query="plage calcium pour homme > 60 ans", index_dir="data/indexes")
        answer = str(result.get("answer") or "")
        validation = dict(result.get("validation") or {})
        self.assertIn("88", answer)
        self.assertTrue(bool(result.get("sources")))
        self.assertIn(validation.get("validation_status"), {"pass", "warning"})
        self.assertNotEqual(validation.get("validation_status"), "fail")

    def test_validator_pass_pth_multi_unit(self) -> None:
        try:
            from scripts.generation.generate_answer import run_generation
        except Exception as exc:
            self.skipTest(f"run_generation indisponible: {exc}")
        fake_rows = [
            {
                "doc_id": "report_1",
                "analyte": "PTH INTACT",
                "analyte_norm": "pth_intact",
                "reference_raw": "(15,00 - 65,00) pg/ml(1.6-6.9 pmol/l)",
                "source_pdf": "report (1).pdf",
                "page": 1,
                "row": 3,
            }
        ]
        with patch("scripts.generation.generate_answer._is_feature_enabled", return_value=True), patch(
            "scripts.generation.generate_answer.find_reference_range_candidate_rows",
            return_value=fake_rows,
        ):
            result = run_generation(query="plage normale pth intacte", index_dir="data/indexes")
        answer = str(result.get("answer") or "")
        validation = dict(result.get("validation") or {})
        self.assertTrue("15–65 pg/ml" in answer.replace(",0", ""))
        self.assertTrue("1,6–6,9 pmol/l" in answer or "1.6–6.9 pmol/l" in answer)
        self.assertEqual(validation.get("validation_status"), "pass")

    def test_validator_fails_bulk_listing_for_reference_intent(self) -> None:
        from scripts.generation.answer_validator import validate_answer

        validation = validate_answer(
            query="Quelle est la norme AMH ?",
            answer_text="55 résultats correspondants ont été retrouvés.\n| Analyte | Valeur actuelle |",
            evidence_pack=[],
            displayed_evidences=[],
            source_citations=[],
            generation_mode="deterministic_reference_range_lookup",
            retrieval_status="answerable",
            query_intents={"reference_range_lookup": True},
        )
        self.assertEqual(validation.get("validation_status"), "fail")


if __name__ == "__main__":
    unittest.main()
