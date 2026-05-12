from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_ROOT = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATION_ROOT))

from answer_validator import validate_answer
from generate_answer import build_visualization_payload, compute_reference_metric


class TestVisualizationPayload(unittest.TestCase):
    def _mixed_evidence(self) -> list[dict]:
        return [
            {
                "analyte": "ACTH",
                "analyte_norm": "acth",
                "current_value": "23,00",
                "unit": "pg/ml",
                "reference": "4,70 - 48,80 pg/ml",
                "technical_status": "dans la référence",
            },
            {
                "analyte": "TSHus",
                "analyte_norm": "tshus",
                "current_value": "55,00",
                "unit": "mUI/L",
                "reference": "0,35 - 4,94 mUI/L",
                "technical_status": "au-dessus de la référence",
            },
        ]

    def _report16_out_of_range_evidence(self) -> list[dict]:
        return [
            {
                "doc_id": "report_16",
                "analyte": "INSULINE",
                "analyte_norm": "insuline",
                "current_value": "2,00",
                "unit": "uU/mL",
                "reference": "4 à 20 µIU/mL",
                "technical_status": "en dessous de la référence",
                "technical_status_code": "below_reference",
                "source_label": "report (16).pdf — page 1, ligne 2",
            },
            {
                "doc_id": "report_16",
                "analyte": "ANTI-TG",
                "analyte_norm": "anti_tg",
                "current_value": "77,00",
                "unit": "IU/ml",
                "reference": "<4,11 IU/ml",
                "technical_status": "au-dessus de la référence",
                "technical_status_code": "above_reference",
                "source_label": "report (16).pdf — page 1, ligne 6",
            },
            {
                "doc_id": "report_16",
                "analyte": "TSHus",
                "analyte_norm": "tshus",
                "current_value": "55,00",
                "unit": "mUI/L",
                "reference": "0,35 à 4,94 mUI/l",
                "technical_status": "au-dessus de la référence",
                "technical_status_code": "above_reference",
                "source_label": "report (16).pdf — page 1, ligne 4",
            },
        ]

    def test_00_reference_metric_interval(self) -> None:
        metric = compute_reference_metric("55", "0,35 à 4,94 mUI/l", "above_reference")
        self.assertAlmostEqual(float(metric.get("lower_bound") or 0), 0.35, places=2)
        self.assertAlmostEqual(float(metric.get("upper_bound") or 0), 4.94, places=2)
        self.assertTrue(metric.get("metric_available"))
        self.assertAlmostEqual(float(metric.get("reference_deviation") or 0), (55.0 / 4.94) - 1.0, places=3)

    def test_00b_reference_metric_upper_threshold(self) -> None:
        metric = compute_reference_metric("77", "<4,11 IU/ml", "above_reference")
        self.assertIsNone(metric.get("lower_bound"))
        self.assertAlmostEqual(float(metric.get("upper_bound") or 0), 4.11, places=2)
        self.assertTrue(metric.get("metric_available"))
        self.assertAlmostEqual(float(metric.get("reference_deviation") or 0), (77.0 / 4.11) - 1.0, places=3)

    def test_00c_reference_metric_below_is_negative(self) -> None:
        metric = compute_reference_metric("2", "4 à 20", "below_reference")
        self.assertTrue(metric.get("metric_available"))
        self.assertAlmostEqual(float(metric.get("reference_deviation") or 0), -0.5, places=3)
        self.assertLess(float(metric.get("reference_deviation") or 0), 0.0)

    def test_00d_reference_metric_lower_threshold(self) -> None:
        metric = compute_reference_metric("12", ">10", "above_reference")
        self.assertAlmostEqual(float(metric.get("lower_bound") or 0), 10.0, places=2)
        self.assertIsNone(metric.get("upper_bound"))
        self.assertTrue(metric.get("metric_available"))
        self.assertAlmostEqual(float(metric.get("reference_deviation") or 0), 0.2, places=3)

    def test_00e_reference_metric_not_interpretable(self) -> None:
        metric = compute_reference_metric("12", "Qualitatif", "not_interpretable")
        self.assertFalse(metric.get("metric_available"))
        self.assertIsNone(metric.get("reference_deviation"))

    def test_01_bar_supported(self) -> None:
        payload = build_visualization_payload(
            requested_type="bar",
            evidence_pack=self._mixed_evidence(),
            supported_visualizations=["bar", "line"],
            raw_format_phrase="graphique en barres",
        )
        self.assertEqual(payload.get("requested_type"), "bar")
        self.assertEqual(payload.get("rendered_type"), "bar")
        self.assertTrue(payload.get("supported"))
        self.assertFalse(payload.get("fallback_used"))

    def test_02_radar_not_supported_uses_fallback(self) -> None:
        payload = build_visualization_payload(
            requested_type="radar",
            evidence_pack=self._mixed_evidence(),
            supported_visualizations=["bar", "line"],
            raw_format_phrase="radar chart",
        )
        self.assertEqual(payload.get("requested_type"), "radar")
        self.assertTrue(payload.get("fallback_used"))
        self.assertEqual(payload.get("rendered_type"), "bar")
        self.assertFalse(payload.get("supported"))
        self.assertTrue(str(payload.get("fallback_reason") or ""))

    def test_03_line_unsuitable_falls_back_to_bar(self) -> None:
        payload = build_visualization_payload(
            requested_type="line",
            evidence_pack=self._mixed_evidence(),
            supported_visualizations=["bar", "line"],
            raw_format_phrase="Arithmetic Line-Graph",
        )
        self.assertEqual(payload.get("requested_type"), "line")
        self.assertTrue(payload.get("fallback_used"))
        self.assertEqual(payload.get("rendered_type"), "bar")
        self.assertFalse(payload.get("suitable"))

    def test_04_unknown_format_preserves_fallback(self) -> None:
        payload = build_visualization_payload(
            requested_type="unknown",
            evidence_pack=self._mixed_evidence(),
            supported_visualizations=["bar", "line"],
            raw_format_phrase="bio-clinical matrix radar comparative",
        )
        self.assertEqual(payload.get("requested_type"), "unknown")
        self.assertTrue(payload.get("fallback_used"))
        self.assertIn("bio-clinical", str(payload.get("fallback_reason") or "").lower())

    def test_05_validator_internal_term_visible(self) -> None:
        validation = validate_answer(
            query="Donne-moi les résultats sous forme radar chart.",
            answer_text="Vous avez demandé un graphique radar. rendu chart interne.",
            evidence_pack=self._mixed_evidence(),
            displayed_evidences=self._mixed_evidence(),
            source_citations=[],
            output_format_requested="chart",
            user_requested_visualization=True,
            requested_chart_type="radar",
            visualization_payload=build_visualization_payload(
                requested_type="radar",
                evidence_pack=self._mixed_evidence(),
                supported_visualizations=["bar", "line"],
                raw_format_phrase="radar chart",
            ),
        )
        self.assertTrue(
            "render_internal_term_leak" in (validation.get("errors") or [])
            or "internal_chart_term_visible" in (validation.get("errors") or [])
        )

    def test_06_validator_no_code_execution(self) -> None:
        validation = validate_answer(
            query="Génère le graphique en HTML Chart.js.",
            answer_text="Je peux générer un script JavaScript Chart.js à exécuter en HTML.",
            evidence_pack=self._mixed_evidence(),
            displayed_evidences=self._mixed_evidence(),
            source_citations=[],
            output_format_requested="chart",
            user_requested_visualization=True,
            requested_chart_type="bar",
            visualization_payload=build_visualization_payload(
                requested_type="bar",
                evidence_pack=self._mixed_evidence(),
                supported_visualizations=["bar", "line"],
                raw_format_phrase="graphique en barres",
            ),
        )
        self.assertIn("no_code_execution", validation.get("errors") or [])

    def test_07_validator_requires_fallback_reason_and_alternative(self) -> None:
        viz = build_visualization_payload(
            requested_type="radar",
            evidence_pack=self._mixed_evidence(),
            supported_visualizations=["bar", "line"],
            raw_format_phrase="radar chart",
        )
        validation = validate_answer(
            query="Donne-moi les résultats sous forme radar chart.",
            answer_text="Vous avez demandé un graphique radar. Je garde les données.",
            evidence_pack=self._mixed_evidence(),
            displayed_evidences=self._mixed_evidence(),
            source_citations=[],
            output_format_requested="chart",
            user_requested_visualization=True,
            requested_chart_type="radar",
            visualization_payload=viz,
        )
        self.assertIn("fallback_alternative_not_mentioned", validation.get("errors") or [])
        self.assertIn("fallback_reason_missing_in_answer", validation.get("errors") or [])

    def test_08_chart_data_uses_reference_deviation_report16(self) -> None:
        payload = build_visualization_payload(
            requested_type="bar",
            evidence_pack=self._report16_out_of_range_evidence(),
            supported_visualizations=["bar", "line"],
            raw_format_phrase="graphique en barres",
        )
        self.assertEqual(payload.get("y_field"), "reference_deviation")
        self.assertNotEqual(str(payload.get("metric_label") or "").lower(), "reference_ratio")
        data = list(payload.get("data") or [])
        self.assertTrue(data)
        by_analyte = {str(row.get("analyte") or "").lower(): row for row in data}
        insuline = next((row for key, row in by_analyte.items() if "insuline" in key), None)
        antitg = next((row for key, row in by_analyte.items() if "anti" in key and "tg" in key), None)
        self.assertIsNotNone(insuline)
        self.assertIsNotNone(antitg)
        self.assertLess(float((insuline or {}).get("reference_deviation") or 0), 0.0)
        self.assertTrue(bool((antitg or {}).get("metric_available")))
        self.assertTrue(isinstance((antitg or {}).get("reference_deviation"), (int, float)))
        for row in data:
            self.assertIn("raw_value", row)
            self.assertIn("unit", row)
            self.assertIn("reference", row)
            self.assertIn("status", row)

    def test_09_validator_detects_bad_metric_label_and_generic_conclusion(self) -> None:
        payload = build_visualization_payload(
            requested_type="bar",
            evidence_pack=self._report16_out_of_range_evidence(),
            supported_visualizations=["bar", "line"],
            raw_format_phrase="graphique en barres",
        )
        payload["y_field"] = "reference_ratio"
        validation = validate_answer(
            query="Dans report 16, affiche les résultats hors référence sous forme graphique en barres.",
            answer_text=(
                "Voici le graphique. reference_ratio affiché.\n\n"
                "Conclusion technique : données structurées fournies pour visualisation côté interface."
            ),
            evidence_pack=self._report16_out_of_range_evidence(),
            displayed_evidences=self._report16_out_of_range_evidence(),
            source_citations=[],
            output_format_requested="chart",
            user_requested_visualization=True,
            requested_chart_type="bar",
            visualization_payload=payload,
        )
        self.assertIn("bad_metric_label", validation.get("errors") or [])
        self.assertIn("generic_conclusion", validation.get("warnings") or [])


if __name__ == "__main__":
    unittest.main()
