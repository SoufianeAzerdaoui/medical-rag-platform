"""
tests/test_renderer.py
Unit tests for ClinicalDeterministicRenderer (Phase 3 — Option A).

Coverage:
- Compact mode: header, summary, key rows, conclusion, sources
- Detailed mode: table columns, markdown format, conclusion
- Not-found: with doc_id, without doc_id, exact template tokens
- Edge cases: multi-doc grouping, status mapping, empty evidences,
              missing fields, debug payload isolation
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the generation scripts importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from professional_answer_composer import ClinicalDeterministicRenderer

R = ClinicalDeterministicRenderer()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _row(
    analyte="Créatinine",
    analyte_norm="creatinine",
    current_value="12",
    unit="mg/L",
    status="above_reference",
    reference="5–12 mg/L",
    doc_id="report_24",
    page=1,
) -> dict:
    return {
        "analyte": analyte,
        "analyte_norm": analyte_norm,
        "current_value": current_value,
        "unit": unit,
        "technical_status": status,
        "reference": reference,
        "doc_id": doc_id,
        "page": page,
    }


def _qu(
    doc_ids=None,
    analytes=None,
    question="bilan rénal",
    style="compact",
) -> dict:
    return {
        "requested_doc_ids": doc_ids or [],
        "requested_analytes": analytes or [],
        "original_user_question": question,
        "answer_style": style,
    }


# ---------------------------------------------------------------------------
# 1. Compact mode
# ---------------------------------------------------------------------------

class TestCompactMode(unittest.TestCase):

    def test_header_present(self):
        rows = [_row()]
        result = R.render_compact(rows, _qu(doc_ids=["report_24"], question="bilan rénal"))
        self.assertIn("Bilan demandé", result["text"])
        self.assertIn("report_24", result["text"])

    def test_conclusion_present(self):
        rows = [_row()]
        result = R.render_compact(rows, _qu())
        self.assertIn("Conclusion technique", result["conclusion"])

    def test_compact_at_most_5_content_lines(self):
        """Lines in text (excl. sources block and conclusion) must be ≤ 5."""
        rows = [_row()] * 6  # more than 3
        result = R.render_compact(rows, _qu(doc_ids=["report_24"]))
        # text has header + summary + up-to-3 rows = max 5 lines
        lines = [ln for ln in result["text"].splitlines() if ln.strip()]
        self.assertLessEqual(len(lines), 5)

    def test_anomaly_count_in_summary(self):
        above = _row(status="above_reference")
        within = _row(status="within_reference", analyte="Urée", analyte_norm="uree")
        result = R.render_compact([above, within], _qu())
        self.assertIn("1 valeur(s) hors référence", result["text"])

    def test_no_anomaly_summary(self):
        rows = [_row(status="within_reference")]
        result = R.render_compact(rows, _qu())
        self.assertIn("Aucune anomalie", result["text"])

    def test_sources_populated(self):
        rows = [_row(doc_id="report_24"), _row(doc_id="report_10", analyte="CRP")]
        result = R.render_compact(rows, _qu())
        self.assertGreaterEqual(len(result["sources"]), 1)
        self.assertTrue(any("report_24" in s for s in result["sources"]))

    def test_mode_is_compact(self):
        result = R.render_compact([_row()], _qu())
        self.assertEqual(result["mode"], "compact")

    def test_no_debug_by_default(self):
        result = R.render_compact([_row()], _qu())
        self.assertIsNone(result["debug"])

    def test_debug_present_when_requested(self):
        result = R.render_compact([_row()], _qu(analytes=["creatinine"]), debug=True)
        self.assertIsNotNone(result["debug"])
        self.assertIn("found_count", result["debug"])
        self.assertEqual(result["debug"]["found_count"], 1)


# ---------------------------------------------------------------------------
# 2. Detailed mode
# ---------------------------------------------------------------------------

class TestDetailedMode(unittest.TestCase):

    def test_table_columns_present(self):
        rows = [_row()]
        result = R.render_detailed(rows, _qu())
        self.assertIsNotNone(result["table"])
        row0 = result["table"][0]
        self.assertIn("Analyte", row0)
        self.assertIn("Valeur (unit source)", row0)
        self.assertIn("Statut", row0)
        self.assertIn("Réf concise", row0)
        self.assertIn("Document (doc_id)", row0)

    def test_markdown_table_in_text(self):
        rows = [_row()]
        result = R.render_detailed(rows, _qu())
        self.assertIn("| Analyte |", result["text"])
        self.assertIn("|---------|", result["text"])

    def test_doc_id_in_table(self):
        rows = [_row(doc_id="report_12")]
        result = R.render_detailed(rows, _qu(doc_ids=["report_12"]))
        self.assertEqual(result["table"][0]["Document (doc_id)"], "report_12")

    def test_statut_above_reference(self):
        rows = [_row(status="above_reference")]
        result = R.render_detailed(rows, _qu())
        self.assertEqual(result["table"][0]["Statut"], "au-dessus de la référence")

    def test_statut_below_reference(self):
        rows = [_row(status="below_reference")]
        result = R.render_detailed(rows, _qu())
        self.assertEqual(result["table"][0]["Statut"], "en dessous de la référence")

    def test_statut_within_reference(self):
        rows = [_row(status="within_reference")]
        result = R.render_detailed(rows, _qu())
        self.assertEqual(result["table"][0]["Statut"], "dans la référence")

    def test_ref_concise_max_140(self):
        long_ref = "X" * 200
        rows = [_row(reference=long_ref)]
        result = R.render_detailed(rows, _qu())
        self.assertLessEqual(len(result["table"][0]["Réf concise"]), 140)

    def test_mode_is_detailed(self):
        result = R.render_detailed([_row()], _qu())
        self.assertEqual(result["mode"], "detailed")

    def test_multi_doc_all_rows_present(self):
        r1 = _row(doc_id="report_10", analyte="TSHus")
        r2 = _row(doc_id="report_16", analyte="T3")
        r3 = _row(doc_id="report_24", analyte="Créatinine")
        result = R.render_detailed([r1, r2, r3], _qu(doc_ids=["report_10", "report_16", "report_24"]))
        doc_ids_in_table = {tr["Document (doc_id)"] for tr in result["table"]}
        self.assertEqual(doc_ids_in_table, {"report_10", "report_16", "report_24"})

    def test_conclusion_present(self):
        result = R.render_detailed([_row()], _qu())
        self.assertIn("Conclusion technique", result["conclusion"])

    def test_header_contains_scope(self):
        result = R.render_detailed([_row(doc_id="report_24")], _qu(doc_ids=["report_24"]))
        self.assertIn("report_24", result["text"])


# ---------------------------------------------------------------------------
# 3. Not-found template
# ---------------------------------------------------------------------------

class TestNotFoundMode(unittest.TestCase):

    def test_not_found_with_doc_id(self):
        result = R.render_not_found("cortisol", "report_12")
        self.assertIn("CORTISOL", result["text"])
        self.assertIn("report_12", result["text"])
        self.assertIn("Aucune valeur numérique", result["text"])

    def test_not_found_cta_options(self):
        result = R.render_not_found("cortisol", "report_12")
        self.assertIn("(1)", result["text"])
        self.assertIn("(2)", result["text"])
        self.assertIn("(3)", result["text"])

    def test_not_found_conclusion_contains_analyte(self):
        result = R.render_not_found("cortisol", "report_12")
        self.assertIn("CORTISOL", result["conclusion"])
        self.assertIn("Conclusion technique", result["conclusion"])

    def test_not_found_global_no_doc_id(self):
        result = R.render_not_found("cortisol")
        self.assertIn("CORTISOL", result["text"])
        self.assertNotIn("rapport demandé", result["conclusion"])
        self.assertIn("rapports disponibles", result["conclusion"])

    def test_not_found_mode_field(self):
        result = R.render_not_found("cortisol", "report_12")
        self.assertEqual(result["mode"], "not_found")

    def test_not_found_empty_sources(self):
        result = R.render_not_found("cortisol", "report_12")
        self.assertEqual(result["sources"], [])

    def test_not_found_debug_info_passed_through(self):
        dbg = {"requested_analytes": ["cortisol"], "confidence_score": 0.0}
        result = R.render_not_found("cortisol", "report_12", debug_info=dbg)
        self.assertEqual(result["debug"], dbg)

    def test_not_found_when_empty_evidences_compact(self):
        qu = _qu(doc_ids=["report_12"], analytes=["cortisol"])
        result = R.render_compact([], qu)
        self.assertEqual(result["mode"], "not_found")
        self.assertIn("CORTISOL", result["text"])
        self.assertIn("report_12", result["text"])

    def test_not_found_when_empty_evidences_detailed(self):
        qu = _qu(doc_ids=["report_12"], analytes=["cortisol"])
        result = R.render_detailed([], qu)
        self.assertEqual(result["mode"], "not_found")


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_missing_unit_handled(self):
        row = _row(unit="")
        result = R.render_compact([row], _qu())
        self.assertNotIn("None", result["text"])

    def test_missing_value_shows_dash(self):
        row = {**_row(), "current_value": None, "value_raw": None}
        val = ClinicalDeterministicRenderer._get_value_str(row)
        self.assertEqual(val, "–")

    def test_doc_id_token_exact_format(self):
        """doc_id must appear exactly as provided (underscore form)."""
        row = _row(doc_id="report_12")
        result = R.render_detailed([row], _qu(doc_ids=["report_12"]))
        self.assertIn("report_12", result["table"][0]["Document (doc_id)"])

    def test_scope_label_single_doc(self):
        qu = _qu(doc_ids=["report_29"])
        label = ClinicalDeterministicRenderer._scope_label(qu)
        self.assertEqual(label, "report_29")

    def test_scope_label_multi_doc(self):
        qu = _qu(doc_ids=["report_10", "report_16", "report_24"])
        label = ClinicalDeterministicRenderer._scope_label(qu)
        self.assertIn("report_10", label)

    def test_scope_label_global(self):
        qu = _qu(doc_ids=[])
        label = ClinicalDeterministicRenderer._scope_label(qu)
        self.assertEqual(label, "global")

    def test_render_dispatcher_compact(self):
        result = R.render([_row()], _qu(), answer_style="compact")
        self.assertEqual(result["mode"], "compact")

    def test_render_dispatcher_detailed(self):
        result = R.render([_row()], _qu(), answer_style="detailed")
        self.assertEqual(result["mode"], "detailed")

    def test_status_unknown_maps_to_non_numerique(self):
        row = _row(status="unknown_xyz")
        label = ClinicalDeterministicRenderer._get_status_fr(row)
        self.assertEqual(label, "valeur non numériquement exploitable")

    def test_no_pii_in_output(self):
        """Output must not contain patient name, DOB patterns."""
        rows = [_row()]
        result = R.render_compact(rows, _qu())
        for pii_pattern in ["Dupont", "né le", "patient :", "M.", "Mme"]:
            self.assertNotIn(pii_pattern, result["text"])

    def test_no_diagnosis_phrase(self):
        rows = [_row(status="above_reference")]
        result = R.render_compact(rows, _qu())
        forbidden = ["diagnostic de", "compatible avec", "évocateur de", "le patient souffre"]
        for phrase in forbidden:
            self.assertNotIn(phrase, result["text"].lower())

    def test_conclusion_starts_with_conclusion_technique(self):
        result = R.render_compact([_row()], _qu())
        self.assertTrue(
            result["conclusion"].startswith("Conclusion technique"),
            f"Conclusion must start with 'Conclusion technique', got: {result['conclusion'][:60]}"
        )

    def test_debug_contains_all_required_keys(self):
        qu = _qu(analytes=["creatinine"], doc_ids=["report_24"])
        result = R.render_compact([_row()], qu, debug=True)
        dbg = result["debug"]
        for key in ("requested_analytes", "requested_doc_ids", "found_count", "displayed_evidences_count"):
            self.assertIn(key, dbg)

    def test_multiple_docs_sources_all_cited(self):
        rows = [
            _row(doc_id="report_10", analyte="CRP"),
            _row(doc_id="report_16", analyte="TSHus"),
            _row(doc_id="report_24", analyte="Créatinine"),
        ]
        result = R.render_detailed(rows, _qu(doc_ids=["report_10", "report_16", "report_24"]))
        for doc in ["report_10", "report_16", "report_24"]:
            self.assertTrue(any(doc in s for s in result["sources"]), f"{doc} missing from sources")


if __name__ == "__main__":
    unittest.main()
