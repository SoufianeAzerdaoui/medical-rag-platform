from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DIR = PROJECT_ROOT / "scripts" / "generation"
if str(GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATION_DIR))

from analyte_aliases import ANALYTE_ALIAS_GROUPS
from analyte_resolver import resolve_requested_analytes
from medical_entity_resolver import (
    canonicalize_analyte,
    get_display_analyte_label,
    is_analyte_match,
)


class TestAnalyteResolver(unittest.TestCase):
    def _resolve_one(self, query: str, available: list[dict] | None = None) -> dict:
        resolved = resolve_requested_analytes(
            query=query,
            available_analytes=available or [{"display_name": k.upper(), "analyte_norm": k} for k in ANALYTE_ALIAS_GROUPS.keys()],
            aliases=ANALYTE_ALIAS_GROUPS,
            max_candidates=5,
        )
        self.assertTrue(resolved)
        return resolved[0]

    def test_parentheses_glucose(self) -> None:
        r = self._resolve_one("Glycémie (Glucose)")
        self.assertEqual(r.get("analyte_norm"), "glucose")

    def test_glycemie_without_accent(self) -> None:
        r = self._resolve_one("glycemie")
        self.assertEqual(r.get("analyte_norm"), "glucose")

    def test_tsh_variants(self) -> None:
        r1 = self._resolve_one("TSH ultrasensible")
        r2 = self._resolve_one("TSH us")
        r3 = self._resolve_one("TSH-us")
        self.assertEqual(r1.get("analyte_norm"), "tshus")
        self.assertEqual(r2.get("analyte_norm"), "tshus")
        self.assertEqual(r3.get("analyte_norm"), "tshus")

    def test_ca_15_3_numeric_kept(self) -> None:
        r = self._resolve_one("CA 15-3")
        self.assertEqual(r.get("analyte_norm"), "ca_15_3")

    def test_pth_intacte(self) -> None:
        r = self._resolve_one("PTH intacte")
        self.assertEqual(r.get("analyte_norm"), "pth_intact")

    def test_gh_sth(self) -> None:
        r1 = self._resolve_one("GH")
        r2 = self._resolve_one("STH")
        self.assertEqual(r1.get("analyte_norm"), "gh_ou_sth")
        self.assertEqual(r2.get("analyte_norm"), "gh_ou_sth")

    def test_phosphatase_alias(self) -> None:
        r1 = self._resolve_one("PAL")
        r2 = self._resolve_one("phosphatase alcaline")
        self.assertEqual(r1.get("analyte_norm"), "phosphatase_alcaline")
        self.assertEqual(r2.get("analyte_norm"), "phosphatase_alcaline")

    def test_dynamic_vocab_without_alias(self) -> None:
        available = [{"display_name": "TEST X BETA", "analyte_norm": "test_x_beta"}]
        r = self._resolve_one("compare TEST X beta", available=available)
        self.assertEqual(r.get("analyte_norm"), "test_x_beta")

    def test_acide_urique_aliases(self) -> None:
        for q in ["acide urique", "uric acid", "urate", "uricémie"]:
            r = self._resolve_one(q)
            self.assertEqual(r.get("analyte_norm"), "acide_urique")

    def test_hdl_aliases(self) -> None:
        for q in ["cholestérol hdl", "hdl cholesterol", "chol hdl"]:
            r = self._resolve_one(q)
            self.assertEqual(r.get("analyte_norm"), "cholesterol_hdl")

    def test_canonicalize_creatinine_variants(self) -> None:
        for value in ["créatinine", "creatinine", "créat", "creat", "créatininémie"]:
            self.assertEqual(canonicalize_analyte(value), "creatinine")

    def test_canonicalize_acide_urique_variants(self) -> None:
        for value in ["acide urique", "uricémie", "uric acid"]:
            self.assertEqual(canonicalize_analyte(value), "acide_urique")

    def test_canonicalize_phosphatase_alcaline_variants(self) -> None:
        for value in ["phosphatase alcaline", "PAL", "ALP"]:
            self.assertEqual(canonicalize_analyte(value), "phosphatase_alcaline")

    def test_tsh_tshus_match_but_keep_source_label(self) -> None:
        row = {
            "analyte": "TSHus",
            "analyte_norm": "tshus",
            "value_raw": "55,00",
            "unit": "mUI/L",
        }
        self.assertTrue(is_analyte_match("TSH", row))
        self.assertTrue(is_analyte_match("tshus", row))
        self.assertEqual(get_display_analyte_label(row), "TSHus")

    def test_cortisol_absent_still_not_found(self) -> None:
        self.assertEqual(canonicalize_analyte("cortisol"), "cortisol")
        row = {"analyte": "CRP", "analyte_norm": "crp"}
        self.assertFalse(is_analyte_match("cortisol", row))

    def test_unknown_analyte_safe_canonicalization(self) -> None:
        self.assertEqual(canonicalize_analyte("analyte_inconnu_xyz"), "analyte_inconnu_xyz")

    def test_alias_matching_does_not_cross_match(self) -> None:
        crp_row = {"analyte": "CRP", "analyte_norm": "crp"}
        creat_row = {"analyte": "Créatinine", "analyte_norm": "creatinine"}
        t4_row = {"analyte": "T4 libre", "analyte_norm": "t4_libre"}
        urea_row = {"analyte": "Urée", "analyte_norm": "uree"}
        self.assertFalse(is_analyte_match("creatinine", crp_row))
        self.assertFalse(is_analyte_match("tsh", t4_row))
        self.assertFalse(is_analyte_match("acide urique", urea_row))
        self.assertFalse(is_analyte_match("crp", creat_row))


    def test_find_compatible_evidence_rows_basic_import(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        result = find_compatible_evidence_rows(["creatinine"], [])
        self.assertIsInstance(result, dict)
        self.assertIn("found_rows", result)
        self.assertIn("not_found_analytes", result)

    def test_no_not_found_when_compatible_evidence_exists(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [{"analyte": "Créatinine", "analyte_norm": "creatinine", "current_value": "12", "unit": "mg/L", "doc_id": "report_29"}]
        result = find_compatible_evidence_rows(["creat"], evidence)
        self.assertTrue(len(result["found_rows"]) > 0)
        self.assertEqual(len(result["not_found_analytes"]), 0)

    def test_report29_creatinine_status_variants(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [{"analyte": "Créatinine", "analyte_norm": "creatinine", "current_value": "9", "unit": "mg/L", "doc_id": "report_29"}]
        result = find_compatible_evidence_rows(["créatininémie"], evidence)
        self.assertTrue(len(result["found_rows"]) > 0)
        self.assertIn("report_29", result["found_rows"][0].get("doc_id", "") or "")

    def test_report16_tsh_request_matches_tshus(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [{"analyte": "TSHus", "analyte_norm": "tshus", "current_value": "5.5", "unit": "mUI/L", "doc_id": "report_16"}]
        result = find_compatible_evidence_rows(["TSH"], evidence)
        self.assertTrue(len(result["found_rows"]) > 0)
        self.assertIn("family", result.get("matching_strategy", ""))

    def test_report24_uricemia_matches_acide_urique(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [{"analyte": "Acide Urique", "analyte_norm": "acide_urique", "current_value": "60", "unit": "mg/L", "doc_id": "report_24"}]
        result = find_compatible_evidence_rows(["uricémie"], evidence)
        self.assertTrue(len(result["found_rows"]) > 0)

    def test_topic_like_request_matches_renal_evidence(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [
            {"analyte": "Créatinine", "analyte_norm": "creatinine", "current_value": "12", "unit": "mg/L", "doc_id": "report_29"}
        ]
        result = find_compatible_evidence_rows(["bilan rénal"], evidence)
        self.assertTrue(len(result["found_rows"]) > 0)
        self.assertIn(result.get("matching_strategy"), {"topic", "mixed"})

    def test_absent_cortisol_remains_not_found(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [{"analyte": "Créatinine", "analyte_norm": "creatinine", "current_value": "12", "unit": "mg/L"}]
        result = find_compatible_evidence_rows(["cortisol"], evidence)
        self.assertEqual(len(result["found_rows"]), 0)
        self.assertIn("cortisol", result["not_found_analytes"])

    def test_tsh_t3_topic_thyroid_but_not_same_analyte(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [{"analyte": "T3 Libre", "analyte_norm": "t3", "current_value": "3.5", "unit": "pg/mL", "doc_id": "report_16"}]
        result = find_compatible_evidence_rows(["TSH"], evidence)
        self.assertEqual(len(result["found_rows"]), 0)

    def test_multiple_doc_ids_scope_filtering(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [
            {"analyte": "creatinine", "analyte_norm": "creatinine", "doc_id": "report_10", "current_value": "10", "unit": "mg/L"},
            {"analyte": "creatinine", "analyte_norm": "creatinine", "doc_id": "report_29", "current_value": "12", "unit": "mg/L"},
            {"analyte": "creatinine", "analyte_norm": "creatinine", "doc_id": "report_24", "current_value": "11", "unit": "mg/L"},
        ]
        result = find_compatible_evidence_rows(["creatinine"], evidence, scope_doc_ids=["report_29"])
        self.assertEqual(len(result["found_rows"]), 1)
        self.assertEqual(result["found_rows"][0]["doc_id"], "report_29")

    def test_partially_found_confidence_score(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [{"analyte": "creatinine", "analyte_norm": "creatinine", "current_value": "12", "unit": "mg/L"}]
        result = find_compatible_evidence_rows(["creatinine", "cortisol"], evidence)
        self.assertTrue(result["partially_found"])
        self.assertEqual(result["confidence_score"], 0.5)

    def test_no_hallucination_on_different_fields(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [{"parameter": "CREAT", "analyte": "XYZ", "analyte_norm": "xyz", "current_value": "10", "unit": "mg/L"}]
        result = find_compatible_evidence_rows(["creatinine"], evidence)
        self.assertEqual(len(result["found_rows"]), 0)

    def test_all_label_fields_checked(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [
            {
                "analyte": "code_123",
                "analyte_norm": "code_123",
                "display_name": "Créatinine",
                "analytical_label": "Creat",
                "current_value": "12",
                "unit": "mg/L"
            }
        ]
        result = find_compatible_evidence_rows(["creatinine"], evidence)
        self.assertTrue(len(result["found_rows"]) > 0)

    def test_confidence_score_all_found(self) -> None:
        from medical_entity_resolver import find_compatible_evidence_rows
        evidence = [
            {"analyte": "creatinine", "analyte_norm": "creatinine", "current_value": "12", "unit": "mg/L"},
            {"analyte": "TSHus", "analyte_norm": "tshus", "current_value": "5.5", "unit": "mUI/L"},
        ]
        result = find_compatible_evidence_rows(["creatinine", "TSH"], evidence)
        self.assertEqual(result["confidence_score"], 1.0)
        self.assertFalse(result["partially_found"])


if __name__ == "__main__":
    unittest.main()
